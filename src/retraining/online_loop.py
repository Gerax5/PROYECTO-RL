"""Reentrenamiento continuo con arms race (sección 7.3).

Cada ciclo:
    1. El adversario evoluciona camuflajes contra el campeón actual (opcional).
    2. Las mesas de fine-tuning mezclan camuflajes nuevos e históricos (`ScenarioBuffer`).
    3. Fine-tuning del campeón durante `finetune_timesteps`.
    4. Validación en un conjunto FIJO: mismas semillas y niveles de disfraz en cada ciclo.
    5. Se promueve si el F1 mejora en `min_improvement` sin pasar `max_fp_rate`.

`out_dir` queda como una corrida más (config.json + best/), así que el campeón final se
carga con `load_supervisor(out_dir)` y se evalúa con `scripts/evaluate_baselines.py`.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path

from stable_baselines3.common.vec_env import VecNormalize

from eval import SupervisorMetrics, evaluate
from supervisor.train import (
    ALGOS,
    TrainConfig,
    build_table_config,
    checkpoint_dir,
    make_vec_env,
    save_checkpoint,
    supervisor_from_checkpoint,
)

from .adversary import AdversaryConfig, Evolver
from .buffer import ScenarioBuffer


@dataclass(frozen=True)
class OnlineConfig:
    cycles: int = 5
    finetune_timesteps: int = 500_000
    n_envs: int = 8
    vec_env: str = "subproc"
    replay_ratio_old: float = 0.3
    buffer_size: int = 10
    use_adversary: bool = True
    adversary: AdversaryConfig = field(
        default_factory=lambda: AdversaryConfig(population=16, generations=5, ev_rounds=100_000)
    )
    val_disguises: tuple[float, ...] = (0.0, 0.3, 0.6, 0.9)
    val_episodes: int = 5               # por nivel de disfraz
    val_seed: int = 50_000
    min_improvement: float = 0.01
    max_fp_rate: float = 0.05
    seed: int = 0


def validate(checkpoint: Path, train_cfg: TrainConfig, cfg: OnlineConfig) -> dict[str, float]:
    """Mismas semillas y niveles en cada ciclo, para comparar versiones."""
    policy = supervisor_from_checkpoint(checkpoint, train_cfg.algo)
    seeds = list(range(cfg.val_seed, cfg.val_seed + cfg.val_episodes))
    total, per_level = SupervisorMetrics(), {}
    for level in cfg.val_disguises:
        table = build_table_config(replace(train_cfg, disguise=level))
        m = evaluate(policy, seeds, table, train_cfg.reward, train_cfg.rounds_per_episode)
        per_level[f"recall_stealth_{level:.1f}"] = m.recall("stealth")
        total = total.merge(m)
    return {**total.summary(), **per_level}


def finetune(checkpoint: Path, train_cfg: TrainConfig, cfg: OnlineConfig, tables, timesteps: int,
             out: Path, seed: int, tb_dir: Path) -> None:
    venv = make_vec_env(tables, train_cfg.reward, train_cfg.rounds_per_episode, seed, cfg.vec_env)
    try:
        venv = VecNormalize.load(str(checkpoint / "vecnormalize.pkl"), venv)
        venv.training = True
        model = ALGOS[train_cfg.algo][0].load(checkpoint / "model.zip", env=venv, device="cpu")
        model.tensorboard_log = str(tb_dir)
        model.learn(timesteps, reset_num_timesteps=False, tb_log_name="online")
        save_checkpoint(model, out)
    finally:
        venv.close()


def run_online(initial_run: str | Path, out_dir: str | Path, cfg: OnlineConfig | None = None,
               map_fn=map) -> Path:
    cfg = cfg if cfg is not None else OnlineConfig()
    initial_run, out_dir = Path(initial_run), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = TrainConfig.load(initial_run / "config.json")
    train_cfg.save(out_dir / "config.json")
    base = build_table_config(train_cfg)

    champion = checkpoint_dir(initial_run, "best")
    champion_score = validate(champion, train_cfg, cfg)
    history = [{"cycle": 0, "promoted": True, "checkpoint": str(champion), **champion_score}]
    print(f"[ciclo 0] F1 {champion_score['f1_counters']:.3f} FP {champion_score['fp_rate']:.1%}", flush=True)

    buffer = ScenarioBuffer(cfg.buffer_size, cfg.replay_ratio_old)
    evolver = None
    if cfg.use_adversary:
        evolver = Evolver(lambda: None, cfg.adversary, start=base.stealth, seed=cfg.seed)

    for cycle in range(1, cfg.cycles + 1):
        record: dict = {"cycle": cycle}
        if evolver is not None:
            evolver.make_supervisor = partial(supervisor_from_checkpoint, champion, train_cfg.algo, 1)
            adv = evolver.run_cycle(map_fn)
            buffer.add(adv.stealth)
            record |= {"adversary_ev": adv.ev_per_100, "adversary_expel_rate": adv.expel_rate,
                       "adversary_fitness": adv.fitness}

        candidate = out_dir / f"cycle_{cycle}"
        tables = buffer.table_configs(cfg.n_envs, base)
        finetune(champion, train_cfg, cfg, tables, cfg.finetune_timesteps, candidate,
                 cfg.seed + cycle, out_dir / "tb")
        score = validate(candidate, train_cfg, cfg)
        promoted = (score["fp_rate"] <= cfg.max_fp_rate
                    and score["f1_counters"] >= champion_score["f1_counters"] + cfg.min_improvement)
        if promoted:
            champion, champion_score = candidate, score
            shutil.copytree(candidate, out_dir / "best", dirs_exist_ok=True)
        buffer.end_cycle()

        history.append({**record, "promoted": promoted, "checkpoint": str(champion), **score})
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"[ciclo {cycle}] F1 {score['f1_counters']:.3f} FP {score['fp_rate']:.1%} "
              f"{'PROMOVIDO' if promoted else 'descartado'}", flush=True)

    if not (out_dir / "best" / "model.zip").exists():
        # Ningún ciclo mejoró: el campeón sigue siendo el inicial.
        shutil.copytree(champion, out_dir / "best", dirs_exist_ok=True)
    return out_dir
