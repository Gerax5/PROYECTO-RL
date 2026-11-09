"""Entrenamiento del supervisor con (Recurrent)PPO de stable-baselines3 (sección 6.5).

Cada corrida deja en `run_dir`:
    config.json         la `TrainConfig` usada
    tb/                 TensorBoard (entrenamiento y métricas de evaluación eval/*)
    history.json        métricas de cada evaluación periódica
    last/, best/, final/   model.zip + vecnormalize.pkl
`best/` es la mejor evaluación con tasa de falsos positivos <= `max_fp_rate` (por F1).
"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

from env import CasinoTableEnv, RewardConfig, SeatEvent, TableConfig
from env.curriculum import Curriculum
from eval import evaluate
from players import StealthConfig

from .policy import SeatSetEncoder

ALGOS: dict[str, tuple[type[BaseAlgorithm], str]] = {
    "recurrent_ppo": (RecurrentPPO, "MultiInputLstmPolicy"),
    "ppo": (PPO, "MultiInputPolicy"),
}


@dataclass(frozen=True)
class TrainConfig:
    algo: str = "recurrent_ppo"         # "recurrent_ppo" | "ppo" (sin memoria, para comparar)
    total_timesteps: int = 5_000_000
    n_envs: int = 8
    vec_env: str = "subproc"            # "subproc" | "dummy" (un solo proceso; tests)
    rounds_per_episode: int = 600
    seed: int = 0
    # PPO
    learning_rate: float = 3e-4
    n_steps: int = 2048                 # por mesa
    # Con 256 y 10 épocas la LSTM iba a ~140 pasos/s; así va a ~310 (PPO a ~1200).
    batch_size: int = 2048
    n_epochs: int = 5
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    # Red
    seat_encoder: tuple[int, ...] = (64, 64)
    global_encoder: tuple[int, ...] = (32,)
    lstm_hidden: int = 128
    head: tuple[int, ...] = (64,)
    # Mesa
    disguise: float | None = None       # None = el de TableConfig
    equal_sessions: bool = True         # misma duración media para todos (evita reward hacking)
    session_mean: float = 100.0
    curriculum: bool = False
    # Evaluación periódica
    eval_freq: int = 100_000            # en timesteps totales
    eval_episodes: int = 10
    eval_seed: int = 10_000
    max_fp_rate: float = 0.05
    reward: RewardConfig = field(default_factory=RewardConfig)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TrainConfig":
        d = json.loads(path.read_text(encoding="utf-8"))
        d["reward"] = RewardConfig(**d["reward"])
        for key in ("seat_encoder", "global_encoder", "head"):
            d[key] = tuple(d[key])
        return cls(**d)


def build_table_config(cfg: TrainConfig, base: TableConfig | None = None) -> TableConfig:
    table = base if base is not None else TableConfig()
    if cfg.disguise is not None:
        table = replace(table, stealth=StealthConfig.from_disguise(cfg.disguise))
    if cfg.equal_sessions:
        table = replace(table, session_mean_rounds={k: cfg.session_mean for k in table.population_mix})
    return table


def make_vec_env(
    table_configs: list[TableConfig],
    reward: RewardConfig,
    rounds_per_episode: int,
    seed: int,
    kind: str = "subproc",
) -> VecMonitor:
    """Una mesa por configuración; sin `Monitor` por mesa para que `set_attr` llegue al ambiente."""
    fns = [partial(CasinoTableEnv, t, reward, rounds_per_episode) for t in table_configs]
    venv = SubprocVecEnv(fns) if kind == "subproc" and len(fns) > 1 else DummyVecEnv(fns)
    venv.seed(seed)
    return VecMonitor(venv)


def make_model(cfg: TrainConfig, venv, tensorboard_log: str | None = None) -> BaseAlgorithm:
    algo_cls, policy = ALGOS[cfg.algo]
    policy_kwargs: dict = {
        "features_extractor_class": SeatSetEncoder,
        "features_extractor_kwargs": {
            "seat_hidden": cfg.seat_encoder, "global_hidden": cfg.global_encoder,
        },
        "net_arch": {"pi": list(cfg.head), "vf": list(cfg.head)},
    }
    if cfg.algo == "recurrent_ppo":
        policy_kwargs["lstm_hidden_size"] = cfg.lstm_hidden
    return algo_cls(
        policy,
        venv,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        seed=cfg.seed,
        device="cpu",
        verbose=1,
    )


# ---------------------------------------------------------------- política entrenada

class SB3Supervisor:
    """Adapta un modelo entrenado a las políticas de `eval` (`act(obs)`) y del
    adversario (`act(obs, events)`). Normaliza la observación igual que en el entrenamiento."""

    def __init__(self, model: BaseAlgorithm, normalizer: VecNormalize | None, deterministic: bool = True):
        self.model = model
        self.normalizer = normalizer
        self.deterministic = deterministic
        self.reset()

    def reset(self) -> None:
        self._state = None
        self._episode_start = np.ones((1,), dtype=bool)

    def act(self, obs: dict[str, np.ndarray], events: list[SeatEvent] | None = None) -> np.ndarray:
        if self.normalizer is not None:
            obs = self.normalizer.normalize_obs(obs)
        action, self._state = self.model.predict(
            obs, state=self._state, episode_start=self._episode_start,
            deterministic=self.deterministic,
        )
        self._episode_start = np.zeros((1,), dtype=bool)
        return np.asarray(action)


def save_checkpoint(model: BaseAlgorithm, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save(path / "model.zip")
    normalizer = model.get_vec_normalize_env()
    if normalizer is not None:
        normalizer.save(str(path / "vecnormalize.pkl"))


def checkpoint_dir(run_dir: Path, which: str = "best") -> Path:
    """`which` si existe; si no, `final` y luego `last`."""
    for name in (which, "final", "last"):
        if (run_dir / name / "model.zip").exists():
            return run_dir / name
    raise FileNotFoundError(f"no hay checkpoints en {run_dir}")


def supervisor_from_checkpoint(path: str | Path, algo: str, threads: int | None = None) -> SB3Supervisor:
    """Serializable con `functools.partial`, para pasarlo al adversario en otros procesos."""
    if threads is not None:
        torch.set_num_threads(threads)
    path = Path(path)
    model = ALGOS[algo][0].load(path / "model.zip", device="cpu")
    normalizer = None
    if (path / "vecnormalize.pkl").exists():
        with open(path / "vecnormalize.pkl", "rb") as f:
            normalizer = pickle.load(f)
        normalizer.training = False
    return SB3Supervisor(model, normalizer)


def load_supervisor(run_dir: str | Path, which: str = "best", threads: int | None = None) -> SB3Supervisor:
    run_dir = Path(run_dir)
    cfg = TrainConfig.load(run_dir / "config.json")
    return supervisor_from_checkpoint(checkpoint_dir(run_dir, which), cfg.algo, threads)


# ---------------------------------------------------------------- evaluación periódica

class SupervisorEvalCallback(BaseCallback):
    """Evalúa con semillas fijas, registra eval/* en TensorBoard, guarda `best/` y
    hace avanzar el curriculum."""

    def __init__(self, cfg: TrainConfig, run_dir: Path, table_config: TableConfig,
                 curriculum: Curriculum | None = None):
        super().__init__()
        self.cfg, self.run_dir = cfg, run_dir
        self.table_config = table_config
        self.curriculum = curriculum
        self.best_f1 = -math.inf
        self.history: list[dict] = []
        self._last_eval = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval >= self.cfg.eval_freq:
            self.run_evaluation()
        return True

    def _on_training_end(self) -> None:
        if self.num_timesteps > self._last_eval:
            self.run_evaluation()

    def run_evaluation(self) -> dict[str, float]:
        cfg = self.cfg
        elapsed, self._last_eval = self.num_timesteps - self._last_eval, self.num_timesteps
        policy = SB3Supervisor(self.model, self.model.get_vec_normalize_env())
        seeds = list(range(cfg.eval_seed, cfg.eval_seed + cfg.eval_episodes))
        s = evaluate(policy, seeds, self.table_config, cfg.reward, cfg.rounds_per_episode).summary()
        for key, value in s.items():
            self.logger.record(f"eval/{key}", value)

        stage = self.curriculum.index if self.curriculum is not None else None
        save_checkpoint(self.model, self.run_dir / "last")
        if s["fp_rate"] <= cfg.max_fp_rate and s["f1_counters"] > self.best_f1:
            self.best_f1 = s["f1_counters"]
            save_checkpoint(self.model, self.run_dir / "best")

        if self.curriculum is not None:
            self.logger.record("curriculum/stage", stage)
            if self.curriculum.update(s["recall_stealth"], s["fp_rate"], elapsed):
                # La nueva etapa se aplica a cada mesa en su siguiente reset.
                self.table_config = self.curriculum.table_config()
                self.training_env.set_attr("table_config", self.table_config)
                self.best_f1 = -math.inf     # las etapas más difíciles no son comparables

        self.history.append({"timesteps": self.num_timesteps, "stage": stage, **s})
        (self.run_dir / "history.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        print(f"[eval {self.num_timesteps:,}] rec.cont {s['recall_counter']:.0%} "
              f"rec.disi {s['recall_stealth']:.0%} FP {s['fp_rate']:.1%} "
              f"F1 {s['f1_counters']:.2f} R/100 {s['reward_per_100']:+.1f}"
              + (f" etapa {stage}" if stage is not None else ""), flush=True)
        return s


def train(cfg: TrainConfig, run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.json")

    base = build_table_config(cfg)
    curriculum = Curriculum(base) if cfg.curriculum else None
    table = curriculum.table_config() if curriculum is not None else base

    venv = make_vec_env([table] * cfg.n_envs, cfg.reward, cfg.rounds_per_episode, cfg.seed, cfg.vec_env)
    venv = VecNormalize(venv, gamma=cfg.gamma, clip_obs=10.0)
    model = make_model(cfg, venv, tensorboard_log=str(run_dir / "tb"))
    callback = SupervisorEvalCallback(cfg, run_dir, table, curriculum)
    try:
        model.learn(cfg.total_timesteps, callback=callback, tb_log_name=cfg.algo)
        save_checkpoint(model, run_dir / "final")
    finally:
        venv.close()
    return run_dir
