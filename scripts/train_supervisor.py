"""Entrena el supervisor con RecurrentPPO (o PPO sin memoria) y TensorBoard.

Uso:
    python scripts/train_supervisor.py --run runs/rppo_base
    python scripts/train_supervisor.py --run runs/rppo_curr --curriculum
    python scripts/train_supervisor.py --run runs/ppo_base --algo ppo --timesteps 2000000
    tensorboard --logdir runs
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from env import RewardConfig  # noqa: E402
from supervisor.train import ALGOS, TrainConfig, train  # noqa: E402


def parse_args(argv=None) -> tuple[TrainConfig, Path]:
    d = TrainConfig()
    r = RewardConfig()
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="carpeta de la corrida (p.ej. runs/rppo)")
    p.add_argument("--algo", choices=sorted(ALGOS), default=d.algo)
    p.add_argument("--timesteps", type=int, default=d.total_timesteps)
    p.add_argument("--n-envs", type=int, default=d.n_envs)
    p.add_argument("--vec-env", choices=("subproc", "dummy"), default=d.vec_env)
    p.add_argument("--rounds", type=int, default=d.rounds_per_episode, help="rondas por episodio")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--lr", type=float, default=d.learning_rate)
    p.add_argument("--n-steps", type=int, default=d.n_steps)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--n-epochs", type=int, default=d.n_epochs)
    p.add_argument("--gamma", type=float, default=d.gamma)
    p.add_argument("--ent-coef", type=float, default=d.ent_coef)
    p.add_argument("--lstm-hidden", type=int, default=d.lstm_hidden)
    p.add_argument("--disguise", type=float, default=d.disguise)
    p.add_argument("--curriculum", action="store_true")
    p.add_argument("--unequal-sessions", action="store_true",
                   help="usar la duración de sesión distinta por tipo de TableConfig")
    p.add_argument("--eval-freq", type=int, default=d.eval_freq)
    p.add_argument("--eval-episodes", type=int, default=d.eval_episodes)
    p.add_argument("--max-fp-rate", type=float, default=d.max_fp_rate)
    p.add_argument("--expel-casual", type=float, default=r.expel_casual)
    p.add_argument("--expel-counter", type=float, default=r.expel_counter)
    p.add_argument("--expel-stealth", type=float, default=r.expel_stealth)
    a = p.parse_args(argv)

    reward = replace(r, expel_casual=a.expel_casual, expel_counter=a.expel_counter,
                     expel_stealth=a.expel_stealth)
    cfg = replace(
        d, algo=a.algo, total_timesteps=a.timesteps, n_envs=a.n_envs, vec_env=a.vec_env,
        rounds_per_episode=a.rounds, seed=a.seed, learning_rate=a.lr, n_steps=a.n_steps,
        batch_size=a.batch_size, n_epochs=a.n_epochs, gamma=a.gamma, ent_coef=a.ent_coef,
        lstm_hidden=a.lstm_hidden, disguise=a.disguise, curriculum=a.curriculum,
        equal_sessions=not a.unequal_sessions, eval_freq=a.eval_freq,
        eval_episodes=a.eval_episodes, max_fp_rate=a.max_fp_rate, reward=reward,
    )
    return cfg, a.run


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg, run = parse_args()
    print(f"Entrenando {cfg.algo} {cfg.total_timesteps:,} pasos con {cfg.n_envs} mesas -> {run}")
    train(cfg, run)
    print(f"Listo. Modelos en {run}/best y {run}/final; métricas en {run}/history.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
