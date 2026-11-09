"""Reentrenamiento continuo del supervisor contra el adversario evolutivo (sección 7.3).

Parte de una corrida ya entrenada con `train_supervisor.py`.

Uso:
    python scripts/online_retrain.py --from runs/rppo_base --out runs/online --cycles 5
    python scripts/online_retrain.py --from runs/rppo_base --out runs/online_noadv --no-adversary
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retraining.adversary import AdversaryConfig  # noqa: E402
from retraining.online_loop import OnlineConfig, run_online  # noqa: E402


def main() -> int:
    d = OnlineConfig()
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--from", dest="initial", type=Path, required=True, help="corrida inicial")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cycles", type=int, default=d.cycles)
    p.add_argument("--finetune-timesteps", type=int, default=d.finetune_timesteps)
    p.add_argument("--n-envs", type=int, default=d.n_envs)
    p.add_argument("--replay-ratio-old", type=float, default=d.replay_ratio_old)
    p.add_argument("--no-adversary", action="store_true")
    p.add_argument("--population", type=int, default=d.adversary.population)
    p.add_argument("--generations", type=int, default=d.adversary.generations)
    p.add_argument("--ev-rounds", type=int, default=d.adversary.ev_rounds)
    p.add_argument("--val-episodes", type=int, default=d.val_episodes)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--seed", type=int, default=d.seed)
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    adversary = AdversaryConfig(population=a.population, generations=a.generations, ev_rounds=a.ev_rounds)
    cfg = replace(d, cycles=a.cycles, finetune_timesteps=a.finetune_timesteps, n_envs=a.n_envs,
                  replay_ratio_old=a.replay_ratio_old, use_adversary=not a.no_adversary,
                  adversary=adversary, val_episodes=a.val_episodes, seed=a.seed)
    with ProcessPoolExecutor(a.workers) as pool:
        out = run_online(a.initial, a.out, cfg, pool.map)
    print(f"Listo. Campeón en {out}/best; historial en {out}/history.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
