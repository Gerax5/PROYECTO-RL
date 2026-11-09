"""Barrido de `expel_casual`: curva de trade-off detección vs. falsos positivos (sección 6.4).

Entrena una corrida por valor y evalúa su mejor modelo con las mismas semillas.
Escribe `<out>/sweep.csv`.

Uso:
    python scripts/sweep_reward.py --values -5 -10 -25 -50 --timesteps 1000000
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval import evaluate  # noqa: E402
from supervisor.train import TrainConfig, build_table_config, load_supervisor, train  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--values", type=float, nargs="+", default=[-5, -10, -25, -50])
    p.add_argument("--algo", default="recurrent_ppo")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--episodes", type=int, default=20, help="episodios de evaluación")
    p.add_argument("--seed", type=int, default=1000, help="primera semilla de evaluación")
    p.add_argument("--out", type=Path, default=ROOT / "runs" / "sweep_expel_casual")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    rows = []
    seeds = list(range(args.seed, args.seed + args.episodes))
    for value in args.values:
        base = TrainConfig()
        cfg = replace(base, algo=args.algo, total_timesteps=args.timesteps, n_envs=args.n_envs,
                      reward=replace(base.reward, expel_casual=value))
        run = train(cfg, args.out / f"expel_casual_{value:g}")
        s = evaluate(load_supervisor(run), seeds, build_table_config(cfg), cfg.reward,
                     cfg.rounds_per_episode).summary()
        rows.append({"expel_casual": value, **s})
        print(f"expel_casual={value:g}: rec.cont {s['recall_counter']:.0%} "
              f"rec.disi {s['recall_stealth']:.0%} FP {s['fp_rate']:.1%}", flush=True)

        with open(args.out / "sweep.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
