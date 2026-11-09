"""Evoluciona disimulados contra el supervisor heurístico y mide si mejoraron.

El camuflaje inicial y el final se comparan con semillas que la evolución no usó,
con más rondas y episodios que durante la búsqueda.

Uso:
    python scripts/run_adversary.py --cycles 2 --population 16 --generations 5
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from functools import partial, reduce
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from players import PlayerStats, StealthConfig, StealthCounterPlayer, simulate_table  # noqa: E402
from retraining.adversary import (  # noqa: E402
    AdversaryConfig,
    CorrelationSupervisor,
    Evolver,
    measure_expel_rate,
)


def _ev_chunk(job: tuple) -> PlayerStats:
    stealth, rounds, seed = job
    player = StealthCounterPlayer(stealth, rng=np.random.default_rng(seed))
    return simulate_table([player], rounds, seed)[0]


def holdout(pool, label, stealth, cfg, workers, rounds, seed, make_supervisor) -> None:
    jobs = [(stealth, rounds // workers // 1000 * 1000, seed + i) for i in range(workers)]
    s = reduce(PlayerStats.merge, pool.map(_ev_chunk, jobs))
    lo, hi = s.ci95_per_100
    long_cfg = replace(cfg, episodes=40)
    rate = measure_expel_rate(stealth, make_supervisor(), long_cfg, seed)
    fields = ", ".join(f"{k}={v:.2f}" for k, v in asdict(stealth).items())
    print(f"{label:10s} EV {s.units_per_100:+.2f} u/100 IC95 [{lo:+.2f}, {hi:+.2f}]  "
          f"expulsión {rate:.0%}  aptitud {s.units_per_100 - cfg.expelled_penalty * rate:+.2f}\n"
          f"           {fields}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--ev-rounds", type=int, default=100_000)
    parser.add_argument("--holdout-rounds", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--supervisor", type=Path, default=None,
                        help="corrida entrenada (scripts/train_supervisor.py); por defecto la heurística")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = AdversaryConfig(
        population=args.population, generations=args.generations, ev_rounds=args.ev_rounds
    )
    make_supervisor = CorrelationSupervisor
    if args.supervisor is not None:
        from supervisor.train import load_supervisor

        make_supervisor = partial(load_supervisor, args.supervisor, "best", 1)
    start = StealthConfig.from_disguise(0.3)
    evolver = Evolver(make_supervisor, cfg, start=start, seed=args.seed)
    holdout_seed = args.seed + 777_000_000

    with ProcessPoolExecutor(args.workers) as pool:
        holdout(pool, "Inicial", start, cfg, args.workers, args.holdout_rounds, holdout_seed,
                make_supervisor)
        for cycle in range(1, args.cycles + 1):
            r = evolver.run_cycle(pool.map)
            print(f"Ciclo {cycle}: aptitud reevaluada {r.fitness:+.2f} "
                  f"(EV {r.ev_per_100:+.2f}, expulsión {r.expel_rate:.0%})", flush=True)
        holdout(pool, "Final", evolver.best, cfg, args.workers, args.holdout_rounds, holdout_seed,
                make_supervisor)


if __name__ == "__main__":
    main()
