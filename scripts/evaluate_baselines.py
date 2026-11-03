"""Mide los baselines sin aprendizaje sobre episodios con semillas fijas.

Es la vara contra la que se compara el supervisor RL: misma mesa, mismas semillas.

Uso:
    python scripts/evaluate_baselines.py --episodes 20 --rounds 600
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from env import TableConfig  # noqa: E402
from eval import evaluate  # noqa: E402
from players import StealthConfig  # noqa: E402
from supervisor import HeuristicSupervisor, NeverExpel  # noqa: E402

COLUMNS = (
    ("recall_counter", "rec.cont", "{:8.1%}"),
    ("recall_stealth", "rec.disi", "{:8.1%}"),
    ("fp_rate", "FP", "{:6.1%}"),
    ("precision", "prec.", "{:6.1%}"),
    ("f1_counters", "F1", "{:5.2f}"),
    ("detection_rounds", "rondas", "{:6.1f}"),
    ("counter_net_per_100", "cont.u/100", "{:+10.2f}"),
    ("house_net_per_100", "casa u/100", "{:+10.2f}"),
    ("reward_per_100", "R/100", "{:+8.2f}"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=600, help="rondas por episodio")
    parser.add_argument("--seed", type=int, default=1000, help="primera semilla de evaluación")
    parser.add_argument("--disguise", type=float, default=None,
                        help="disguise_level del disimulado (por defecto el de TableConfig)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = TableConfig()
    if args.disguise is not None:
        cfg = replace(cfg, stealth=StealthConfig.from_disguise(args.disguise))
    seeds = list(range(args.seed, args.seed + args.episodes))

    policies = {"nunca expulsa": NeverExpel()}
    for threshold in (0.4, 0.5, 0.6, 0.7):
        policies[f"heurística corr>{threshold} n>30"] = HeuristicSupervisor(threshold, 30)
    policies["heurística spearman>0.5 n>30"] = HeuristicSupervisor(0.5, 30, "rank_corr_bet_tc")

    print(f"{args.episodes} episodios x {args.rounds} rondas, semillas {seeds[0]}..{seeds[-1]}\n")
    print(f"{'política':32s} " + " ".join(h for _, h, _ in COLUMNS))
    for name, policy in policies.items():
        m = evaluate(policy, seeds, cfg, rounds_per_episode=args.rounds)
        s = m.summary()
        cells = " ".join(fmt.format(s[key]).rjust(len(h)) for key, h, fmt in COLUMNS)
        print(f"{name:32s} {cells}", flush=True)

    print(f"\njugadores vistos: {dict(m.seen)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
