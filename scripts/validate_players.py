"""Prueba por simulación: el contador gana dinero y el casual lo pierde.

Cada escenario se reparte en semillas independientes entre varios procesos.
Un veredicto sólo es PASA si todo el intervalo del 95% está del lado esperado.
Termina con código 1 si falla alguno de los veredictos exigidos.

Uso:
    python scripts/validate_players.py --rounds 2000000
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import reduce
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from players import (  # noqa: E402
    BATCH_ROUNDS,
    BET_STYLES,
    BasePlayer,
    CasualConfig,
    CasualPlayer,
    CounterConfig,
    CounterPlayer,
    PlayerStats,
    StealthConfig,
    StealthCounterPlayer,
    simulate_table,
)

DISGUISE_LEVELS = (0.0, 0.3, 0.6, 0.9, 1.0)


def build(spec: tuple, rng: np.random.Generator) -> BasePlayer:
    kind, value = spec
    if kind == "casual":
        return CasualPlayer(CasualConfig(bet_style=value), rng=rng)
    if kind == "counter":
        return CounterPlayer(CounterConfig(wonging=value), rng=rng)
    if kind == "stealth":
        cfg = StealthConfig.from_disguise(value)
        return StealthCounterPlayer(cfg, rng=rng)
    raise ValueError(kind)


def run_chunk(args: tuple) -> list[PlayerStats]:
    specs, rounds, entropy = args
    seq = np.random.SeedSequence(entropy)
    table_seq, *player_seqs = seq.spawn(len(specs) + 1)
    players = [build(s, np.random.default_rng(q)) for s, q in zip(specs, player_seqs)]
    seed = int(table_seq.generate_state(1)[0])
    return simulate_table(players, rounds, seed)


def run_scenario(pool, specs, rounds, chunks, entropy) -> list[PlayerStats]:
    per_chunk = max(rounds // chunks // BATCH_ROUNDS, 2) * BATCH_ROUNDS
    jobs = [(specs, per_chunk, (entropy, i)) for i in range(chunks)]
    results = list(pool.map(run_chunk, jobs))
    return [reduce(PlayerStats.merge, seat) for seat in zip(*results)]


def row(label: str, s: PlayerStats) -> str:
    lo, hi = s.ci95_per_100
    return (f"{label:38s} {s.edge_pct:+7.2f}%  {s.units_per_100:+7.2f} u/100  "
            f"IC95 [{lo:+6.2f}, {hi:+6.2f}]  {s.rounds_played / s.rounds:5.0%} jugadas")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=2_000_000, help="rondas por escenario")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    scenarios: list[tuple[str, list[tuple], int]] = []
    for style in BET_STYLES:
        scenarios.append((f"Casual {style}", [("casual", style)], 0))
    scenarios.append(("Contador abierto", [("counter", False)], 0))
    for d in DISGUISE_LEVELS:
        scenarios.append((f"Disimulado disguise={d:.1f}", [("stealth", d)], 0))
    scenarios.append(("Contador con wonging (junto a un casual)",
                      [("counter", True), ("casual", "flat_noise")], 0))
    mixed = [("casual", "hot_streak"), ("counter", False), ("casual", "martingale"),
             ("stealth", 0.6), ("casual", "flat_noise")]
    mixed_labels = ["Mesa mixta: casual hot_streak", "Mesa mixta: contador",
                    "Mesa mixta: casual martingale", "Mesa mixta: disimulado 0.6",
                    "Mesa mixta: casual flat_noise"]

    print(f"{args.rounds:,} rondas por escenario, {args.workers} procesos, semilla {args.seed}\n")
    casual_ok = counter_ok = True
    stealth: dict[str, PlayerStats] = {}

    with ProcessPoolExecutor(args.workers) as pool:
        chunks = args.workers * 2
        for i, (label, specs, seat) in enumerate(scenarios):
            s = run_scenario(pool, specs, args.rounds, chunks, (args.seed, i))[seat]
            print(row(label, s), flush=True)
            lo, hi = s.ci95_per_100
            if label.startswith("Casual"):
                casual_ok &= hi < 0
            elif label.startswith("Contador"):
                counter_ok &= lo > 0
            elif label.startswith("Disimulado"):
                stealth[label] = s

        print()
        table = run_scenario(pool, mixed, args.rounds, chunks, (args.seed, len(scenarios)))
        for label, s in zip(mixed_labels, table):
            print(row(label, s), flush=True)
            lo, hi = s.ci95_per_100
            if "casual" in label:
                casual_ok &= hi < 0
            elif "contador" in label:
                counter_ok &= lo > 0

    print()
    print(f"El casual pierde dinero (todo el IC95 < 0):  {'PASA' if casual_ok else 'FALLA'}")
    print(f"El contador gana dinero (todo el IC95 > 0):  {'PASA' if counter_ok else 'FALLA'}")
    for label, s in stealth.items():
        lo, hi = s.ci95_per_100
        verdict = "gana" if lo > 0 else "pierde" if hi < 0 else "no distinguible de 0"
        print(f"{label}: {verdict} (informativo)")
    return 0 if casual_ok and counter_ok else 1


if __name__ == "__main__":
    sys.exit(main())
