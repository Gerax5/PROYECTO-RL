"""Valida el motor: ventaja de la estrategia básica (apuesta plana) y de un
contador Hi-Lo con Illustrious 18 y rampa 1-2-4-8-12.

Referencia (6 mazos, S17, DAS, 3:2, sin surrender, 75% de penetración):
    estrategia básica ≈ -0.4% ; contador 1-12 ≈ +1 unidad mínima por cada 100 rondas.

Uso:
    python scripts/simulate_ev.py --rounds 300000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blackjack import (  # noqa: E402
    Action,
    Rules,
    Shoe,
    basic_strategy,
    counter_strategy,
    hi_lo_value,
    play_round,
    take_insurance,
    true_count,
)


def bet_ramp(tc: float) -> float:
    """Rampa típica de un contador: 1 unidad con TC <= 1, luego 2-4-8-12."""
    return {2: 2.0, 3: 4.0, 4: 8.0}.get(int(np.floor(tc)), 12.0 if tc >= 5 else 1.0)


def simulate(rounds: int, counter: bool, seed: int, rules: Rules) -> tuple[float, float, int]:
    shoe = Shoe(n_decks=6, penetration=0.75, rng=np.random.default_rng(seed))
    rc = 0
    deviations = 0

    def on_card(card: int) -> None:
        nonlocal rc
        rc += hi_lo_value(card)

    def tc() -> float:
        return true_count(rc, shoe.decks_remaining)

    def decide(seat, hand, up, allowed):
        nonlocal deviations
        kw = dict(
            can_double=Action.DOUBLE in allowed,
            can_split=Action.SPLIT in allowed,
            can_surrender=Action.SURRENDER in allowed,
        )
        if not counter:
            return basic_strategy(hand, up, rules, **kw)
        action, _ = counter_strategy(hand, up, tc(), rules, **kw)
        if action != basic_strategy(hand, up, rules, **kw):
            deviations += 1
        return action

    def insure(seat, hand, up):
        return counter and take_insurance(tc())

    net = wagered = 0.0
    for _ in range(rounds):
        if shoe.needs_shuffle:
            shoe.shuffle()
            rc = 0
        bet = bet_ramp(tc()) if counter else 1.0
        res = play_round(shoe, rules, {0: bet}, decide, insure, on_card)
        net += res.seats[0].net
        wagered += bet
    return net, wagered, deviations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rules = Rules()
    sys.stdout.reconfigure(encoding="utf-8")

    for name, counter in [("Estrategia básica (plana)", False), ("Contador Hi-Lo 1-12 + I18", True)]:
        t0 = time.time()
        net, wagered, devs = simulate(args.rounds, counter, args.seed, rules)
        edge = 100 * net / wagered
        per100 = 100 * net / args.rounds
        print(f"{name:28s} ventaja = {edge:+.2f}%  {per100:+.2f} u/100 rondas  "
              f"(neto {net:+.0f} u, apostado {wagered:.0f} u, desviaciones {devs}, {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
