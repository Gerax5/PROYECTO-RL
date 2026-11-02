"""Illustrious 18: las 18 desviaciones de la estrategia básica que más valor
aportan a un contador Hi-Lo (Don Schlesinger, *Blackjack Attack*).

Cada jugada indica un umbral de True Count (TC): con `TC >= threshold` se
hace `at_or_above`; por debajo, `below`. Para las jugadas con índice negativo
(p.ej. 13 vs 2), la estrategia básica coincide con `at_or_above` y la
desviación ocurre cuando el conteo es muy negativo.
"""

from __future__ import annotations

from dataclasses import dataclass

from .basic_strategy import basic_strategy
from .cards import ACE, TEN
from .hand import Hand
from .rules import Action, Rules

INSURANCE_INDEX = 3.0


@dataclass(frozen=True)
class IndexPlay:
    name: str
    kind: str          # "hard" o "pair" (par de dieces)
    total: int
    dealer_up: int
    threshold: float
    at_or_above: Action
    below: Action


ILLUSTRIOUS_18: list[IndexPlay] = [
    # 1. Seguro con TC >= +3 (se maneja aparte en `take_insurance`).
    IndexPlay("16 vs 10",  "hard", 16, TEN, 0,  Action.STAND,  Action.HIT),
    IndexPlay("15 vs 10",  "hard", 15, TEN, 4,  Action.STAND,  Action.HIT),
    IndexPlay("TT vs 5",   "pair", 20, 5,   5,  Action.SPLIT,  Action.STAND),
    IndexPlay("TT vs 6",   "pair", 20, 6,   4,  Action.SPLIT,  Action.STAND),
    IndexPlay("10 vs 10",  "hard", 10, TEN, 4,  Action.DOUBLE, Action.HIT),
    IndexPlay("12 vs 3",   "hard", 12, 3,   2,  Action.STAND,  Action.HIT),
    IndexPlay("12 vs 2",   "hard", 12, 2,   3,  Action.STAND,  Action.HIT),
    IndexPlay("11 vs A",   "hard", 11, ACE, 1,  Action.DOUBLE, Action.HIT),
    IndexPlay("9 vs 2",    "hard", 9,  2,   1,  Action.DOUBLE, Action.HIT),
    IndexPlay("10 vs A",   "hard", 10, ACE, 4,  Action.DOUBLE, Action.HIT),
    IndexPlay("9 vs 7",    "hard", 9,  7,   3,  Action.DOUBLE, Action.HIT),
    IndexPlay("16 vs 9",   "hard", 16, 9,   5,  Action.STAND,  Action.HIT),
    IndexPlay("13 vs 2",   "hard", 13, 2,   -1, Action.STAND,  Action.HIT),
    IndexPlay("12 vs 4",   "hard", 12, 4,   0,  Action.STAND,  Action.HIT),
    IndexPlay("12 vs 5",   "hard", 12, 5,   -2, Action.STAND,  Action.HIT),
    IndexPlay("12 vs 6",   "hard", 12, 6,   -1, Action.STAND,  Action.HIT),
    IndexPlay("13 vs 3",   "hard", 13, 3,   -2, Action.STAND,  Action.HIT),
]


def take_insurance(tc: float) -> bool:
    return tc >= INSURANCE_INDEX


def find_index_play(
    hand: Hand,
    dealer_up: int,
    basic: Action,
    can_double: bool,
    can_split: bool,
) -> IndexPlay | None:
    """Jugada del I18 que aplica a esta situación, o None."""
    if basic == Action.SURRENDER:
        return None
    if hand.is_pair and hand.cards[0] == TEN:
        if not can_split:
            return None
        kind = "pair"
    elif hand.is_soft or basic == Action.SPLIT:
        return None
    else:
        kind = "hard"

    for play in ILLUSTRIOUS_18:
        if play.kind == kind and play.total == hand.total and play.dealer_up == dealer_up:
            if play.at_or_above == Action.DOUBLE and not can_double:
                return None
            return play
    return None


def counter_strategy(
    hand: Hand,
    dealer_up: int,
    tc: float,
    rules: Rules,
    can_double: bool = True,
    can_split: bool = True,
    can_surrender: bool = False,
) -> tuple[Action, IndexPlay | None]:
    """Estrategia básica + Illustrious 18.

    Devuelve la acción y la jugada índice que se consultó (None si ninguna aplica).
    La acción es una desviación real si difiere de `basic_strategy(...)`.
    """
    basic = basic_strategy(hand, dealer_up, rules, can_double, can_split, can_surrender)
    play = find_index_play(hand, dealer_up, basic, can_double, can_split)
    if play is None:
        return basic, None
    action = play.at_or_above if tc >= play.threshold else play.below
    return action, play
