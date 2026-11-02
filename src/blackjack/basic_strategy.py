"""Estrategia básica para 4-8 mazos (S17 o H17, con o sin DAS).

Orden de decisión (el estándar): rendirse → dividir → doblar/pedir/plantarse.

Códigos de las tablas:
    H  = pedir           S  = plantarse
    D  = doblar, si no se puede → pedir
    Ds = doblar, si no se puede → plantarse
    Y  = dividir         N  = no dividir
    Ph = dividir sólo si se permite doblar después de dividir (DAS)
"""

from __future__ import annotations

from .cards import ACE, TEN
from .hand import Hand
from .rules import Action, Rules

# Columnas: carta visible del dealer 2, 3, 4, 5, 6, 7, 8, 9, 10, A
UPCARDS = [2, 3, 4, 5, 6, 7, 8, 9, TEN, ACE]


def _row(codes: str) -> dict[int, str]:
    parts = codes.split()
    assert len(parts) == 10
    return dict(zip(UPCARDS, parts))


# Totales duros (S17). Los totales <= 8 siempre piden y >= 17 siempre se plantan.
HARD = {
    9:  _row("H  D  D  D  D  H  H  H  H  H"),
    10: _row("D  D  D  D  D  D  D  D  H  H"),
    11: _row("D  D  D  D  D  D  D  D  D  H"),
    12: _row("H  H  S  S  S  H  H  H  H  H"),
    13: _row("S  S  S  S  S  H  H  H  H  H"),
    14: _row("S  S  S  S  S  H  H  H  H  H"),
    15: _row("S  S  S  S  S  H  H  H  H  H"),
    16: _row("S  S  S  S  S  H  H  H  H  H"),
}

# Totales suaves (S17), indexados por total: 13 = A2 ... 20 = A9.
SOFT = {
    13: _row("H  H  H  D  D  H  H  H  H  H"),
    14: _row("H  H  H  D  D  H  H  H  H  H"),
    15: _row("H  H  D  D  D  H  H  H  H  H"),
    16: _row("H  H  D  D  D  H  H  H  H  H"),
    17: _row("H  D  D  D  D  H  H  H  H  H"),
    18: _row("S  Ds Ds Ds Ds S  S  H  H  H"),
    19: _row("S  S  S  S  S  S  S  S  S  S"),
    20: _row("S  S  S  S  S  S  S  S  S  S"),
}

# Pares: ¿dividir? Indexado por la carta del par.
PAIRS = {
    ACE: _row("Y  Y  Y  Y  Y  Y  Y  Y  Y  Y"),
    TEN: _row("N  N  N  N  N  N  N  N  N  N"),
    9:   _row("Y  Y  Y  Y  Y  N  Y  Y  N  N"),
    8:   _row("Y  Y  Y  Y  Y  Y  Y  Y  Y  Y"),
    7:   _row("Y  Y  Y  Y  Y  Y  N  N  N  N"),
    6:   _row("Ph Y  Y  Y  Y  N  N  N  N  N"),
    5:   _row("N  N  N  N  N  N  N  N  N  N"),
    4:   _row("N  N  N  Ph Ph N  N  N  N  N"),
    3:   _row("Ph Ph Y  Y  Y  Y  N  N  N  N"),
    2:   _row("Ph Ph Y  Y  Y  Y  N  N  N  N"),
}

# Cambios cuando el dealer pide con 17 suave (H17).
H17_HARD = {(11, ACE): "D"}
H17_SOFT = {(18, 2): "Ds", (19, 6): "Ds"}

# Late surrender sobre totales duros (no aplica al par de 8).
SURRENDER_S17 = {(16, 9), (16, TEN), (16, ACE), (15, TEN)}
SURRENDER_H17 = SURRENDER_S17 | {(15, ACE), (17, ACE)}


def _resolve(code: str, can_double: bool) -> Action:
    if code == "H":
        return Action.HIT
    if code == "S":
        return Action.STAND
    if code == "D":
        return Action.DOUBLE if can_double else Action.HIT
    if code == "Ds":
        return Action.DOUBLE if can_double else Action.STAND
    raise ValueError(f"código desconocido: {code}")


def should_split(pair_card: int, dealer_up: int, rules: Rules) -> bool:
    code = PAIRS[pair_card][dealer_up]
    return code == "Y" or (code == "Ph" and rules.double_after_split)


def should_surrender(hand: Hand, dealer_up: int, rules: Rules) -> bool:
    h17 = rules.dealer_hits_soft_17
    if hand.is_pair and hand.cards[0] == 8:
        return h17 and dealer_up == ACE
    if hand.is_soft:
        return False
    table = SURRENDER_H17 if h17 else SURRENDER_S17
    return (hand.total, dealer_up) in table


def basic_strategy(
    hand: Hand,
    dealer_up: int,
    rules: Rules,
    can_double: bool = True,
    can_split: bool = True,
    can_surrender: bool = False,
) -> Action:
    """Acción de estrategia básica para `hand` contra la carta visible `dealer_up`."""
    if can_surrender and should_surrender(hand, dealer_up, rules):
        return Action.SURRENDER

    if can_split and hand.is_pair and should_split(hand.cards[0], dealer_up, rules):
        return Action.SPLIT

    total = hand.total
    h17 = rules.dealer_hits_soft_17

    if hand.is_soft:
        if total <= 12:   # A,A sin poder dividir
            return Action.HIT
        if total >= 21:
            return Action.STAND
        code = (h17 and H17_SOFT.get((total, dealer_up))) or SOFT[total][dealer_up]
        return _resolve(code, can_double)

    if total <= 8:
        return Action.HIT
    if total >= 17:
        return Action.STAND
    code = (h17 and H17_HARD.get((total, dealer_up))) or HARD[total][dealer_up]
    return _resolve(code, can_double)
