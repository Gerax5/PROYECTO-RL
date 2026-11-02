"""Interfaz común de los jugadores y utilidades de decisión.

Los jugadores son políticas fijas: la mesa les pide una apuesta y sus decisiones,
y les avisa de cada carta pública, de cada barajado y del resultado de la ronda.
Las apuestas se expresan en unidades (1 unidad = apuesta mínima de la mesa).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from blackjack import Action, Hand, Rules, basic_strategy


def round_to_chip(units: float) -> float:
    """Redondea a media unidad: un jugador real no apuesta 2.37 unidades."""
    return round(units * 2) / 2


def basic_action(hand: Hand, dealer_up: int, rules: Rules, allowed: frozenset[Action]) -> Action:
    return basic_strategy(
        hand,
        dealer_up,
        rules,
        can_double=Action.DOUBLE in allowed,
        can_split=Action.SPLIT in allowed,
        can_surrender=Action.SURRENDER in allowed,
    )


def plausible_mistakes(hand: Hand, best: Action, allowed: frozenset[Action]) -> list[Action]:
    """Errores que una persona cometería de verdad.

    Se descartan los absurdos (pedir con 20, plantarse con 8 duro, doblar 15 duro),
    que además servirían al supervisor como una señal artificial de "es casual".
    La lista va ordenada porque iterar un frozenset depende de PYTHONHASHSEED y
    rompería la reproducibilidad con semilla.
    """
    options = set(allowed) - {best, Action.SURRENDER}
    if hand.total >= 19:
        options -= {Action.HIT, Action.DOUBLE, Action.SPLIT}
    if not hand.is_soft:
        if hand.total <= 11:
            options.discard(Action.STAND)
        else:
            options.discard(Action.DOUBLE)
    return sorted(options, key=lambda a: a.value)


class BasePlayer(ABC):
    player_type: ClassVar[str]

    def __init__(self, rules: Rules | None = None, rng: np.random.Generator | None = None):
        self.rules = rules if rules is not None else Rules()
        self.rng = rng if rng is not None else np.random.default_rng()

    @abstractmethod
    def place_bet(self) -> float:
        """Apuesta de la siguiente ronda en unidades; 0 = no juega esta ronda."""

    @abstractmethod
    def decide(self, hand: Hand, dealer_up: int, allowed: frozenset[Action]) -> Action:
        """Acción para `hand`; debe pertenecer a `allowed`."""

    def insure(self, hand: Hand, dealer_up: int) -> bool:
        return False

    def on_shuffle(self, decks_remaining: float) -> None:
        """El zapato se acaba de barajar."""

    def observe_card(self, card: int, decks_remaining: float) -> None:
        """Una carta se acaba de volver pública."""

    def on_round_end(self, net: float) -> None:
        """Ganancia neta de la ronda; sólo se llama si el jugador apostó."""

    def _maybe_mistake(self, hand: Hand, best: Action, allowed: frozenset[Action], prob: float) -> Action:
        if prob > 0 and self.rng.random() < prob:
            options = plausible_mistakes(hand, best, allowed)
            if options:
                return options[self.rng.integers(len(options))]
        return best
