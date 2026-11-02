"""Contador abierto: Hi-Lo, rampa 1-2-4-8-12 e Illustrious 18."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from blackjack import Action, Hand, Rules, counter_strategy, hi_lo_value, take_insurance, true_count

from .base_player import BasePlayer


def ramp_bet(tc: float) -> float:
    """1 unidad con TC <= 1, 2 con TC 2, 4 con TC 3, 8 con TC 4, 12 con TC >= 5."""
    level = math.floor(tc)
    if level >= 5:
        return 12.0
    return {2: 2.0, 3: 4.0, 4: 8.0}.get(level, 1.0)


@dataclass(frozen=True)
class CounterConfig:
    wonging: bool = False       # si es True, no apuesta mientras TC < wong_out_tc
    wong_out_tc: float = -1.0


class CounterPlayer(BasePlayer):
    player_type = "counter"

    def __init__(
        self,
        config: CounterConfig | None = None,
        rules: Rules | None = None,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(rules, rng)
        self.config = config if config is not None else CounterConfig()
        self.running_count = 0
        self.decks_remaining = 6.0

    @property
    def tc(self) -> float:
        return true_count(self.running_count, self.decks_remaining)

    def on_shuffle(self, decks_remaining: float) -> None:
        self.running_count = 0
        self.decks_remaining = decks_remaining

    def observe_card(self, card: int, decks_remaining: float) -> None:
        self.running_count += hi_lo_value(card)
        self.decks_remaining = decks_remaining

    def place_bet(self) -> float:
        if self.config.wonging and self.tc < self.config.wong_out_tc:
            return 0.0
        return ramp_bet(self.tc)

    def decide(self, hand: Hand, dealer_up: int, allowed: frozenset[Action]) -> Action:
        action, _ = counter_strategy(
            hand,
            dealer_up,
            self.tc,
            self.rules,
            can_double=Action.DOUBLE in allowed,
            can_split=Action.SPLIT in allowed,
            can_surrender=Action.SURRENDER in allowed,
        )
        return action

    def insure(self, hand: Hand, dealer_up: int) -> bool:
        return take_insurance(self.tc)
