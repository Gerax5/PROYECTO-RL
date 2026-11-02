"""Jugador casual: estrategia básica con errores y apuestas por sesgos humanos.

Nunca mira el conteo: sus apuestas dependen sólo de sus propios resultados.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blackjack import Action, Hand, Rules

from .base_player import BasePlayer, basic_action, round_to_chip

BET_STYLES = ("hot_streak", "martingale", "flat_noise")


@dataclass(frozen=True)
class CasualConfig:
    error_rate: tuple[float, float] = (0.05, 0.20)  # cada jugador sortea la suya en este rango
    bet_style: str | None = None                    # None = sorteado entre BET_STYLES
    insurance_prob: float = 0.10                    # "asegurar", sin relación con el conteo
    max_bet_units: float = 8.0
    flat_noise_std: float = 0.25


class CasualPlayer(BasePlayer):
    player_type = "casual"

    def __init__(
        self,
        config: CasualConfig | None = None,
        rules: Rules | None = None,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(rules, rng)
        self.config = config if config is not None else CasualConfig()
        self.error_rate = float(self.rng.uniform(*self.config.error_rate))
        self.bet_style = self.config.bet_style or str(self.rng.choice(BET_STYLES))
        if self.bet_style not in BET_STYLES:
            raise ValueError(f"bet_style desconocido: {self.bet_style}")
        self._bet = 1.0

    def place_bet(self) -> float:
        if self.bet_style == "flat_noise":
            noisy = 1.0 + self.rng.normal(0.0, self.config.flat_noise_std)
            return max(1.0, round_to_chip(noisy))
        return self._bet

    def on_round_end(self, net: float) -> None:
        if net == 0:
            return
        if self.bet_style == "hot_streak":
            self._bet = self._bet + 1.0 if net > 0 else 1.0
        elif self.bet_style == "martingale":
            self._bet = self._bet * 2.0 if net < 0 else 1.0
        self._bet = min(self._bet, self.config.max_bet_units)

    def decide(self, hand: Hand, dealer_up: int, allowed: frozenset[Action]) -> Action:
        best = basic_action(hand, dealer_up, self.rules, allowed)
        return self._maybe_mistake(hand, best, allowed, self.error_rate)

    def insure(self, hand: Hand, dealer_up: int) -> bool:
        return bool(self.rng.random() < self.config.insurance_prob)
