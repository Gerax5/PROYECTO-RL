"""Contador disimulado: cuenta igual que el abierto, pero se camufla.

`StealthConfig` guarda los parámetros de camuflaje por separado para que el
adversario evolutivo pueda optimizarlos; `StealthConfig.from_disguise` los
interpola con un solo escalar para el curriculum.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from blackjack import Action, Hand, Rules, counter_strategy, take_insurance

from .base_player import basic_action, round_to_chip
from .counter import CounterPlayer, ramp_bet

CHEAP_ERROR_TC_RANGE = 2.0
COVER_TC_RANGE = (-1.0, 2.0)    # [min, max) del TC en que se permiten apuestas de cobertura


@dataclass(frozen=True)
class StealthConfig:
    bet_spread_max: float = 4.0
    bet_noise_std: float = 0.3
    cover_bet_prob: float = 0.25
    index_play_prob: float = 0.5
    deliberate_error_prob: float = 0.03
    delay_bet_ramp_rounds: int = 1

    @classmethod
    def from_disguise(cls, level: float) -> "StealthConfig":
        """0 = idéntico al contador abierto; 1 = los valores por defecto de esta clase."""
        if not 0.0 <= level <= 1.0:
            raise ValueError("disguise_level debe estar en [0, 1]")
        full = cls()
        return cls(
            bet_spread_max=12.0 + level * (full.bet_spread_max - 12.0),
            bet_noise_std=level * full.bet_noise_std,
            cover_bet_prob=level * full.cover_bet_prob,
            index_play_prob=1.0 + level * (full.index_play_prob - 1.0),
            deliberate_error_prob=level * full.deliberate_error_prob,
            delay_bet_ramp_rounds=round(level * full.delay_bet_ramp_rounds),
        )


class StealthCounterPlayer(CounterPlayer):
    player_type = "stealth"

    def __init__(
        self,
        stealth: StealthConfig | None = None,
        rules: Rules | None = None,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(rules=rules, rng=rng)
        self.stealth = stealth if stealth is not None else StealthConfig()
        self._tc_history: deque[float] = deque(maxlen=self.stealth.delay_bet_ramp_rounds + 1)
        self._last_bet = 1.0
        self._won_last = False

    def _clip_bet(self, units: float) -> float:
        return float(np.clip(round_to_chip(units), 1.0, self.stealth.bet_spread_max))

    def place_bet(self) -> float:
        cfg = self.stealth
        self._tc_history.append(self.tc)

        if (
            COVER_TC_RANGE[0] <= self.tc < COVER_TC_RANGE[1]
            and cfg.cover_bet_prob > 0
            and self.rng.random() < cfg.cover_bet_prob
        ):
            # Apuesta de cobertura: "racha caliente", igual que un casual. Sólo con TC
            # neutro, donde la casa apenas tiene ventaja: con TC alto le quitaba las
            # apuestas grandes y con TC negativo pagaba la ventaja de la casa.
            units = self._last_bet * 2.0 if self._won_last else 1.0
        else:
            # Con historial lleno, _tc_history[0] es el TC de hace `delay` rondas.
            units = ramp_bet(self._tc_history[0])
            if cfg.bet_noise_std > 0:
                units *= 1.0 + self.rng.normal(0.0, cfg.bet_noise_std)

        self._last_bet = self._clip_bet(units)
        return self._last_bet

    def on_round_end(self, net: float) -> None:
        self._won_last = net > 0

    def decide(self, hand: Hand, dealer_up: int, allowed: frozenset[Action]) -> Action:
        action, play = counter_strategy(
            hand,
            dealer_up,
            self.tc,
            self.rules,
            can_double=Action.DOUBLE in allowed,
            can_split=Action.SPLIT in allowed,
            can_surrender=Action.SURRENDER in allowed,
        )
        basic = basic_action(hand, dealer_up, self.rules, allowed)
        if action != basic and self.rng.random() >= self.stealth.index_play_prob:
            action = basic
        # Un error deliberado sólo se comete cerca del índice, donde casi no cuesta:
        # errores al azar le quitaban casi toda la ganancia al contador.
        if (
            play is not None
            and play.kind == "hard"
            and abs(self.tc - play.threshold) <= CHEAP_ERROR_TC_RANGE
            and self.rng.random() < self.stealth.deliberate_error_prob
        ):
            return play.below if action == play.at_or_above else play.at_or_above
        return action

    def insure(self, hand: Hand, dealer_up: int) -> bool:
        return take_insurance(self.tc) and bool(self.rng.random() < self.stealth.index_play_prob)
