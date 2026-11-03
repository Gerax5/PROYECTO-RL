"""Baselines sin aprendizaje (sección 7.1), para saber si el RL aporta algo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from env.casino_env import ACTION_EXPEL, ACTION_NONE
from env.features import F


class NeverExpel:
    """Cota inferior: lo que pierde la casa sin supervisor."""

    def reset(self) -> None:
        pass

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        return np.full(len(obs["seats"]), ACTION_NONE)


@dataclass
class HeuristicSupervisor:
    """Expulsa si la correlación apuesta–TC supera `threshold` tras `min_rounds` rondas."""

    threshold: float = 0.6
    min_rounds: int = 30
    feature: str = "corr_bet_tc"

    def reset(self) -> None:
        pass

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        seats = obs["seats"]
        expel = (
            (seats[:, F["occupied"]] > 0)
            & (seats[:, F["rounds_seen"]] > self.min_rounds)
            & (seats[:, F[self.feature]] > self.threshold)
        )
        return np.where(expel, ACTION_EXPEL, ACTION_NONE)
