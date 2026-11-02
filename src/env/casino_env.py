"""Ambiente Gymnasium del supervisor: un `step` es una ronda de la mesa.

Flujo de un paso: se aplican expulsiones y vigilancias, se juega la ronda con quienes
siguen sentados, se calcula la recompensa y el extractor construye la observación.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import gymnasium as gym
import numpy as np

from .events import SeatEvent
from .table import Table, TableConfig

ACTION_NONE, ACTION_WATCH, ACTION_EXPEL = 0, 1, 2
COUNTER_TYPES = ("counter", "stealth")

# Ventaja aproximada de un jugador con Hi-Lo: -0.5% sin conteo, +0.5% por cada TC.
BASE_EDGE = -0.005
EDGE_PER_TC = 0.005


def expected_counter_gain(bet: float, tc: float) -> float:
    return bet * (BASE_EDGE + EDGE_PER_TC * tc)


@dataclass(frozen=True)
class RewardConfig:
    """Pesos de la sección 6.4; los define la persona B."""

    expel_counter: float = 10.0
    expel_stealth: float = 15.0
    expel_casual: float = -25.0
    counter_edge_per_round: float = -1.0
    watch_cost: float = -0.05
    early_detection_k: float = 20.0     # el bono es k / rondas_vistas, sólo si acierta


class FeatureExtractor(Protocol):
    """Lo que el ambiente necesita del extractor de la persona B."""

    n_seat_features: int
    n_global_features: int

    def reset(self) -> None: ...

    def transform(self, events: list[SeatEvent]) -> dict[str, np.ndarray]: ...


class MinimalFeatureExtractor:
    """Provisional hasta tener `src/env/features.py`; no incluye las señales de TC."""

    n_seat_features = 8
    n_global_features = 3

    def __init__(self, seats: int = 5):
        self.seats = seats
        self.reset()

    def reset(self) -> None:
        self._history: list[list[SeatEvent]] = [[] for _ in range(self.seats)]
        self._tc = self._penetration = 0.0
        self._occupied = [False] * self.seats

    def transform(self, events: list[SeatEvent]) -> dict[str, np.ndarray]:
        for e in events:
            if e.event == "sit":
                self._occupied[e.seat] = True
                self._history[e.seat] = []
            elif e.event in ("leave", "expelled"):
                self._occupied[e.seat] = False
                self._history[e.seat] = []
            elif e.event == "play":
                self._history[e.seat].append(e)
                self._tc = e.true_count
                self._penetration = e.shoe_penetration

        seats = np.zeros((self.seats, self.n_seat_features), dtype=np.float32)
        for seat, history in enumerate(self._history):
            seats[seat, 0] = self._occupied[seat]
            played = [e for e in history if e.bet > 0]
            seats[seat, 1] = len(history)
            if played:
                bets = [e.bet for e in played]
                seats[seat, 2] = bets[-1]
                seats[seat, 3] = np.mean(bets)
                seats[seat, 4] = max(bets) / min(bets)
                seats[seat, 5] = np.mean([e.deviated_from_basic for e in played])
                seats[seat, 6] = np.mean([e.matched_index_play for e in played])
                seats[seat, 7] = sum(e.result_units for e in played)

        global_features = np.array(
            [self._tc, self._penetration, sum(self._occupied)], dtype=np.float32
        )
        return {"seats": seats, "global": global_features}


class CasinoTableEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        table_config: TableConfig | None = None,
        reward_config: RewardConfig | None = None,
        rounds_per_episode: int = 600,
        extractor: FeatureExtractor | None = None,
    ):
        self.table_config = table_config if table_config is not None else TableConfig()
        self.reward_config = reward_config if reward_config is not None else RewardConfig()
        self.rounds_per_episode = rounds_per_episode
        n_seats = self.table_config.seats
        self.extractor = extractor if extractor is not None else MinimalFeatureExtractor(n_seats)
        self.action_space = gym.spaces.MultiDiscrete([3] * n_seats)
        self.observation_space = gym.spaces.Dict(
            {
                "seats": gym.spaces.Box(
                    -np.inf, np.inf, (n_seats, self.extractor.n_seat_features), np.float32
                ),
                "global": gym.spaces.Box(
                    -np.inf, np.inf, (self.extractor.n_global_features,), np.float32
                ),
            }
        )
        self.table: Table | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.table = Table(self.table_config, self.np_random)
        self.extractor.reset()
        events = self.table.fill_initial_seats()
        return self.extractor.transform(events), {"events": events}

    def _expel_reward(self, true_type: str, rounds_seen: int) -> tuple[float, float]:
        """(recompensa por la expulsión, bono de detección temprana)."""
        cfg = self.reward_config
        if true_type == "casual":
            return cfg.expel_casual, 0.0
        base = cfg.expel_counter if true_type == "counter" else cfg.expel_stealth
        return base, cfg.early_detection_k / max(rounds_seen, 1)

    def step(self, action):
        if self.table is None:
            raise RuntimeError("hay que llamar a reset() antes de step()")
        cfg = self.reward_config
        expel = detection = watch = 0.0
        expulsions, watched = [], []

        for seat, a in enumerate(np.asarray(action)):
            if a == ACTION_EXPEL:
                sp = self.table.expel(seat)
                if sp is not None:
                    true_type = sp.player.player_type
                    base, bonus = self._expel_reward(true_type, sp.rounds_seen)
                    expel += base
                    detection += bonus
                    expulsions.append(
                        {"seat": seat, "true_type": true_type, "rounds_seen": sp.rounds_seen}
                    )
            elif a == ACTION_WATCH and self.table.seats[seat] is not None:
                watch += cfg.watch_cost
                watched.append(seat)

        events = self.table.play_round()

        counter_gain = sum(
            expected_counter_gain(e.bet, e.true_count)
            for e in events
            if e.event == "play" and e.true_type in COUNTER_TYPES
        )
        edge = cfg.counter_edge_per_round * counter_gain
        reward = expel + detection + watch + edge

        info = {
            "events": events,
            "expulsions": expulsions,
            "watched": watched,
            "house_net": -sum(e.result_units for e in events if e.event == "play"),
            "reward_terms": {"expel": expel, "detection": detection, "watch": watch, "edge": edge},
        }
        truncated = self.table.round_id >= self.rounds_per_episode
        return self.extractor.transform(events), float(reward), False, truncated, info
