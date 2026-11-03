"""Extractor de features por silla (sección 6.2): lo que ve el supervisor.

Sólo usa información pública de los eventos; nunca lee `true_type`. Las señales de
apuesta y juego se calculan sobre una ventana de las últimas `window` rondas jugadas;
`rounds_seen`, `entry_tc` y `net_win_units` cubren toda la estancia del jugador.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .events import SeatEvent

SEAT_FEATURES = (
    "occupied",                   # silla ocupada (0/1)
    "rounds_seen",                # rondas desde que se sentó
    "bet_norm",                   # apuesta actual / apuesta mínima de la ventana
    "corr_bet_tc",                # Pearson apuesta ~ TC  ← señal clave
    "rank_corr_bet_tc",           # Spearman apuesta ~ TC (robusta al ruido de apuesta)
    "bet_tc_slope",               # pendiente de regresión apuesta ~ TC
    "bet_spread",                 # max / min de la apuesta
    "corr_bet_prev_result",       # Pearson apuesta ~ resultado de la ronda anterior
    "index_play_rate",            # desviaciones que coinciden con un índice / desviaciones
    "basic_strategy_error_rate",  # desviaciones que no son índice / rondas jugadas
    "insurance_rate",             # seguros tomados / rondas jugadas
    "insurance_tc_mean",          # TC medio cuando toma seguro (0 si nunca)
    "sit_out_rate",               # rondas sin apostar / rondas sentado (wonging)
    "entry_tc",                   # TC cuando se sentó
    "net_win_units",              # ganancia acumulada en la estancia
)
GLOBAL_FEATURES = ("true_count", "shoe_penetration", "n_players", "n_expelled")

F = {name: i for i, name in enumerate(SEAT_FEATURES)}
G = {name: i for i, name in enumerate(GLOBAL_FEATURES)}

# Mínimo de rondas con apuesta para calcular correlaciones y pendientes.
MIN_SAMPLES = 5


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < MIN_SAMPLES:
        return 0.0
    sx, sy = x.std(), y.std()
    if sx < 1e-9 or sy < 1e-9:
        return 0.0
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))


def _ranks(x: np.ndarray) -> np.ndarray:
    """Rangos con empates promediados (como `scipy.stats.rankdata`)."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    return sums[inverse] / counts[inverse]


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < MIN_SAMPLES or x.var() < 1e-9:
        return 0.0
    return float(np.mean((x - x.mean()) * (y - y.mean())) / x.var())


@dataclass
class _SeatHistory:
    entry_tc: float
    rounds_seen: int = 0
    net_win: float = 0.0
    window: deque[SeatEvent] = field(default_factory=deque)


class SeatFeatureExtractor:
    """Implementa el protocolo `FeatureExtractor` de `casino_env`."""

    n_seat_features = len(SEAT_FEATURES)
    n_global_features = len(GLOBAL_FEATURES)

    def __init__(self, seats: int = 5, window: int = 50):
        self.seats = seats
        self.window = window
        self.reset()

    def reset(self) -> None:
        self._history: list[_SeatHistory | None] = [None] * self.seats
        self._tc = self._penetration = 0.0
        self._n_expelled = 0

    def _update(self, events: list[SeatEvent]) -> None:
        for e in events:
            if e.event == "sit":
                self._history[e.seat] = _SeatHistory(
                    entry_tc=e.true_count, window=deque(maxlen=self.window)
                )
            elif e.event in ("leave", "expelled"):
                self._history[e.seat] = None
                self._n_expelled += e.event == "expelled"
            elif e.event == "play":
                h = self._history[e.seat]
                h.rounds_seen += 1
                h.net_win += e.result_units
                h.window.append(e)
                self._tc = e.true_count
                self._penetration = e.shoe_penetration

    @staticmethod
    def _seat_vector(h: _SeatHistory) -> np.ndarray:
        v = np.zeros(len(SEAT_FEATURES), dtype=np.float32)
        v[F["occupied"]] = 1.0
        v[F["rounds_seen"]] = h.rounds_seen
        v[F["entry_tc"]] = h.entry_tc
        v[F["net_win_units"]] = h.net_win
        if not h.window:
            return v

        v[F["sit_out_rate"]] = np.mean([e.bet == 0 for e in h.window])
        played = [e for e in h.window if e.bet > 0]
        if not played:
            return v

        bets = np.array([e.bet for e in played])
        tcs = np.array([e.true_count for e in played])
        v[F["bet_norm"]] = bets[-1] / bets.min()
        v[F["bet_spread"]] = bets.max() / bets.min()
        v[F["corr_bet_tc"]] = _pearson(bets, tcs)
        v[F["rank_corr_bet_tc"]] = _pearson(_ranks(bets), _ranks(tcs))
        v[F["bet_tc_slope"]] = _slope(tcs, bets)
        results = np.array([e.result_units for e in played])
        v[F["corr_bet_prev_result"]] = _pearson(bets[1:], results[:-1])

        deviated = np.array([e.deviated_from_basic for e in played])
        matched = np.array([e.matched_index_play for e in played])
        if deviated.any():
            v[F["index_play_rate"]] = matched[deviated].mean()
        v[F["basic_strategy_error_rate"]] = np.mean(deviated & ~matched)

        insured = [e.true_count for e in played if "insurance" in e.actions]
        v[F["insurance_rate"]] = len(insured) / len(played)
        if insured:
            v[F["insurance_tc_mean"]] = np.mean(insured)
        return v

    def transform(self, events: list[SeatEvent]) -> dict[str, np.ndarray]:
        self._update(events)
        seats = np.zeros((self.seats, self.n_seat_features), dtype=np.float32)
        for seat, h in enumerate(self._history):
            if h is not None:
                seats[seat] = self._seat_vector(h)
        n_players = sum(h is not None for h in self._history)
        global_features = np.array(
            [self._tc, self._penetration, n_players, self._n_expelled], dtype=np.float32
        )
        return {"seats": seats, "global": global_features}
