"""Baseline supervisado (sección 7.1): clasificador sobre las features + umbral.

Se entrena con las etiquetas reales del simulador, algo que el supervisor RL no tiene;
por eso es una cota exigente: si el RL no lo supera, la recompensa o la red fallan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from env import CasinoTableEnv, SeatEvent, TableConfig
from env.casino_env import ACTION_EXPEL, ACTION_NONE
from env.features import F

COUNTER_TYPES = ("counter", "stealth")
KINDS = ("logistic", "gboost")


def collect_samples(
    seeds: list[int],
    table_config: TableConfig | None = None,
    rounds: int = 600,
    min_rounds: int = 10,
    every: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Features por silla con etiqueta 1 = contador (abierto o disimulado).

    Sin expulsiones, para que cada jugador complete su sesión. Se toma una muestra cada
    `every` rondas porque rondas seguidas de la misma silla casi no cambian.
    """
    env = CasinoTableEnv(table_config, rounds_per_episode=rounds)
    n_seats = env.table_config.seats
    xs, ys = [], []
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        types: list[str | None] = [None] * n_seats
        done, step = False, 0
        while True:
            _track_types(types, info["events"])
            if step % every == 0:
                for seat, kind in enumerate(types):
                    row = obs["seats"][seat]
                    if kind is not None and row[F["rounds_seen"]] >= min_rounds:
                        xs.append(row.copy())
                        ys.append(kind in COUNTER_TYPES)
            if done:
                break
            obs, _, _, done, info = env.step(np.full(n_seats, ACTION_NONE))
            step += 1
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=int)


def _track_types(types: list[str | None], events: list[SeatEvent]) -> None:
    for e in events:
        if e.event == "sit":
            types[e.seat] = e.true_type
        elif e.event in ("leave", "expelled"):
            types[e.seat] = None


def fit_classifier(x: np.ndarray, y: np.ndarray, kind: str = "gboost", seed: int = 0) -> Pipeline:
    if kind == "logistic":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    elif kind == "gboost":
        model = make_pipeline(HistGradientBoostingClassifier(random_state=seed, class_weight="balanced"))
    else:
        raise ValueError(f"tipo de clasificador desconocido: {kind}")
    return model.fit(x, y)


@dataclass
class ClassifierSupervisor:
    """Expulsa si P(contador) > `threshold` tras `min_rounds` rondas."""

    model: Pipeline
    threshold: float = 0.9
    min_rounds: int = 30
    name: str = field(default="clasificador")

    @classmethod
    def load(cls, path: str | Path, **kw) -> "ClassifierSupervisor":
        return cls(joblib.load(path), name=Path(path).stem, **kw)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def reset(self) -> None:
        pass

    def act(self, obs: dict[str, np.ndarray], events: list[SeatEvent] | None = None) -> np.ndarray:
        seats = obs["seats"]
        action = np.full(len(seats), ACTION_NONE)
        ready = (seats[:, F["occupied"]] > 0) & (seats[:, F["rounds_seen"]] > self.min_rounds)
        if ready.any():
            proba = self.model.predict_proba(seats[ready])[:, 1]
            action[np.flatnonzero(ready)[proba > self.threshold]] = ACTION_EXPEL
        return action
