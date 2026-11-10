"""Explicabilidad (XAI) del supervisor: por qué vigila o expulsa a cada jugador.

Un explicador tiene `explain(obs) -> list[dict | None]` (uno por silla, None si está
vacía) y se llama con la misma observación que recibe la política, *antes* de `act`.
Cada dict trae al menos:
    score   probabilidad que resume la decisión (P(expulsar) o P(contador))
    base    valor de referencia del que parte la explicación
    attr    {feature de la silla: aporte}; positivo = empuja a expulsar
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from env.features import SEAT_FEATURES


def make_explainer(policy, **kw):
    """El explicador adecuado para `policy`, o None si no hay uno (heurística, NeverExpel)."""
    # Por atributos, para no importar sklearn ni torch si la política no los usa.
    model = getattr(policy, "model", None)
    if hasattr(model, "predict_proba"):           # ClassifierSupervisor
        from .tabular import ClassifierExplainer

        return ClassifierExplainer(policy, **kw)
    if hasattr(model, "policy"):                  # SB3Supervisor
        from .policy import PolicyExplainer

        return PolicyExplainer(policy, **kw)
    return None


def top_features(attr: dict[str, float], k: int = 3) -> list[tuple[str, float]]:
    return sorted(attr.items(), key=lambda kv: -abs(kv[1]))[:k]


class AttributionStats:
    """Acumula atribuciones por grupo (p. ej. "expulsión correcta", "falso positivo")."""

    def __init__(self):
        self.sum: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(SEAT_FEATURES)))
        self.abs: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(SEAT_FEATURES)))
        self.n: dict[str, int] = defaultdict(int)

    def add(self, group: str, attr: dict[str, float]) -> None:
        v = np.array([attr[name] for name in SEAT_FEATURES])
        self.sum[group] += v
        self.abs[group] += np.abs(v)
        self.n[group] += 1

    def table(self, group: str) -> list[dict]:
        n = max(self.n[group], 1)
        rows = [{"feature": name, "mean": float(self.sum[group][i] / n), "mean_abs": float(self.abs[group][i] / n)}
                for i, name in enumerate(SEAT_FEATURES)]
        return sorted(rows, key=lambda r: -r["mean_abs"])

    def groups(self) -> list[str]:
        return [g for g in self.n if self.n[g] > 0]


__all__ = ["AttributionStats", "make_explainer", "top_features"]
