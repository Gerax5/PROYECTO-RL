"""Explicaciones del baseline supervisado (`ClassifierSupervisor`).

- Local: oclusión marginal. Para cada feature j,
      atribución_j = P(x) − E_b[ P(x con x_j := b_j) ]
  con b tomado de una muestra de referencia (jugadores reales del simulador). Es la
  versión "una feature a la vez" de SHAP intervencional: exacta para modelos aditivos
  (la logística en escala de log-odds) y una aproximación para el gradient boosting.
- Global: importancia por permutación (caída de AUC al barajar cada feature).
"""

from __future__ import annotations

import numpy as np

from env.features import F, SEAT_FEATURES


def reference_sample(n: int = 64, seeds: tuple[int, ...] = (90_001, 90_002, 90_003), rounds: int = 400,
                     table_config=None) -> np.ndarray:
    """Filas de features de jugadores de todos los tipos, para usar como referencia."""
    from supervisor.classifier import collect_samples

    x, _ = collect_samples(list(seeds), table_config, rounds)
    rng = np.random.default_rng(0)
    return x[rng.choice(len(x), size=min(n, len(x)), replace=False)]


def occlusion_attributions(predict, x: np.ndarray, background: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`predict(X) -> P(contador)`. Devuelve (P(x) de cada fila, atribuciones (filas, features))."""
    x = np.atleast_2d(x)
    n, d = x.shape
    b = len(background)
    p = predict(x)
    attr = np.empty((n, d))
    for j in range(d):
        z = np.repeat(x, b, axis=0)
        z[:, j] = np.tile(background[:, j], n)
        attr[:, j] = p - predict(z).reshape(n, b).mean(1)
    return p, attr


class ClassifierExplainer:
    method = "Oclusión marginal sobre P(contador)"
    score_label = "P(contador)"

    def __init__(self, supervisor, background: np.ndarray | None = None):
        self.supervisor = supervisor
        self.background = background if background is not None else reference_sample()
        self._predict = lambda z: supervisor.model.predict_proba(z)[:, 1]
        # P media de la referencia: el punto de partida de la explicación
        self.base = float(self._predict(self.background).mean())

    def explain(self, obs: dict[str, np.ndarray]) -> list[dict | None]:
        seats = np.asarray(obs["seats"])
        occupied = seats[:, F["occupied"]] > 0
        result: list[dict | None] = [None] * len(seats)
        if not occupied.any():
            return result
        rows = np.flatnonzero(occupied)
        p, attr = occlusion_attributions(self._predict, seats[rows], self.background)
        for k, s in enumerate(rows):
            result[s] = {
                "score": round(float(p[k]), 4),
                "base": round(self.base, 4),
                "attr": {name: round(float(attr[k, i]), 4) for i, name in enumerate(SEAT_FEATURES)},
            }
        return result


def permutation_importance_table(model, x: np.ndarray, y: np.ndarray, repeats: int = 10,
                                 seed: int = 0) -> list[dict]:
    """Importancia global (caída de AUC), ordenada de mayor a menor."""
    from sklearn.inspection import permutation_importance

    r = permutation_importance(model, x, y, scoring="roc_auc", n_repeats=repeats, random_state=seed)
    rows = [{"feature": name, "auc_drop": float(r.importances_mean[i]), "std": float(r.importances_std[i])}
            for i, name in enumerate(SEAT_FEATURES)]
    return sorted(rows, key=lambda row: -row["auc_drop"])
