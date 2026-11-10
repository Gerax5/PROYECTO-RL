"""Gradientes integrados sobre la política entrenada (PPO o RecurrentPPO).

Para cada silla se explica el log-odds de expulsar, log P(expulsar) − log P(no expulsar)
(no se satura cuando P ≈ 1, a diferencia de log P): cuánto empujó cada feature
hacia la expulsión (positivo) o en contra (negativo), comparando con una observación
de referencia. La referencia es la observación normalizada nula, es decir, la mesa
promedio que vio `VecNormalize` durante el entrenamiento.

Con RecurrentPPO la memoria de la LSTM se deja fija en el estado con el que el
supervisor tomó la decisión: se atribuye sólo la observación actual. Lo que la
memoria aporta queda en `base` (el valor del objetivo con la observación de referencia).

Completitud: sum(atribuciones) ≈ objetivo(x) − objetivo(referencia).
"""

from __future__ import annotations

import numpy as np
import torch

from env.casino_env import ACTION_EXPEL
from env.features import F, GLOBAL_FEATURES, SEAT_FEATURES


class PolicyExplainer:
    """Explica las decisiones de un `SB3Supervisor`. Llamar a `explain(obs)` justo
    antes de `supervisor.act(obs)`, para usar el mismo estado de la LSTM."""

    method = "Gradientes integrados sobre log-odds de expulsar"
    score_label = "P(expulsar)"

    def __init__(self, supervisor, steps: int = 32, target_action: int = ACTION_EXPEL):
        self.supervisor = supervisor
        self.policy = supervisor.model.policy
        self.recurrent = hasattr(self.policy, "lstm_actor")
        self.steps = steps
        self.target_action = target_action

    # ------------------------------------------------------------ red
    def _log_probs(self, seats: torch.Tensor, glob: torch.Tensor, n: int) -> torch.Tensor:
        """(n, sillas, acciones) con log-probabilidades, para un lote de n observaciones."""
        obs = {"seats": seats, "global": glob}
        if self.recurrent:
            h, c = self._lstm_state()
            states = (h.repeat(1, n, 1), c.repeat(1, n, 1))
            starts = torch.full((n,), float(self.supervisor._episode_start[0]))
            dist, _ = self.policy.get_distribution(obs, states, starts)
        else:
            dist = self.policy.get_distribution(obs)
        # MultiCategorical: una Categorical por silla; .logits ya está normalizado (log-softmax)
        return torch.stack([d.logits for d in dist.distribution], dim=1)

    def _odds(self, logp: torch.Tensor) -> torch.Tensor:
        """log P(a) − log Σ_{b≠a} P(b) sobre el último eje."""
        a = self.target_action
        others = torch.cat([logp[..., :a], logp[..., a + 1:]], dim=-1)
        return logp[..., a] - torch.logsumexp(others, dim=-1)

    def _log_odds(self, seats: torch.Tensor, glob: torch.Tensor, n: int) -> torch.Tensor:
        """(n, sillas) con el log-odds de la acción objetivo."""
        return self._odds(self._log_probs(seats, glob, n))

    def _lstm_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.supervisor._state
        if state is None:
            shape = self.policy.lstm_hidden_state_shape
            zeros = torch.zeros(shape)
            return zeros, zeros.clone()
        return torch.as_tensor(state[0], dtype=torch.float32), torch.as_tensor(state[1], dtype=torch.float32)

    # ------------------------------------------------------------ explicación
    def explain(self, obs: dict[str, np.ndarray]) -> list[dict | None]:
        normalizer = self.supervisor.normalizer
        norm = normalizer.normalize_obs(obs) if normalizer is not None else obs
        x_s = torch.as_tensor(np.asarray(norm["seats"]), dtype=torch.float32)    # (S, F)
        x_g = torch.as_tensor(np.asarray(norm["global"]), dtype=torch.float32)   # (G,)
        n_seats, m, a = x_s.shape[0], self.steps, self.target_action

        with torch.no_grad():
            logp = self._log_probs(x_s[None], x_g[None], 1)[0]                     # (S, A)
            odds = self._odds(logp)
            base_odds = self._log_odds(torch.zeros_like(x_s)[None], torch.zeros_like(x_g)[None], 1)[0]

        # Lote: para cada silla objetivo, m puntos en el camino referencia -> x (Riemann en el punto medio).
        alphas = (torch.arange(m, dtype=torch.float32) + 0.5) / m
        path_s = (alphas[:, None, None] * x_s).repeat(n_seats, 1, 1).requires_grad_(True)  # (S*m, S, F)
        path_g = (alphas[:, None] * x_g).repeat(n_seats, 1).requires_grad_(True)            # (S*m, G)
        out = self._log_odds(path_s, path_g, n_seats * m)                                   # (S*m, S)
        target_seat = torch.arange(n_seats).repeat_interleave(m)
        out[torch.arange(n_seats * m), target_seat].sum().backward()

        grad_s = path_s.grad.view(n_seats, m, *x_s.shape).mean(1)    # (silla objetivo, S, F)
        grad_g = path_g.grad.view(n_seats, m, -1).mean(1)            # (silla objetivo, G)
        attr_s = (grad_s * x_s).numpy()
        attr_g = (grad_g * x_g).numpy()

        probs = logp.exp().numpy()
        occupied = np.asarray(obs["seats"])[:, F["occupied"]] > 0
        result: list[dict | None] = []
        for s in range(n_seats):
            if not occupied[s]:
                result.append(None)
                continue
            own = attr_s[s, s]
            others = float(attr_s[s].sum() - own.sum())
            result.append({
                "score": round(float(probs[s, a]), 4),
                "probs": [round(float(p), 4) for p in probs[s]],
                "action": int(probs[s].argmax()),
                "target": round(float(odds[s]), 4),
                "base": round(float(base_odds[s]), 4),
                "attr": {name: round(float(own[i]), 4) for i, name in enumerate(SEAT_FEATURES)},
                "other_seats": round(others, 4),
                "global": round(float(attr_g[s].sum()), 4),
                "global_attr": {name: round(float(attr_g[s, i]), 4) for i, name in enumerate(GLOBAL_FEATURES)},
            })
        return result
