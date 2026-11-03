"""Métricas del supervisor (sección 8), calculadas a partir de los `info` del ambiente.

Usa `true_type` de los eventos: es la etiqueta del simulador y sólo sirve para medir.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

COUNTER_TYPES = ("counter", "stealth")
PLAYER_TYPES = ("casual", *COUNTER_TYPES)


def _ratio(num: float, den: float) -> float:
    return num / den if den else 0.0


@dataclass
class SupervisorMetrics:
    seen: Counter = field(default_factory=Counter)          # jugadores sentados, por tipo
    expelled: Counter = field(default_factory=Counter)      # expulsados, por tipo
    detection_rounds: list[int] = field(default_factory=list)  # rondas_vistas al expulsar a un contador
    rounds: int = 0
    reward: float = 0.0
    house_net: float = 0.0
    counter_rounds: int = 0       # rondas con apuesta jugadas por contadores
    counter_net: float = 0.0      # lo que ganaron los contadores (unidades)

    def update(self, info: dict, reward: float = 0.0) -> None:
        """Acepta el `info` de `reset` (sólo eventos) y el de cada `step`."""
        for e in info["events"]:
            if e.event == "sit":
                self.seen[e.true_type] += 1
            elif e.event == "play" and e.true_type in COUNTER_TYPES and e.bet > 0:
                self.counter_rounds += 1
                self.counter_net += e.result_units
        for x in info.get("expulsions", ()):
            self.expelled[x["true_type"]] += 1
            if x["true_type"] in COUNTER_TYPES:
                self.detection_rounds.append(x["rounds_seen"])
        if "house_net" in info:
            self.rounds += 1
            self.house_net += info["house_net"]
            self.reward += reward

    def merge(self, other: "SupervisorMetrics") -> "SupervisorMetrics":
        return SupervisorMetrics(
            seen=self.seen + other.seen,
            expelled=self.expelled + other.expelled,
            detection_rounds=self.detection_rounds + other.detection_rounds,
            rounds=self.rounds + other.rounds,
            reward=self.reward + other.reward,
            house_net=self.house_net + other.house_net,
            counter_rounds=self.counter_rounds + other.counter_rounds,
            counter_net=self.counter_net + other.counter_net,
        )

    def recall(self, kind: str) -> float:
        return _ratio(self.expelled[kind], self.seen[kind])

    @property
    def false_positive_rate(self) -> float:
        return self.recall("casual")

    @property
    def precision(self) -> float:
        hits = sum(self.expelled[k] for k in COUNTER_TYPES)
        return _ratio(hits, sum(self.expelled.values()))

    @property
    def f1_counters(self) -> float:
        hits = sum(self.expelled[k] for k in COUNTER_TYPES)
        recall = _ratio(hits, sum(self.seen[k] for k in COUNTER_TYPES))
        return _ratio(2 * self.precision * recall, self.precision + recall)

    def summary(self) -> dict[str, float]:
        return {
            "recall_counter": self.recall("counter"),
            "recall_stealth": self.recall("stealth"),
            "fp_rate": self.false_positive_rate,
            "precision": self.precision,
            "f1_counters": self.f1_counters,
            "detection_rounds": _ratio(sum(self.detection_rounds), len(self.detection_rounds)),
            "house_net_per_100": 100 * _ratio(self.house_net, self.rounds),
            "counter_net_per_100": 100 * _ratio(self.counter_net, self.rounds),
            "counter_rounds_per_100": 100 * _ratio(self.counter_rounds, self.rounds),
            "reward_per_100": 100 * _ratio(self.reward, self.rounds),
        }
