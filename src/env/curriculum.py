"""Curriculum de disfraz: sube la dificultad de los disimulados por etapas (sección 7.2).

El curriculum no mide nada: quien entrena le pasa las métricas de evaluación y él
decide si se avanza. Para aplicarlo: `env.table_config = curriculum.table_config()`;
la mesa nueva se crea con esa configuración en el siguiente `reset()`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from players import StealthConfig

from .table import TableConfig


@dataclass(frozen=True)
class Stage:
    disguise_level: float
    stealth_share: float
    until_detection_rate: float | None = None
    until_timesteps: int | None = None


DEFAULT_STAGES = (
    Stage(0.0, 0.05, until_detection_rate=0.90),
    Stage(0.3, 0.10, until_detection_rate=0.85),
    Stage(0.6, 0.15, until_detection_rate=0.80),
    Stage(0.9, 0.15, until_timesteps=2_000_000),
)


class Curriculum:
    def __init__(
        self,
        base_config: TableConfig | None = None,
        stages: tuple[Stage, ...] = DEFAULT_STAGES,
        max_false_positive_rate: float = 0.05,
        counter_share: float = 0.15,
    ):
        self.base_config = base_config if base_config is not None else TableConfig()
        self.stages = stages
        self.max_false_positive_rate = max_false_positive_rate
        self.counter_share = counter_share
        self.index = 0
        self._stage_timesteps = 0

    @property
    def stage(self) -> Stage:
        return self.stages[self.index]

    @property
    def finished(self) -> bool:
        s = self.stage
        return (
            self.index == len(self.stages) - 1
            and s.until_timesteps is not None
            and self._stage_timesteps >= s.until_timesteps
        )

    def table_config(self) -> TableConfig:
        s = self.stage
        return replace(
            self.base_config,
            stealth=StealthConfig.from_disguise(s.disguise_level),
            stealth_pool=(),
            population_mix={
                "casual": 1.0 - self.counter_share - s.stealth_share,
                "counter": self.counter_share,
                "stealth": s.stealth_share,
            },
        )

    def update(self, detection_rate: float, false_positive_rate: float, timesteps: int) -> bool:
        """Registra una evaluación; devuelve True si se pasó a la siguiente etapa."""
        self._stage_timesteps += timesteps
        s = self.stage
        if false_positive_rate > self.max_false_positive_rate:
            return False
        reached = (s.until_detection_rate is not None and detection_rate >= s.until_detection_rate) or (
            s.until_timesteps is not None and self._stage_timesteps >= s.until_timesteps
        )
        if not reached or self.index == len(self.stages) - 1:
            return False
        self.index += 1
        self._stage_timesteps = 0
        return True
