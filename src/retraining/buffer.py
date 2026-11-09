"""Mezcla de escenarios para el fine-tuning (sección 7.3).

PPO aprende de datos de su propia política, así que no se reutilizan transiciones
viejas: lo que se "repite" son los escenarios. Una fracción `replay_ratio_old` de las
mesas recibe disimulados con camuflajes de ciclos anteriores (evita olvidar cómo
detectarlos) y el resto los camuflajes nuevos de este ciclo.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace

from env import TableConfig
from players import StealthConfig


class ScenarioBuffer:
    def __init__(self, capacity: int = 10, replay_ratio_old: float = 0.3):
        if not 0.0 <= replay_ratio_old <= 1.0:
            raise ValueError("replay_ratio_old debe estar en [0, 1]")
        self.replay_ratio_old = replay_ratio_old
        self.new: list[StealthConfig] = []
        self.old: deque[StealthConfig] = deque(maxlen=capacity)

    def add(self, stealth: StealthConfig) -> None:
        self.new.append(stealth)

    def end_cycle(self) -> None:
        self.old.extend(self.new)
        self.new = []

    def table_configs(self, n_envs: int, base: TableConfig) -> list[TableConfig]:
        """Una configuración por mesa: primero las nuevas, después las históricas."""
        new_pool = tuple(self.new) if self.new else (base.stealth,)
        n_old = round(n_envs * self.replay_ratio_old) if self.old else 0
        n_old = min(n_old, n_envs - 1) if n_envs > 1 else 0
        new = replace(base, stealth_pool=new_pool)
        old = replace(base, stealth_pool=tuple(self.old))
        return [new] * (n_envs - n_old) + [old] * n_old
