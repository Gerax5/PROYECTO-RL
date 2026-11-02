"""Reglas de la mesa y acciones posibles."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum


class Action(str, Enum):
    HIT = "hit"
    STAND = "stand"
    DOUBLE = "double"
    SPLIT = "split"
    SURRENDER = "surrender"


@dataclass(frozen=True)
class Rules:
    dealer_hits_soft_17: bool = False   # False = S17
    blackjack_payout: float = 1.5       # 3:2
    double_after_split: bool = True     # DAS
    max_splits: int = 3                 # hasta 4 manos
    resplit_aces: bool = False
    hit_split_aces: bool = False        # ases divididos reciben una sola carta
    surrender: bool = False             # late surrender
    insurance: bool = True
    dealer_peeks: bool = True           # el dealer revisa BJ con As o 10 (regla americana)

    @classmethod
    def from_dict(cls, cfg: dict) -> "Rules":
        """Construye desde la sección `rules` de `configs/env.yaml` ignorando claves extra."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in cfg.items() if k in names})
