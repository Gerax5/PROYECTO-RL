"""Contrato de eventos entre la mesa (A) y el extractor de features (B); sección 11.2."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SeatEvent:
    round_id: int
    seat: int
    player_id: int | None
    event: str                  # "sit" | "play" | "leave" | "expelled"
    bet: float                  # en unidades; 0 si el jugador no apostó en la ronda
    true_count: float           # TC público al inicio de la ronda
    actions: list[str] = field(default_factory=list)
    deviated_from_basic: bool = False
    matched_index_play: bool = False
    result_units: float = 0.0
    shoe_penetration: float = 0.0
    true_type: str = ""         # SOLO para métricas/etiquetas, nunca en la observación
