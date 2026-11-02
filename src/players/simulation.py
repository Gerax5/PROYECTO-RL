"""Simulación de jugadores en una mesa fija para medir su valor esperado.

No es la mesa del entorno (`src/env/table.py`): aquí nadie entra, sale ni es
expulsado. Los intervalos de confianza usan medias por lotes porque las rondas de
un mismo zapato no son independientes: la apuesta de un contador depende del conteo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from blackjack import Rules, Shoe, play_round

from .base_player import BasePlayer

BATCH_ROUNDS = 1_000
Z_95 = 1.96


@dataclass
class PlayerStats:
    player_type: str
    net_by_batch: np.ndarray    # ganancia neta en cada lote de BATCH_ROUNDS rondas
    wagered: float = 0.0        # apuestas iniciales (sin dobles, divisiones ni seguro)
    rounds_played: int = 0

    @property
    def rounds(self) -> int:
        return len(self.net_by_batch) * BATCH_ROUNDS

    @property
    def net(self) -> float:
        return float(self.net_by_batch.sum())

    @property
    def edge_pct(self) -> float:
        """Ganancia por unidad de apuesta inicial, en %."""
        return 100 * self.net / self.wagered

    @property
    def units_per_100(self) -> float:
        return 100 * self.net / self.rounds

    @property
    def ci95_per_100(self) -> tuple[float, float]:
        """Intervalo de confianza del 95% de `units_per_100`."""
        sem = self.net_by_batch.std(ddof=1) / np.sqrt(len(self.net_by_batch))
        half = Z_95 * sem * 100 / BATCH_ROUNDS
        return self.units_per_100 - half, self.units_per_100 + half

    def merge(self, other: "PlayerStats") -> "PlayerStats":
        return PlayerStats(
            self.player_type,
            np.concatenate([self.net_by_batch, other.net_by_batch]),
            self.wagered + other.wagered,
            self.rounds_played + other.rounds_played,
        )


def simulate_table(
    players: Sequence[BasePlayer],
    rounds: int,
    seed: int,
    rules: Rules | None = None,
    n_decks: int = 6,
    penetration: float = 0.75,
) -> list[PlayerStats]:
    """Juega `rounds` rondas con `players` sentados en sillas consecutivas.

    `rounds` debe ser múltiplo de BATCH_ROUNDS. Si en una ronda nadie apuesta, la
    ronda cuenta pero no se reparte nada, así que no se puede dejar sola en la mesa
    a un jugador que deja de apostar según el conteo: nunca volvería a ver cartas.
    """
    if rounds < 2 * BATCH_ROUNDS or rounds % BATCH_ROUNDS:
        raise ValueError(f"rounds debe ser un múltiplo de {BATCH_ROUNDS} y al menos el doble")

    rules = rules if rules is not None else Rules()
    shoe = Shoe(n_decks=n_decks, penetration=penetration, rng=np.random.default_rng(seed))
    stats = [
        PlayerStats(p.player_type, np.zeros(rounds // BATCH_ROUNDS))
        for p in players
    ]

    def on_card(card: int) -> None:
        for p in players:
            p.observe_card(card, shoe.decks_remaining)

    def decide(seat, hand, up, allowed):
        return players[seat].decide(hand, up, allowed)

    def insure(seat, hand, up):
        return players[seat].insure(hand, up)

    for p in players:
        p.on_shuffle(shoe.decks_remaining)

    for i in range(rounds):
        if shoe.needs_shuffle:
            shoe.shuffle()
            for p in players:
                p.on_shuffle(shoe.decks_remaining)

        bets = {seat: p.place_bet() for seat, p in enumerate(players)}
        if not any(bets.values()):
            continue
        result = play_round(shoe, rules, bets, decide, insure, on_card)

        for seat_result in result.seats:
            seat = seat_result.seat
            stats[seat].net_by_batch[i // BATCH_ROUNDS] += seat_result.net
            stats[seat].wagered += bets[seat]
            stats[seat].rounds_played += 1
            players[seat].on_round_end(seat_result.net)

    return stats
