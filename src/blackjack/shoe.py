"""Zapato (shoe) de varios mazos con carta de corte."""

from __future__ import annotations

import numpy as np

from .cards import DECK


class Shoe:
    """Zapato de `n_decks` mazos.

    Al pasar la carta de corte (`penetration` del zapato repartido) se marca
    `needs_shuffle`; la mesa termina la ronda en curso y luego llama a `shuffle()`.
    Si el zapato se agotara a mitad de ronda, se rebaraja automáticamente.
    """

    def __init__(
        self,
        n_decks: int = 6,
        penetration: float = 0.75,
        burn_cards: int = 1,
        rng: np.random.Generator | None = None,
    ):
        if n_decks < 1:
            raise ValueError("n_decks debe ser >= 1")
        if not 0.0 < penetration < 1.0:
            raise ValueError("penetration debe estar en (0, 1)")
        self.n_decks = n_decks
        self.penetration = penetration
        self.burn_cards = burn_cards
        self.rng = rng if rng is not None else np.random.default_rng()
        self.total_cards = 52 * n_decks
        self.cut_index = int(self.total_cards * penetration)
        self.shuffle_count = 0
        self.shuffle()

    def shuffle(self) -> None:
        self._cards = np.array(DECK * self.n_decks, dtype=np.int8)
        self.rng.shuffle(self._cards)
        self._pos = 0
        self.shuffle_count += 1
        # Las cartas quemadas no se muestran: no cuentan para el conteo público.
        self._pos += self.burn_cards

    def draw(self) -> int:
        if self._pos >= self.total_cards:
            self.shuffle()
        card = int(self._cards[self._pos])
        self._pos += 1
        return card

    @property
    def cards_dealt(self) -> int:
        return self._pos

    @property
    def cards_remaining(self) -> int:
        return self.total_cards - self._pos

    @property
    def decks_remaining(self) -> float:
        return self.cards_remaining / 52

    @property
    def penetration_used(self) -> float:
        """Fracción del zapato ya repartida (información pública)."""
        return self._pos / self.total_cards

    @property
    def needs_shuffle(self) -> bool:
        return self._pos >= self.cut_index
