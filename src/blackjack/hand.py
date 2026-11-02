"""Mano de blackjack (jugador o dealer)."""

from __future__ import annotations

from dataclasses import dataclass, field

from .cards import ACE, card_name


@dataclass
class Hand:
    cards: list[int] = field(default_factory=list)
    bet: float = 0.0
    from_split: bool = False
    split_aces: bool = False
    doubled: bool = False
    surrendered: bool = False
    stood: bool = False

    def add(self, card: int) -> None:
        self.cards.append(card)

    @property
    def hard_total(self) -> int:
        """Total contando todos los ases como 1."""
        return sum(self.cards)

    @property
    def total(self) -> int:
        """Mejor total (un as cuenta 11 si no se pasa)."""
        t = self.hard_total
        if ACE in self.cards and t + 10 <= 21:
            return t + 10
        return t

    @property
    def is_soft(self) -> bool:
        return ACE in self.cards and self.hard_total + 10 <= 21

    @property
    def is_blackjack(self) -> bool:
        return len(self.cards) == 2 and self.total == 21 and not self.from_split

    @property
    def is_bust(self) -> bool:
        return self.total > 21

    @property
    def is_pair(self) -> bool:
        return len(self.cards) == 2 and self.cards[0] == self.cards[1]

    @property
    def is_done(self) -> bool:
        return (
            self.stood
            or self.doubled
            or self.surrendered
            or self.is_bust
            or self.total == 21
        )

    def __repr__(self) -> str:
        cards = " ".join(card_name(c) for c in self.cards)
        soft = "s" if self.is_soft else ""
        return f"Hand([{cards}] = {soft}{self.total}, bet={self.bet})"
