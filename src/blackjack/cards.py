"""Representación de cartas y conteo Hi-Lo.

Las cartas se representan como enteros por su valor de blackjack:
    1 = As, 2..9 = valor nominal, 10 = 10/J/Q/K.
Los palos no importan para el juego ni para el conteo.
"""

ACE = 1
TEN = 10

# Cartas de un mazo de 52 por valor: cuatro de cada 1..9 y dieciséis de valor 10.
DECK = [rank for rank in range(1, 10) for _ in range(4)] + [TEN] * 16

CARD_NAMES = {ACE: "A", TEN: "T", **{r: str(r) for r in range(2, 10)}}


def card_name(card: int) -> str:
    return CARD_NAMES[card]


def hi_lo_value(card: int) -> int:
    """Valor Hi-Lo: 2-6 → +1, 7-9 → 0, 10/As → -1."""
    if 2 <= card <= 6:
        return 1
    if card == TEN or card == ACE:
        return -1
    return 0


def true_count(running_count: float, decks_remaining: float) -> float:
    """TC = RC / mazos restantes (con un mínimo de medio mazo para no dividir entre ~0)."""
    return running_count / max(decks_remaining, 0.5)
