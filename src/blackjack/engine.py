"""Motor de una ronda de blackjack para varias sillas.

El motor no sabe quién juega: recibe callbacks para las decisiones, de modo que
la mesa (`src/env/table.py`) puede conectar cualquier tipo de jugador.

Orden de la ronda (reglas americanas, el dealer revisa blackjack):
    1. Reparto: una carta a cada silla, carta visible del dealer, segunda carta
       a cada silla, carta oculta del dealer.
    2. Seguro (si la visible es As).
    3. El dealer revisa blackjack con As o 10; si lo tiene, la ronda termina.
    4. Cada silla juega sus manos (pedir, plantarse, doblar, dividir, rendirse).
    5. El dealer revela la oculta y juega.
    6. Pagos.

`on_card` se llama con cada carta en el momento en que se vuelve pública
(la oculta del dealer, al revelarse): es lo que permite contar cartas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from .cards import ACE, TEN
from .hand import Hand
from .rules import Action, Rules


class CardSource(Protocol):
    def draw(self) -> int: ...


# decide(seat, hand, dealer_up, allowed) -> Action
DecideFn = Callable[[int, Hand, int, frozenset[Action]], Action]
# insure(seat, hand, dealer_up) -> bool
InsureFn = Callable[[int, Hand, int], bool]
OnCardFn = Callable[[int], None]


@dataclass
class SeatResult:
    seat: int
    hands: list[Hand]
    actions: list[Action] = field(default_factory=list)
    insurance_bet: float = 0.0
    net: float = 0.0            # ganancia neta de la silla en la ronda (incluye seguro)


@dataclass
class RoundResult:
    seats: list[SeatResult]
    dealer: Hand
    dealer_blackjack: bool


def dealer_should_hit(hand: Hand, rules: Rules) -> bool:
    total = hand.total
    return total < 17 or (total == 17 and hand.is_soft and rules.dealer_hits_soft_17)


def settle_hand(hand: Hand, dealer: Hand, rules: Rules) -> float:
    """Ganancia neta de una mano del jugador (sin seguro)."""
    if hand.surrendered:
        return -hand.bet / 2
    if hand.is_bust:
        return -hand.bet
    if dealer.is_blackjack:
        return 0.0 if hand.is_blackjack else -hand.bet
    if hand.is_blackjack:
        return hand.bet * rules.blackjack_payout
    if dealer.is_bust or hand.total > dealer.total:
        return hand.bet
    if hand.total < dealer.total:
        return -hand.bet
    return 0.0


def allowed_actions(hand: Hand, n_hands: int, rules: Rules) -> frozenset[Action]:
    two_cards = len(hand.cards) == 2
    can_split = (
        hand.is_pair
        and n_hands <= rules.max_splits
        and (not hand.split_aces or rules.resplit_aces)
    )

    if hand.split_aces and not rules.hit_split_aces:
        acts = {Action.STAND}
        if can_split:
            acts.add(Action.SPLIT)
        return frozenset(acts)

    acts = {Action.HIT, Action.STAND}
    if two_cards and (not hand.from_split or rules.double_after_split):
        acts.add(Action.DOUBLE)
    if can_split:
        acts.add(Action.SPLIT)
    if rules.surrender and two_cards and n_hands == 1 and not hand.from_split:
        acts.add(Action.SURRENDER)
    return frozenset(acts)


def play_round(
    shoe: CardSource,
    rules: Rules,
    bets: dict[int, float],
    decide: DecideFn,
    insure: InsureFn | None = None,
    on_card: OnCardFn | None = None,
) -> RoundResult:
    """Juega una ronda completa. `bets` = {silla: apuesta} sólo de sillas ocupadas."""
    seats = sorted(s for s, b in bets.items() if b > 0)

    def deal(hand: Hand, visible: bool = True) -> int:
        card = shoe.draw()
        hand.add(card)
        if visible and on_card is not None:
            on_card(card)
        return card

    results = {s: SeatResult(seat=s, hands=[Hand(bet=bets[s])]) for s in seats}
    dealer = Hand()

    # 1. Reparto
    for s in seats:
        deal(results[s].hands[0])
    deal(dealer)
    for s in seats:
        deal(results[s].hands[0])
    deal(dealer, visible=False)
    up = dealer.cards[0]

    def reveal_hole() -> None:
        if on_card is not None:
            on_card(dealer.cards[1])

    # 2. Seguro
    if up == ACE and rules.insurance and insure is not None:
        for s in seats:
            r = results[s]
            if insure(s, r.hands[0], up):
                r.insurance_bet = r.hands[0].bet / 2

    # 3. Revisión de blackjack del dealer
    if up in (ACE, TEN) and dealer.is_blackjack:
        reveal_hole()
        for s in seats:
            r = results[s]
            r.net = settle_hand(r.hands[0], dealer, rules) + 2 * r.insurance_bet
        return RoundResult([results[s] for s in seats], dealer, dealer_blackjack=True)

    # 4. Turno de los jugadores
    for s in seats:
        r = results[s]
        i = 0
        while i < len(r.hands):
            hand = r.hands[i]
            if hand.is_blackjack:
                i += 1
                continue
            while not hand.is_done:
                allowed = allowed_actions(hand, len(r.hands), rules)
                if allowed == {Action.STAND}:
                    hand.stood = True
                    break
                action = decide(s, hand, up, allowed)
                if action not in allowed:
                    raise ValueError(f"silla {s}: acción {action} no permitida {set(allowed)}")
                r.actions.append(action)

                if action == Action.HIT:
                    deal(hand)
                elif action == Action.STAND:
                    hand.stood = True
                elif action == Action.DOUBLE:
                    hand.bet *= 2
                    hand.doubled = True
                    deal(hand)
                elif action == Action.SURRENDER:
                    hand.surrendered = True
                elif action == Action.SPLIT:
                    aces = hand.cards[0] == ACE
                    new = Hand(
                        cards=[hand.cards.pop()],
                        bet=hand.bet,
                        from_split=True,
                        split_aces=aces,
                    )
                    hand.from_split = True
                    hand.split_aces = aces
                    deal(hand)
                    deal(new)
                    r.hands.insert(i + 1, new)
            i += 1

    # 5. Turno del dealer
    reveal_hole()
    live = any(
        not h.is_bust and not h.surrendered and not h.is_blackjack
        for s in seats
        for h in results[s].hands
    )
    if live:
        while dealer_should_hit(dealer, rules):
            deal(dealer)

    # 6. Pagos (el seguro se pierde porque el dealer no tenía blackjack)
    for s in seats:
        r = results[s]
        r.net = sum(settle_hand(h, dealer, rules) for h in r.hands) - r.insurance_bet

    return RoundResult([results[s] for s in seats], dealer, dealer_blackjack=False)
