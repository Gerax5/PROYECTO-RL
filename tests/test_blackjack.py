from collections import Counter

import numpy as np
import pytest

from blackjack import (
    ACE,
    TEN,
    Action,
    Hand,
    Rules,
    Shoe,
    basic_strategy,
    counter_strategy,
    hi_lo_value,
    play_round,
    take_insurance,
    true_count,
)

S17 = Rules()
H17 = Rules(dealer_hits_soft_17=True)
NO_DAS = Rules(double_after_split=False)


class StackedShoe:
    """Zapato con cartas en orden fijo, para tests deterministas."""

    def __init__(self, cards):
        self.cards = list(cards)

    def draw(self):
        return self.cards.pop(0)


def h(*cards, **kw):
    return Hand(cards=list(cards), bet=kw.pop("bet", 10), **kw)


def always(action):
    return lambda seat, hand, up, allowed: action


def basic_player(rules):
    def decide(seat, hand, up, allowed):
        return basic_strategy(
            hand, up, rules,
            can_double=Action.DOUBLE in allowed,
            can_split=Action.SPLIT in allowed,
            can_surrender=Action.SURRENDER in allowed,
        )
    return decide


# ---------------------------------------------------------------- cartas / shoe

def test_deck_composition():
    shoe = Shoe(n_decks=6, burn_cards=0, rng=np.random.default_rng(0))
    cards = [shoe.draw() for _ in range(312)]
    counts = Counter(cards)
    assert counts[TEN] == 96
    assert all(counts[r] == 24 for r in range(1, 10))


def test_hi_lo_balanced():
    shoe = Shoe(n_decks=6, burn_cards=0, rng=np.random.default_rng(1))
    assert sum(hi_lo_value(shoe.draw()) for _ in range(312)) == 0


def test_shoe_penetration_and_reshuffle():
    shoe = Shoe(n_decks=6, penetration=0.75, burn_cards=1, rng=np.random.default_rng(2))
    assert shoe.cards_dealt == 1
    while not shoe.needs_shuffle:
        shoe.draw()
    assert shoe.cards_dealt == int(312 * 0.75)
    assert shoe.decks_remaining == pytest.approx(78 / 52)
    shoe.shuffle()
    assert not shoe.needs_shuffle
    assert shoe.shuffle_count == 2


def test_shoe_seed_is_deterministic():
    a = Shoe(rng=np.random.default_rng(7))
    b = Shoe(rng=np.random.default_rng(7))
    assert [a.draw() for _ in range(50)] == [b.draw() for _ in range(50)]


def test_true_count():
    assert true_count(6, 3) == 2
    assert true_count(4, 0.1) == 8   # mínimo de medio mazo


# ---------------------------------------------------------------- mano

def test_hand_totals():
    assert h(ACE, 6).total == 17 and h(ACE, 6).is_soft
    assert h(ACE, 6, TEN).total == 17 and not h(ACE, 6, TEN).is_soft
    assert h(ACE, ACE).total == 12
    assert h(TEN, 6, 8).is_bust
    assert h(ACE, TEN).is_blackjack
    assert not h(ACE, TEN, from_split=True).is_blackjack
    assert not h(7, 7, 7).is_blackjack


# ---------------------------------------------------------------- estrategia básica

@pytest.mark.parametrize("cards, up, expected", [
    ((TEN, 6), TEN, Action.HIT),
    ((TEN, 6), 6, Action.STAND),
    ((TEN, 2), 2, Action.HIT),
    ((TEN, 2), 4, Action.STAND),
    ((6, 5), ACE, Action.HIT),       # S17
    ((6, 5), TEN, Action.DOUBLE),
    ((5, 4), 3, Action.DOUBLE),
    ((5, 4), 2, Action.HIT),
    ((5, 5), 9, Action.DOUBLE),      # 5,5 se juega como 10 duro
    ((ACE, 7), 2, Action.STAND),
    ((ACE, 7), 4, Action.DOUBLE),
    ((ACE, 7), 9, Action.HIT),
    ((ACE, 6), 3, Action.DOUBLE),
    ((ACE, ACE), TEN, Action.SPLIT),
    ((8, 8), TEN, Action.SPLIT),
    ((9, 9), 7, Action.STAND),
    ((9, 9), 8, Action.SPLIT),
    ((TEN, TEN), 6, Action.STAND),
    ((2, 2), 2, Action.SPLIT),       # DAS
    ((4, 4), 5, Action.SPLIT),       # DAS
])
def test_basic_strategy_s17(cards, up, expected):
    assert basic_strategy(h(*cards), up, S17) == expected


def test_basic_strategy_double_fallback():
    assert basic_strategy(h(6, 5), 6, S17, can_double=False) == Action.HIT
    assert basic_strategy(h(ACE, 7), 4, S17, can_double=False) == Action.STAND


def test_basic_strategy_no_das():
    assert basic_strategy(h(2, 2), 2, NO_DAS) == Action.HIT
    assert basic_strategy(h(4, 4), 5, NO_DAS) == Action.HIT      # se juega como 8 duro
    assert basic_strategy(h(6, 6), 2, NO_DAS) == Action.HIT


def test_basic_strategy_h17():
    assert basic_strategy(h(6, 5), ACE, H17) == Action.DOUBLE
    assert basic_strategy(h(ACE, 7), 2, H17) == Action.DOUBLE
    assert basic_strategy(h(ACE, 8), 6, H17) == Action.DOUBLE


def test_surrender():
    assert basic_strategy(h(TEN, 6), TEN, S17, can_surrender=True) == Action.SURRENDER
    assert basic_strategy(h(TEN, 5), TEN, S17, can_surrender=True) == Action.SURRENDER
    assert basic_strategy(h(TEN, 5), ACE, S17, can_surrender=True) == Action.HIT
    assert basic_strategy(h(TEN, 5), ACE, H17, can_surrender=True) == Action.SURRENDER
    assert basic_strategy(h(8, 8), TEN, S17, can_surrender=True) == Action.SPLIT
    assert basic_strategy(h(8, 8), ACE, H17, can_surrender=True) == Action.SURRENDER


# ---------------------------------------------------------------- Illustrious 18

@pytest.mark.parametrize("cards, up, tc, expected, deviation", [
    ((TEN, 6), TEN, 0, Action.STAND, True),
    ((TEN, 6), TEN, -1, Action.HIT, False),
    ((7, 4, 5), TEN, 1, Action.STAND, True),       # aplica a manos de 3 cartas
    ((TEN, 5), TEN, 4, Action.STAND, True),
    ((TEN, 5), TEN, 3.9, Action.HIT, False),
    ((TEN, TEN), 6, 4, Action.SPLIT, True),
    ((TEN, TEN), 5, 4, Action.STAND, False),
    ((TEN, TEN), 5, 5, Action.SPLIT, True),
    ((6, 4), TEN, 4, Action.DOUBLE, True),
    ((6, 5), ACE, 1, Action.DOUBLE, True),
    ((5, 4), 2, 1, Action.DOUBLE, True),
    ((5, 4), 7, 3, Action.DOUBLE, True),
    ((TEN, 2), 3, 2, Action.STAND, True),
    ((TEN, 2), 2, 3, Action.STAND, True),
    ((TEN, 2), 4, -1, Action.HIT, True),           # índice negativo
    ((TEN, 2), 4, 0, Action.STAND, False),
    ((TEN, 3), 2, -2, Action.HIT, True),
    ((TEN, 3), 2, -1, Action.STAND, False),
    ((TEN, 2), 6, -1.5, Action.HIT, True),
    ((TEN, 6), 9, 5, Action.STAND, True),
])
def test_illustrious_18(cards, up, tc, expected, deviation):
    hand = h(*cards)
    action, play = counter_strategy(hand, up, tc, S17)
    assert play is not None
    assert action == expected
    assert (action != basic_strategy(hand, up, S17)) == deviation


def test_index_play_not_applicable():
    # 8,8 vs 10 se divide: no se aplica "16 vs 10"
    action, play = counter_strategy(h(8, 8), TEN, 5, S17)
    assert action == Action.SPLIT and play is None
    # 16 suave no es 16 duro
    _, play = counter_strategy(h(ACE, 5), TEN, 5, S17)
    assert play is None
    # no se puede doblar con 3 cartas → no aplica "10 vs 10"
    _, play = counter_strategy(h(3, 3, 4), TEN, 5, S17, can_double=False)
    assert play is None


def test_insurance_index():
    assert take_insurance(3)
    assert not take_insurance(2.9)


# ---------------------------------------------------------------- motor

def test_player_blackjack_pays_3_to_2():
    # silla 0: A, T  | dealer: 9, 7
    shoe = StackedShoe([ACE, 9, TEN, 7])
    res = play_round(shoe, S17, {0: 10}, always(Action.STAND))
    assert res.seats[0].net == 15


def test_dealer_blackjack_takes_only_original_bet():
    # silla 0: T, 9 | dealer: A, T → seguro tomado paga 2:1
    shoe = StackedShoe([TEN, ACE, 9, TEN])
    res = play_round(shoe, S17, {0: 10}, always(Action.STAND), insure=lambda *a: True)
    assert res.dealer_blackjack
    assert res.seats[0].net == -10 + 10   # pierde 10, seguro de 5 gana 10


def test_insurance_lost_without_dealer_blackjack():
    # silla 0: T, 9 = 19 | dealer: A, 6 = 17 suave (S17 se planta)
    shoe = StackedShoe([TEN, ACE, 9, 6])
    res = play_round(shoe, S17, {0: 10}, always(Action.STAND), insure=lambda *a: True)
    assert not res.dealer_blackjack
    assert res.seats[0].net == 10 - 5


def test_double():
    # silla 0: 6, 5 | dealer: 6, T | dobla: T → 21 | dealer 16 pide 2 → 18
    shoe = StackedShoe([6, 6, 5, TEN, TEN, 2])
    res = play_round(shoe, S17, {0: 10}, always(Action.DOUBLE))
    assert res.seats[0].net == 20


def test_split_and_dealer_bust():
    # silla 0: 8, 8 | dealer: 6, T
    # split → mano1: 8 + 3 = 11 (dobla: T → 21) ; mano2: 8 + T = 18 (planta) ; dealer 16 + T = bust
    shoe = StackedShoe([8, 6, 8, TEN, 3, TEN, TEN, TEN])
    res = play_round(shoe, S17, {0: 10}, basic_player(S17))
    seat = res.seats[0]
    assert len(seat.hands) == 2
    assert seat.actions == [Action.SPLIT, Action.DOUBLE, Action.STAND]
    assert seat.net == 20 + 10


def test_split_aces_one_card_each():
    # silla 0: A, A | dealer: 9, 7 | manos: A+T = 21, A+5 = 16 | dealer pide 2 → 18
    shoe = StackedShoe([ACE, 9, ACE, 7, TEN, 5, 2])
    res = play_round(shoe, S17, {0: 10}, basic_player(S17))
    seat = res.seats[0]
    assert [h.total for h in seat.hands] == [21, 16]
    assert not seat.hands[0].is_blackjack   # 21 tras dividir no es blackjack
    assert seat.actions == [Action.SPLIT]   # no se pide sobre ases divididos
    assert res.dealer.total == 18
    assert seat.net == 10 - 10


def test_dealer_soft_17():
    # silla 0: T, 8 | dealer: A, 6 (17 suave). S17 se planta; H17 pide 3 → 20
    cards = [TEN, ACE, 8, 6, 3]
    res = play_round(StackedShoe(cards), S17, {0: 10}, always(Action.STAND))
    assert res.dealer.total == 17 and res.seats[0].net == 10
    res = play_round(StackedShoe(cards), H17, {0: 10}, always(Action.STAND))
    assert res.dealer.total == 20 and res.seats[0].net == -10


def test_surrender_returns_half():
    rules = Rules(surrender=True)
    shoe = StackedShoe([TEN, 9, 6, 8])
    res = play_round(shoe, rules, {0: 10}, always(Action.SURRENDER))
    assert res.seats[0].net == -5


def test_hole_card_hidden_until_reveal():
    # silla 0: T, 7 | dealer: 9, 8 (oculta)
    seen = []
    shoe = StackedShoe([TEN, 9, 7, 8])
    play_round(shoe, S17, {0: 10}, always(Action.STAND), on_card=seen.append)
    assert seen == [TEN, 9, 7, 8]       # la oculta (8) se ve al final

    seen.clear()
    seen_at_decision = []
    shoe = StackedShoe([TEN, 9, 7, 8])
    play_round(shoe, S17, {0: 10},
               lambda *a: seen_at_decision.extend(seen) or Action.STAND,
               on_card=seen.append)
    assert 8 not in seen_at_decision


def test_illegal_action_raises():
    shoe = StackedShoe([TEN, 9, 8, 7])
    with pytest.raises(ValueError):
        play_round(shoe, S17, {0: 10}, always(Action.SPLIT))


def test_multiple_seats():
    # sillas 0 y 3 | reparto: s0, s3, dealer, s0, s3, oculta
    shoe = StackedShoe([TEN, 9, 6, 9, 9, TEN, TEN])
    res = play_round(shoe, S17, {0: 10, 3: 20}, always(Action.STAND))
    assert [r.seat for r in res.seats] == [0, 3]
    assert res.dealer.is_bust
    assert [r.net for r in res.seats] == [10, 20]
