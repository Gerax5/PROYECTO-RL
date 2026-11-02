import numpy as np
import pytest

from blackjack import ACE, TEN, Action, Hand, Rules, basic_strategy
from players import (
    CasualConfig,
    CasualPlayer,
    CounterConfig,
    CounterPlayer,
    StealthConfig,
    StealthCounterPlayer,
    ramp_bet,
    simulate_table,
)
from players.base_player import plausible_mistakes

ALL_ACTIONS = frozenset({Action.HIT, Action.STAND, Action.DOUBLE})


def h(*cards):
    return Hand(cards=list(cards), bet=1)


def rng(seed=0):
    return np.random.default_rng(seed)


def counter_at_tc(cls=CounterPlayer, running_count=12, decks=3.0, **kw):
    player = cls(rng=rng(), **kw)
    player.on_shuffle(6.0)
    player.running_count = running_count
    player.decks_remaining = decks
    return player


def casual(style, **kw):
    return CasualPlayer(CasualConfig(bet_style=style, **kw), rng=rng())


# ---------------------------------------------------------------- errores

def test_plausible_mistakes_excludes_absurd_plays():
    assert plausible_mistakes(h(TEN, TEN), Action.STAND, ALL_ACTIONS) == []
    assert Action.STAND not in plausible_mistakes(h(5, 3), Action.HIT, ALL_ACTIONS)
    assert Action.DOUBLE not in plausible_mistakes(h(TEN, 5), Action.STAND, ALL_ACTIONS)
    assert plausible_mistakes(h(TEN, 6), Action.HIT, ALL_ACTIONS) == [Action.STAND]


# ---------------------------------------------------------------- casual

def test_casual_hot_streak_raises_after_win_and_resets_after_loss():
    p = casual("hot_streak")
    assert p.place_bet() == 1
    p.on_round_end(1)
    assert p.place_bet() == 2
    p.on_round_end(0)
    assert p.place_bet() == 2
    p.on_round_end(-2)
    assert p.place_bet() == 1


def test_casual_martingale_doubles_after_loss_up_to_cap():
    p = casual("martingale", max_bet_units=4)
    for expected in [2, 4, 4]:
        p.on_round_end(-1)
        assert p.place_bet() == expected
    p.on_round_end(4)
    assert p.place_bet() == 1


def test_casual_flat_noise_stays_in_chips_and_above_minimum():
    p = casual("flat_noise")
    bets = {p.place_bet() for _ in range(500)}
    assert min(bets) == 1 and len(bets) > 1
    assert all(b * 2 == int(b * 2) for b in bets)


def test_casual_ignores_the_count():
    a, b = casual("flat_noise"), casual("flat_noise")
    for _ in range(50):
        b.observe_card(5, 3.0)
    assert [a.place_bet() for _ in range(20)] == [b.place_bet() for _ in range(20)]


def test_casual_error_rate_is_sampled_in_range_and_causes_errors():
    p = casual("flat_noise")
    assert 0.05 <= p.error_rate <= 0.20
    hand = h(TEN, 6)
    best = basic_strategy(hand, TEN, p.rules)
    plays = [p.decide(hand, TEN, ALL_ACTIONS) for _ in range(2000)]
    assert 0.03 < sum(a != best for a in plays) / 2000 < 0.22


def test_casual_unknown_style_raises():
    with pytest.raises(ValueError):
        casual("kelly")


# ---------------------------------------------------------------- contador

@pytest.mark.parametrize("tc, bet", [
    (-3, 1), (0, 1), (1.9, 1), (2, 2), (3.5, 4), (4, 8), (5, 12), (9, 12),
])
def test_ramp(tc, bet):
    assert ramp_bet(tc) == bet


def test_counter_tracks_hi_lo_and_resets_on_shuffle():
    p = CounterPlayer(rng=rng())
    p.on_shuffle(6.0)
    for card in [2, 6, 5, 8, TEN]:
        p.observe_card(card, 4.0)
    assert p.running_count == 2 and p.tc == pytest.approx(0.5)
    p.on_shuffle(6.0)
    assert p.running_count == 0


def test_counter_applies_index_plays_and_insurance():
    hot = counter_at_tc(running_count=12)
    cold = counter_at_tc(running_count=-12)
    assert hot.decide(h(TEN, 6), TEN, ALL_ACTIONS) == Action.STAND
    assert cold.decide(h(TEN, 6), TEN, ALL_ACTIONS) == Action.HIT
    assert hot.insure(h(TEN, 9), ACE)
    assert not cold.insure(h(TEN, 9), ACE)


def test_wonging_sits_out_on_negative_count():
    cfg = CounterConfig(wonging=True, wong_out_tc=-1)
    assert counter_at_tc(running_count=-6, config=cfg).place_bet() == 0
    assert counter_at_tc(running_count=-3, config=cfg).place_bet() == 1
    assert counter_at_tc(running_count=-6).place_bet() == 1


# ---------------------------------------------------------------- disimulado

def test_disguise_interpolation_endpoints():
    assert StealthConfig.from_disguise(1.0) == StealthConfig()
    zero = StealthConfig.from_disguise(0.0)
    assert zero.bet_spread_max == 12 and zero.index_play_prob == 1
    assert zero.bet_noise_std == zero.cover_bet_prob == zero.deliberate_error_prob == 0
    assert zero.delay_bet_ramp_rounds == 0
    with pytest.raises(ValueError):
        StealthConfig.from_disguise(1.5)


def test_disguise_zero_plays_exactly_like_open_counter():
    a = simulate_table([CounterPlayer(rng=rng(1))], 5_000, seed=3)[0]
    stealth = StealthCounterPlayer(StealthConfig.from_disguise(0.0), rng=rng(1))
    b = simulate_table([stealth], 5_000, seed=3)[0]
    assert a.net == b.net and a.wagered == b.wagered


def test_stealth_bet_never_exceeds_spread():
    p = counter_at_tc(StealthCounterPlayer, running_count=30, stealth=StealthConfig())
    bets = [p.place_bet() for _ in range(500)]
    assert max(bets) == 4 and min(bets) >= 1


def test_stealth_delays_bet_ramp():
    cfg = StealthConfig(cover_bet_prob=0, bet_noise_std=0, delay_bet_ramp_rounds=1)
    p = counter_at_tc(StealthCounterPlayer, running_count=-3, stealth=cfg)
    assert p.place_bet() == 1
    p.running_count = 30
    assert p.place_bet() == 1      # todavía usa el TC de la ronda anterior
    assert p.place_bet() == 4


def test_stealth_index_play_prob_mixes_basic_and_deviation():
    cfg = StealthConfig(index_play_prob=0.5, deliberate_error_prob=0)
    p = counter_at_tc(StealthCounterPlayer, running_count=12, stealth=cfg)
    plays = [p.decide(h(TEN, 6), TEN, ALL_ACTIONS) for _ in range(1000)]
    assert 0.4 < plays.count(Action.STAND) / 1000 < 0.6
    assert set(plays) == {Action.STAND, Action.HIT}


# ---------------------------------------------------------------- simulación

def test_simulation_is_reproducible():
    run = lambda: simulate_table([casual("hot_streak"), CounterPlayer(rng=rng(1))], 5_000, seed=9)
    a, b = run(), run()
    assert [s.net for s in a] == [s.net for s in b]


def test_simulation_rejects_rounds_that_do_not_fill_batches():
    with pytest.raises(ValueError):
        simulate_table([casual("flat_noise")], 2_500, seed=0)


def test_wonging_player_skips_rounds_at_shared_table():
    wong = CounterPlayer(CounterConfig(wonging=True), rng=rng(1))
    stats = simulate_table([wong, casual("flat_noise")], 10_000, seed=4)
    assert 0 < stats[0].rounds_played < 10_000
    assert stats[1].rounds_played == 10_000


@pytest.mark.slow
def test_counter_wins_and_casual_loses():
    """Prueba estadística: todo el IC95 debe quedar del lado esperado."""
    rounds = 1_500_000
    counter = simulate_table([CounterPlayer(rng=rng(11))], rounds, seed=101)[0]
    assert counter.ci95_per_100[0] > 0

    for i, style in enumerate(["hot_streak", "martingale", "flat_noise"]):
        player = CasualPlayer(CasualConfig(bet_style=style), rng=rng(20 + i))
        stats = simulate_table([player], 200_000, seed=200 + i)[0]
        assert stats.ci95_per_100[1] < 0


def test_basic_rules_default_is_used():
    assert CounterPlayer(rng=rng()).rules == Rules()
