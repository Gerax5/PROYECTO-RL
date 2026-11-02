from dataclasses import replace

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
from gymnasium.utils.env_checker import check_env  # noqa: E402

from blackjack import ACE, TEN  # noqa: E402
from env import CasinoTableEnv, MinimalFeatureExtractor, RewardConfig, Table, TableConfig  # noqa: E402
from env.casino_env import ACTION_EXPEL, ACTION_NONE, ACTION_WATCH, expected_counter_gain  # noqa: E402

NOOP = [ACTION_NONE] * 5


def only(kind, **kw):
    """Mesa llena de un solo tipo, sin salidas por fin de sesión."""
    return TableConfig(
        population_mix={kind: 1.0},
        initial_occupancy=1.0,
        session_mean_rounds={kind: 1e9},
        bankroll_units=(1e6, 1e6),
        **kw,
    )


def make_env(cfg, seed=0, **kw):
    env = CasinoTableEnv(cfg, **kw)
    env.reset(seed=seed)
    return env


def expel_seat(seat):
    action = list(NOOP)
    action[seat] = ACTION_EXPEL
    return action


# ---------------------------------------------------------------- mesa

def test_table_events_follow_the_contract():
    table = Table(only("casual"), np.random.default_rng(0))
    assert [e.event for e in table.fill_initial_seats()] == ["sit"] * 5
    events = table.play_round()
    assert [e.event for e in events] == ["play"] * 5
    assert all(e.bet >= 1 and e.true_type == "casual" for e in events)
    assert sorted(e.seat for e in events) == list(range(5))


def test_table_arrivals_fill_empty_seats():
    cfg = replace(only("casual"), initial_occupancy=0.0, arrival_rate=0.5)
    table = Table(cfg, np.random.default_rng(1))
    assert table.fill_initial_seats() == []
    events = [e for _ in range(20) for e in table.play_round()]
    assert sum(e.event == "sit" for e in events) == 5
    assert all(s is not None for s in table.seats)


def test_table_session_end_emits_leave_and_frees_seat():
    cfg = replace(only("casual"), session_mean_rounds={"casual": 1.0}, arrival_rate=0.0)
    table = Table(cfg, np.random.default_rng(2))
    table.fill_initial_seats()
    events = table.play_round()
    assert sum(e.event == "leave" for e in events) == 5
    assert table.seats == [None] * 5


def test_table_broke_player_leaves_and_never_goes_negative():
    cfg = replace(only("casual"), bankroll_units=(3.0, 3.0), arrival_rate=0.0)
    table = Table(cfg, np.random.default_rng(3))
    table.fill_initial_seats()
    leaves = 0
    for _ in range(3000):
        for sp in table.seats:
            assert sp is None or sp.bankroll >= 1.0
        leaves += sum(e.event == "leave" for e in table.play_round())
    assert leaves == 5


def test_table_public_count_matches_counter_count():
    table = Table(only("counter"), np.random.default_rng(4))
    table.fill_initial_seats()
    for _ in range(500):
        table.play_round()
        assert all(sp.player.running_count == table.running_count for sp in table.seats)


def test_table_reshuffles_at_cut_card():
    table = Table(only("casual"), np.random.default_rng(5))
    table.fill_initial_seats()
    for _ in range(100):
        table.play_round()
    assert table.shoe.shuffle_count > 1


def test_table_flags_index_play_and_plain_error():
    table = Table(only("casual"), np.random.default_rng(6))
    table.fill_initial_seats()
    from blackjack import Action, Hand

    allowed = frozenset({Action.HIT, Action.STAND})
    sp = table.seats[0]
    sp.player.error_rate = 0.0

    table.running_count = 100     # 16 vs 10 con TC alto: plantarse sería la jugada índice
    sp.player.decide = lambda *a: Action.STAND
    table._decide(0, Hand(cards=[TEN, 6], bet=1), TEN, allowed)
    assert sp.deviated and sp.matched_index

    sp.deviated = sp.matched_index = False
    table.running_count = -100    # con TC negativo plantarse ya no coincide con el índice
    table._decide(0, Hand(cards=[TEN, 6], bet=1), TEN, allowed)
    assert sp.deviated and not sp.matched_index


def test_table_insurance_flag_depends_on_count():
    table = Table(only("casual"), np.random.default_rng(7))
    table.fill_initial_seats()
    from blackjack import Hand

    sp = table.seats[0]
    sp.player.insure = lambda *a: True
    table.running_count = 100
    assert table._insure(0, Hand(cards=[TEN, 9], bet=2), ACE)
    assert sp.matched_index and sp.staked == 1


def test_low_bankroll_cannot_double_or_split():
    from blackjack import Action, Hand

    table = Table(only("casual"), np.random.default_rng(8))
    table.fill_initial_seats()
    sp = table.seats[0]
    sp.bankroll, sp.staked = 10.0, 10.0
    seen = {}

    def spy(hand, up, allowed):
        seen["allowed"] = allowed
        return Action.HIT

    sp.player.decide = spy
    table._decide(0, Hand(cards=[5, 5], bet=10), 6,
                  frozenset({Action.HIT, Action.STAND, Action.DOUBLE, Action.SPLIT}))
    assert seen["allowed"] == {Action.HIT, Action.STAND}


# ---------------------------------------------------------------- recompensa

@pytest.mark.parametrize("kind, base", [("casual", -25.0), ("counter", 10.0), ("stealth", 15.0)])
def test_expel_reward_by_type(kind, base):
    env = make_env(only(kind), reward_config=RewardConfig(early_detection_k=0))
    _, _, _, _, info = env.step(expel_seat(0))
    assert info["reward_terms"]["expel"] == base
    assert info["expulsions"] == [{"seat": 0, "true_type": kind, "rounds_seen": 0}]
    assert env.table.seats[0] is None or env.table.seats[0].rounds_seen == 1


def test_early_detection_bonus_only_for_counters_and_decays():
    env = make_env(only("counter"), reward_config=RewardConfig(early_detection_k=20))
    for _ in range(9):
        env.step(NOOP)
    _, _, _, _, info = env.step(expel_seat(0))
    assert info["reward_terms"]["detection"] == pytest.approx(20 / 9)

    env = make_env(only("casual"), reward_config=RewardConfig(early_detection_k=20))
    _, _, _, _, info = env.step(expel_seat(0))
    assert info["reward_terms"]["detection"] == 0


def test_expelling_or_watching_empty_seat_is_a_free_noop():
    cfg = replace(only("casual"), initial_occupancy=0.0, arrival_rate=0.0)
    env = make_env(cfg)
    _, reward, _, _, info = env.step([ACTION_EXPEL, ACTION_WATCH, 0, 0, 0])
    assert reward == 0 and info["expulsions"] == [] and info["watched"] == []


def test_watch_cost():
    env = make_env(only("casual"))
    _, reward, _, _, info = env.step([ACTION_WATCH, ACTION_WATCH, 0, 0, 0])
    assert reward == pytest.approx(-0.1)
    assert info["watched"] == [0, 1]


def test_edge_term_charges_counters_but_not_casuals():
    env = make_env(only("casual"))
    assert env.step(NOOP)[1] == 0

    env = make_env(only("counter"))
    _, reward, _, _, info = env.step(NOOP)
    expected = -sum(expected_counter_gain(e.bet, e.true_count)
                    for e in info["events"] if e.event == "play")
    assert expected != 0 and reward == pytest.approx(expected)


def test_expected_counter_gain():
    assert expected_counter_gain(10, 0) == pytest.approx(-0.04)
    assert expected_counter_gain(10, 3) == pytest.approx(0.11)


def test_expelled_player_does_not_play_that_round():
    env = make_env(only("casual"))
    _, _, _, _, info = env.step(expel_seat(2))
    kinds = [(e.seat, e.event) for e in info["events"]]
    assert (2, "expelled") in kinds and (2, "play") not in kinds


# ---------------------------------------------------------------- ambiente

def test_gymnasium_check_env():
    check_env(CasinoTableEnv(rounds_per_episode=50), skip_render_check=True)


def test_episode_truncates_at_configured_rounds():
    env = make_env(TableConfig(), rounds_per_episode=10)
    for i in range(10):
        _, _, terminated, truncated, _ = env.step(NOOP)
        assert not terminated and truncated == (i == 9)


def test_episode_is_reproducible_with_seed():
    def run():
        env = make_env(TableConfig(), seed=123, rounds_per_episode=200)
        rng = np.random.default_rng(1)
        out = []
        for _ in range(200):
            action = rng.choice([0, 0, 0, 0, 1, 2], size=5)
            obs, r, *_ = env.step(action)
            out.append((r, obs["seats"].tobytes()))
        return out

    assert run() == run()


def test_observation_does_not_depend_on_true_type():
    """El tipo real sólo viaja en `true_type`; borrarlo no debe cambiar la observación."""
    env = make_env(TableConfig(), seed=5)
    a, b = MinimalFeatureExtractor(), MinimalFeatureExtractor()
    for _ in range(100):
        events = env.step(NOOP)[-1]["events"]
        obs_a = a.transform(events)
        obs_b = b.transform([replace(e, true_type="") for e in events])
        assert obs_a["seats"].tobytes() == obs_b["seats"].tobytes()
        assert obs_a["global"].tobytes() == obs_b["global"].tobytes()


def test_long_random_run_with_all_player_types():
    env = make_env(TableConfig(), seed=9, rounds_per_episode=5000)
    rng = np.random.default_rng(0)
    seen = set()
    truncated = False
    while not truncated:
        action = rng.choice([0, 1, 2], size=5, p=[0.97, 0.02, 0.01])
        obs, reward, _, truncated, info = env.step(action)
        assert env.observation_space.contains(obs) and np.isfinite(reward)
        seen |= {e.true_type for e in info["events"]}
    assert seen == {"casual", "counter", "stealth"}
