from dataclasses import replace

import numpy as np
import pytest

from env import SeatEvent, SeatFeatureExtractor, Table, TableConfig
from env.features import F, G, _pearson, _ranks


def sit(seat=0, tc=0.0, round_id=0):
    return SeatEvent(round_id, seat, 1, "sit", 0.0, tc)


def play(bet, tc, seat=0, result=0.0, deviated=False, matched=False, actions=()):
    return SeatEvent(0, seat, 1, "play", bet, tc, list(actions), deviated, matched, result, 0.3)


def features_after(events, **kw):
    ex = SeatFeatureExtractor(**kw)
    obs = None
    for e in events:
        obs = ex.transform([e])
    return obs


# ---------------------------------------------------------------- unitarios

def test_empty_table_is_all_zeros():
    obs = SeatFeatureExtractor().transform([])
    assert obs["seats"].shape == (5, len(F)) and not obs["seats"].any()
    assert obs["global"].shape == (len(G),)


def test_bet_tracks_count_exactly():
    tcs = [-2, -1, 0, 1, 2, 3, 4]
    obs = features_after([sit(tc=1.5)] + [play(2 * tc + 10, tc) for tc in tcs])
    s = obs["seats"][0]
    assert s[F["occupied"]] == 1 and s[F["rounds_seen"]] == 7
    assert s[F["entry_tc"]] == 1.5
    assert s[F["corr_bet_tc"]] == pytest.approx(1.0)
    assert s[F["rank_corr_bet_tc"]] == pytest.approx(1.0)
    assert s[F["bet_tc_slope"]] == pytest.approx(2.0)
    assert s[F["bet_spread"]] == pytest.approx(18 / 6)
    assert s[F["bet_norm"]] == pytest.approx(18 / 6)


def test_flat_bettor_has_no_count_signal():
    obs = features_after([sit()] + [play(1.0, tc) for tc in range(-3, 4)])
    s = obs["seats"][0]
    assert s[F["corr_bet_tc"]] == 0 and s[F["bet_tc_slope"]] == 0
    assert s[F["bet_spread"]] == 1


def test_correlations_need_minimum_samples():
    obs = features_after([sit(), play(1, -1), play(2, 1), play(3, 2)])
    assert obs["seats"][0][F["corr_bet_tc"]] == 0


def test_ranks_average_ties():
    assert _ranks(np.array([10.0, 20.0, 10.0, 30.0])).tolist() == [0.5, 2.0, 0.5, 3.0]


def test_pearson_matches_numpy():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=40), rng.normal(size=40)
    assert _pearson(x, y) == pytest.approx(np.corrcoef(x, y)[0, 1])


def test_bet_after_win_correlation():
    # Sube tras ganar, vuelve a 1 tras perder: correlación con el resultado previo = 1.
    results = [1, 1, -1, 1, -1, -1, 1, 1, -1, 1]
    events, bet = [sit()], 1.0
    for r in results:
        events.append(play(bet, 0, result=r))
        bet = bet + 1 if r > 0 else 1.0
    s = features_after(events)["seats"][0]
    assert s[F["corr_bet_prev_result"]] > 0.8
    assert s[F["net_win_units"]] == sum(results)


def test_deviation_rates():
    events = [sit()] + [
        play(1, 3, deviated=True, matched=True),
        play(1, 3, deviated=True, matched=True),
        play(1, 0, deviated=True, matched=False),
        play(1, 0),
    ]
    s = features_after(events)["seats"][0]
    assert s[F["index_play_rate"]] == pytest.approx(2 / 3)
    assert s[F["basic_strategy_error_rate"]] == pytest.approx(1 / 4)


def test_insurance_features():
    events = [sit(), play(1, 4, actions=["insurance", "stand"]),
              play(1, 2, actions=["insurance", "hit"]), play(1, 0), play(1, 0)]
    s = features_after(events)["seats"][0]
    assert s[F["insurance_rate"]] == pytest.approx(0.5)
    assert s[F["insurance_tc_mean"]] == pytest.approx(3.0)


def test_sit_out_rate_detects_wonging():
    events = [sit()] + [play(0.0, -2)] * 3 + [play(4.0, 3)]
    s = features_after(events)["seats"][0]
    assert s[F["sit_out_rate"]] == pytest.approx(0.75)
    assert s[F["bet_norm"]] == 1


def test_window_keeps_last_rounds_but_totals_cover_the_stay():
    events = [sit()] + [play(5.0, 0, result=1.0)] * 10 + [play(1.0, 0, result=1.0)] * 4
    s = features_after(events, window=4)["seats"][0]
    assert s[F["bet_spread"]] == 1           # la ventana sólo ve las apuestas de 1
    assert s[F["rounds_seen"]] == 14 and s[F["net_win_units"]] == 14


def test_leave_and_expel_clear_the_seat():
    ex = SeatFeatureExtractor()
    ex.transform([sit(0), sit(1), play(3, 2, seat=0), play(1, 2, seat=1)])
    obs = ex.transform([
        SeatEvent(1, 0, 1, "leave", 0, 0),
        SeatEvent(1, 1, 2, "expelled", 0, 0),
    ])
    assert not obs["seats"][:2].any()
    assert obs["global"][G["n_players"]] == 0
    assert obs["global"][G["n_expelled"]] == 1

    obs = ex.transform([sit(0, tc=-1)])
    assert obs["seats"][0][F["rounds_seen"]] == 0 and obs["seats"][0][F["entry_tc"]] == -1
    ex.reset()
    assert ex.transform([])["global"][G["n_expelled"]] == 0


def test_global_features_follow_last_round():
    obs = features_after([sit(), play(1, 2.5)])
    assert obs["global"][G["true_count"]] == 2.5
    assert obs["global"][G["shoe_penetration"]] == pytest.approx(0.3)
    assert obs["global"][G["n_players"]] == 1


# ---------------------------------------------------------------- sobre la mesa real

def mean_feature(kind, feature, rounds=300, seeds=range(4), **kw):
    values = []
    for seed in seeds:
        cfg = TableConfig(population_mix={kind: 1.0}, initial_occupancy=1.0,
                          session_mean_rounds={kind: 1e9}, bankroll_units=(1e6, 1e6), **kw)
        table = Table(cfg, np.random.default_rng(seed))
        ex = SeatFeatureExtractor()
        ex.transform(table.fill_initial_seats())
        for _ in range(rounds):
            obs = ex.transform(table.play_round())
        values.extend(obs["seats"][:, F[feature]])
    return float(np.mean(values))


def test_bet_tc_correlation_separates_counters_from_casuals():
    counter = mean_feature("counter", "corr_bet_tc")
    stealth = mean_feature("stealth", "corr_bet_tc")
    casual = mean_feature("casual", "corr_bet_tc")
    assert counter > 0.5
    assert casual < 0.15
    assert casual < stealth < counter


def test_features_never_read_true_type():
    from env import CasinoTableEnv

    env = CasinoTableEnv(TableConfig())
    events = env.reset(seed=3)[1]["events"]
    a, b = SeatFeatureExtractor(), SeatFeatureExtractor()
    for _ in range(200):
        obs_a = a.transform(events)
        obs_b = b.transform([replace(e, true_type="") for e in events])
        assert obs_a["seats"].tobytes() == obs_b["seats"].tobytes()
        events = env.step([0] * 5)[-1]["events"]
