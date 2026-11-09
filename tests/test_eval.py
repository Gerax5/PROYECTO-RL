import numpy as np
import pytest

from env import SeatEvent, TableConfig
from env.casino_env import ACTION_EXPEL, ACTION_NONE
from env.features import F
from eval import SupervisorMetrics, evaluate, run_episode
from supervisor import HeuristicSupervisor, NeverExpel


def sit(seat, kind):
    return SeatEvent(0, seat, seat, "sit", 0.0, 0.0, true_type=kind)


def play(seat, kind, bet=1.0, result=0.0):
    return SeatEvent(1, seat, seat, "play", bet, 0.0, result_units=result, true_type=kind)


def only(kind):
    return TableConfig(population_mix={kind: 1.0}, initial_occupancy=1.0,
                       session_mean_rounds={kind: 1e9}, bankroll_units=(1e6, 1e6))


# ---------------------------------------------------------------- métricas

def test_metrics_count_expulsions_by_type():
    m = SupervisorMetrics()
    m.update({"events": [sit(0, "casual"), sit(1, "casual"), sit(2, "counter"), sit(3, "stealth")]})
    m.update({
        "events": [play(2, "counter", result=2.0), play(0, "casual", result=-1.0)],
        "expulsions": [{"seat": 2, "true_type": "counter", "rounds_seen": 40},
                       {"seat": 0, "true_type": "casual", "rounds_seen": 10}],
        "house_net": -1.0,
    }, reward=-15.0)
    s = m.summary()
    assert s["recall_counter"] == 1 and s["recall_stealth"] == 0
    assert s["fp_rate"] == 0.5 and s["precision"] == 0.5
    assert s["f1_counters"] == pytest.approx(0.5)
    assert s["detection_rounds"] == 40
    assert m.rounds == 1 and m.counter_rounds == 1 and m.counter_net == 2.0
    assert s["reward_per_100"] == -1500


def test_metrics_without_expulsions_are_zero_not_nan():
    s = SupervisorMetrics().summary()
    assert all(v == 0 for v in s.values())


def test_metrics_merge_adds_up():
    a, b = SupervisorMetrics(), SupervisorMetrics()
    a.update({"events": [sit(0, "counter")]})
    b.update({"events": [sit(0, "counter")],
              "expulsions": [{"seat": 0, "true_type": "counter", "rounds_seen": 5}],
              "house_net": 0.0})
    m = a.merge(b)
    assert m.seen["counter"] == 2 and m.recall("counter") == 0.5 and m.rounds == 1


# ---------------------------------------------------------------- baselines

def test_heuristic_expels_only_confident_high_correlation_seats():
    seats = np.zeros((5, len(F)), dtype=np.float32)
    seats[:, F["occupied"]] = 1
    seats[:, F["rounds_seen"]] = [50, 50, 10, 50, 50]
    seats[:, F["corr_bet_tc"]] = [0.9, 0.2, 0.9, 0.61, 0.6]
    seats[4, F["occupied"]] = 0
    action = HeuristicSupervisor(0.6, 30).act({"seats": seats, "global": np.zeros(4)})
    assert action.tolist() == [ACTION_EXPEL, ACTION_NONE, ACTION_NONE, ACTION_EXPEL, ACTION_NONE]


def test_never_expel_never_expels():
    m = evaluate(NeverExpel(), [0], only("counter"), rounds_per_episode=50)
    assert sum(m.expelled.values()) == 0 and m.rounds == 50


def test_heuristic_catches_open_counters_and_spares_casuals():
    from env import CasinoTableEnv

    counters = run_episode(CasinoTableEnv(only("counter"), rounds_per_episode=200),
                           HeuristicSupervisor(), seed=0)
    casuals = run_episode(CasinoTableEnv(only("casual"), rounds_per_episode=200),
                          HeuristicSupervisor(), seed=0)
    # Los reemplazos de los expulsados llegan tarde; basta con atrapar a los 5 iniciales.
    assert counters.expelled["counter"] >= 5
    assert casuals.false_positive_rate < 0.2


def test_evaluate_is_reproducible():
    a = evaluate(HeuristicSupervisor(), [3, 4], rounds_per_episode=150).summary()
    b = evaluate(HeuristicSupervisor(), [3, 4], rounds_per_episode=150).summary()
    assert a == b
