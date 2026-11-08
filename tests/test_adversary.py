import numpy as np
import pytest

pytest.importorskip("cma")

from env import CasinoTableEnv, SeatEvent, Table, TableConfig  # noqa: E402
from env.casino_env import ACTION_EXPEL, ACTION_NONE  # noqa: E402
from env.curriculum import DEFAULT_STAGES, Curriculum, Stage  # noqa: E402
from players import StealthConfig  # noqa: E402
from retraining.adversary import (  # noqa: E402
    AdversaryConfig,
    CorrelationSupervisor,
    Evaluation,
    Evolver,
    League,
    decode,
    encode,
    measure_expel_rate,
)

TINY = AdversaryConfig(population=4, generations=2, ev_rounds=2_000, episodes=1, rounds_per_episode=100)


def play(seat, bet, tc):
    return SeatEvent(0, seat, 1, "play", bet, tc, true_type="casual")


# ---------------------------------------------------------------- curriculum

def test_curriculum_stages_match_the_plan():
    assert [(s.disguise_level, s.stealth_share) for s in DEFAULT_STAGES] == [
        (0.0, 0.05), (0.3, 0.10), (0.6, 0.15), (0.9, 0.15),
    ]


def test_curriculum_table_config_applies_stage():
    cfg = Curriculum().table_config()
    assert cfg.stealth == StealthConfig.from_disguise(0.0)
    assert cfg.population_mix["stealth"] == 0.05
    assert sum(cfg.population_mix.values()) == pytest.approx(1)


def test_curriculum_advances_on_detection():
    c = Curriculum()
    assert not c.update(0.89, 0.01, 1000)
    assert c.update(0.90, 0.01, 1000)
    assert c.stage.disguise_level == 0.3


def test_curriculum_blocked_by_false_positives():
    c = Curriculum()
    assert not c.update(0.99, 0.06, 1000)
    assert c.index == 0


def test_curriculum_last_stage_finishes_by_timesteps():
    c = Curriculum(stages=(Stage(0.0, 0.1, until_detection_rate=0.5), Stage(0.9, 0.1, until_timesteps=100)))
    assert c.update(0.5, 0.0, 10)
    assert not c.finished
    assert not c.update(0.0, 0.0, 60)
    assert not c.update(0.0, 0.0, 60)
    assert c.finished and c.index == 1


def test_curriculum_applies_to_env_on_reset():
    c = Curriculum(stages=(Stage(0.6, 0.5, until_timesteps=1),))
    env = CasinoTableEnv()
    env.table_config = c.table_config()
    env.reset(seed=0)
    assert env.table.config.stealth == StealthConfig.from_disguise(0.6)


# ---------------------------------------------------------------- errores deliberados

def test_stealth_deliberate_error_only_near_index():
    from blackjack import TEN, Action, Hand
    from players import StealthCounterPlayer

    allowed = frozenset({Action.HIT, Action.STAND})
    cfg = StealthConfig(deliberate_error_prob=1.0, index_play_prob=1.0)
    p = StealthCounterPlayer(cfg, rng=np.random.default_rng(0))
    p.on_shuffle(3.0)
    hand = Hand(cards=[TEN, 6], bet=1)   # índice de 16 vs 10: TC 0

    p.running_count = 3                  # TC 1: cerca, se equivoca y pide
    assert p.decide(hand, TEN, allowed) == Action.HIT
    p.running_count = 12                 # TC 4: lejos, juega bien
    assert p.decide(hand, TEN, allowed) == Action.STAND
    # Sin jugada índice nunca hay error
    assert p.decide(Hand(cards=[TEN, 8], bet=1), TEN, allowed) == Action.STAND


def test_stealth_cover_bets_only_with_neutral_count():
    from players import StealthCounterPlayer, ramp_bet

    cfg = StealthConfig(cover_bet_prob=1.0, bet_noise_std=0.0, delay_bet_ramp_rounds=0, bet_spread_max=12.0)
    p = StealthCounterPlayer(cfg, rng=np.random.default_rng(0))
    p.on_shuffle(3.0)
    p._won_last, p._last_bet = True, 2.0

    p.running_count = 3                  # TC 1: neutro, cobertura "racha caliente"
    assert p.place_bet() == 4.0
    p.running_count = 15                 # TC 5: sigue la rampa
    assert p.place_bet() == ramp_bet(5.0)
    p.running_count = -9                 # TC -3: apuesta mínima, sin cobertura
    assert p.place_bet() == 1.0


# ---------------------------------------------------------------- genoma

def test_encode_decode_roundtrip():
    original = StealthConfig()
    decoded = decode(encode(original))
    for a, b in zip(vars(decoded).values(), vars(original).values()):
        assert a == pytest.approx(b)


def test_decode_clips_and_rounds_delay():
    cfg = decode(np.array([-1, 2, 0.5, 0.5, 0.5, 0.4]))
    assert cfg.bet_spread_max == 2 and cfg.bet_noise_std == 0.6
    assert isinstance(cfg.delay_bet_ramp_rounds, int) and cfg.delay_bet_ramp_rounds == 1


# ---------------------------------------------------------------- supervisor heurístico

def test_correlation_supervisor_expels_correlated_bets_only():
    s = CorrelationSupervisor(min_rounds=30)
    rng = np.random.default_rng(0)
    actions = None
    for _ in range(40):
        tc = rng.normal(0, 2)
        actions = s.act(None, [play(0, max(1, tc * 2), tc), play(1, rng.choice([1, 2, 4]), tc)])
    assert actions[0] == ACTION_EXPEL and actions[1] == ACTION_NONE


def test_correlation_supervisor_waits_for_min_rounds():
    s = CorrelationSupervisor(min_rounds=30)
    for tc in range(29):
        actions = s.act(None, [play(0, tc + 1, tc)])
    assert actions[0] == ACTION_NONE


def test_correlation_supervisor_forgets_on_leave():
    s = CorrelationSupervisor(min_rounds=3)
    for tc in range(5):
        s.act(None, [play(0, tc + 1, tc)])
    s.act(None, [SeatEvent(0, 0, 1, "leave", 0, 0)])
    assert s.act(None, [])[0] == ACTION_NONE


# ---------------------------------------------------------------- adversario

def test_expel_rate_is_a_valid_fraction():
    rate = measure_expel_rate(StealthConfig(), CorrelationSupervisor(), TINY, seed=1)
    assert 0 <= rate <= 1


def test_expel_rate_extremes():
    class ExpelAll:
        def reset(self): ...
        def act(self, obs, events): return [ACTION_EXPEL] * 5

    class Never:
        def reset(self): ...
        def act(self, obs, events): return [ACTION_NONE] * 5

    assert measure_expel_rate(StealthConfig(), Never(), TINY, seed=1) == 0
    # Quien llegó en la última ronda puede quedar sin expulsar; casi todos salen.
    assert measure_expel_rate(StealthConfig(), ExpelAll(), TINY, seed=1) > 0.9


def test_league_keeps_most_recent_and_builds_pool():
    league = League(size=2)
    configs = [StealthConfig(bet_spread_max=x) for x in (3.0, 4.0, 5.0)]
    for c in configs:
        league.add(c)
    assert league.members == configs[1:]
    table = Table(league.table_config(), np.random.default_rng(0))
    spreads = {table._make_player("stealth").stealth.bet_spread_max for _ in range(50)}
    assert spreads == {4.0, 5.0}


def test_empty_stealth_pool_uses_single_config():
    cfg = TableConfig(stealth=StealthConfig(bet_spread_max=7.0))
    assert Table(cfg, np.random.default_rng(0))._make_player("stealth").stealth.bet_spread_max == 7.0


def test_evolver_cycle_updates_best_and_league_and_is_reproducible():
    def run():
        e = Evolver(CorrelationSupervisor, TINY, seed=3)
        return e, e.run_cycle()

    evolver, result = run()
    assert isinstance(result, Evaluation)
    assert evolver.best == result.stealth
    assert evolver.league.members == [result.stealth]
    assert result.fitness == pytest.approx(result.ev_per_100 - TINY.expelled_penalty * result.expel_rate)
    assert run()[1] == result
