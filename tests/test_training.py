"""Pruebas de humo del entrenamiento: corridas diminutas, sólo verifican que todo encaje."""

import json
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

from env import CasinoTableEnv, TableConfig  # noqa: E402
from eval import evaluate  # noqa: E402
from players import StealthConfig  # noqa: E402
from retraining.adversary import AdversaryConfig, measure_expel_rate  # noqa: E402
from retraining.buffer import ScenarioBuffer  # noqa: E402
from retraining.online_loop import OnlineConfig, run_online  # noqa: E402
from supervisor.train import (  # noqa: E402
    TrainConfig,
    build_table_config,
    load_supervisor,
    train,
)

TINY = dict(
    total_timesteps=256, n_envs=2, vec_env="dummy", rounds_per_episode=40, n_steps=64,
    batch_size=64, n_epochs=1, eval_freq=128, eval_episodes=1, seat_encoder=(16,),
    global_encoder=(8,), lstm_hidden=16, head=(16,),
)


@pytest.fixture(scope="module", params=["recurrent_ppo", "ppo"])
def run(request, tmp_path_factory):
    cfg = TrainConfig(algo=request.param, **TINY)
    return train(cfg, tmp_path_factory.mktemp(request.param))


def test_train_writes_checkpoints_and_history(run):
    for name in ("best", "final", "last"):
        if name == "best" and not (run / name).exists():
            continue      # sin evaluación con FP bajo no hay "best"; es válido
        assert (run / name / "model.zip").exists()
        assert (run / name / "vecnormalize.pkl").exists()
    history = json.loads((run / "history.json").read_text())
    assert len(history) >= 2 and "recall_stealth" in history[-1]
    assert TrainConfig.load(run / "config.json").n_envs == 2


def test_trained_supervisor_plays_full_episodes(run):
    policy = load_supervisor(run, "final")
    env = CasinoTableEnv(rounds_per_episode=30)
    obs, _ = env.reset(seed=0)
    policy.reset()
    action = policy.act(obs)
    assert action.shape == (5,) and env.action_space.contains(action)
    m = evaluate(policy, [1, 2], rounds_per_episode=30)
    assert m.rounds == 60


def test_trained_supervisor_works_with_the_adversary(run):
    cfg = AdversaryConfig(episodes=1, rounds_per_episode=30)
    rate = measure_expel_rate(StealthConfig(), load_supervisor(run, "final"), cfg, seed=0)
    assert 0.0 <= rate <= 1.0


def test_initial_policy_rarely_expels(tmp_path):
    import torch

    from supervisor.train import make_model, make_vec_env

    cfg = TrainConfig(algo="ppo", initial_action_logit=-5.0, **TINY)
    venv = make_vec_env([TableConfig()], cfg.reward, 40, 0, "dummy")
    model = make_model(cfg, venv)
    obs = venv.reset()
    obs_t, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        probs = torch.stack([d.probs for d in model.policy.get_distribution(obs_t).distribution])
    assert probs[..., 2].max() < 0.02 and probs[..., 0].min() > 0.95
    venv.close()


def test_curriculum_run(tmp_path):
    cfg = TrainConfig(algo="ppo", curriculum=True, **TINY)
    run = train(cfg, tmp_path / "curr")
    history = json.loads((run / "history.json").read_text())
    assert all(h["stage"] is not None for h in history)


def test_build_table_config_equalizes_sessions():
    table = build_table_config(TrainConfig(disguise=0.6, session_mean=90))
    assert set(table.session_mean_rounds.values()) == {90}
    assert table.stealth == StealthConfig.from_disguise(0.6)
    unequal = build_table_config(TrainConfig(equal_sessions=False))
    assert unequal.session_mean_rounds == TableConfig().session_mean_rounds


def test_config_roundtrip(tmp_path):
    cfg = TrainConfig(algo="ppo", seat_encoder=(32, 16), reward=replace(TrainConfig().reward, expel_casual=-7))
    cfg.save(tmp_path / "c.json")
    assert TrainConfig.load(tmp_path / "c.json") == cfg


# ---------------------------------------------------------------- reentrenamiento

def test_scenario_buffer_mixes_new_and_old():
    base = TableConfig()
    buf = ScenarioBuffer(capacity=2, replay_ratio_old=0.25)
    assert [t.stealth_pool for t in buf.table_configs(4, base)] == [(base.stealth,)] * 4

    a, b, c = (StealthConfig(bet_spread_max=s) for s in (3.0, 5.0, 7.0))
    buf.add(a)
    buf.end_cycle()
    buf.add(b)
    tables = buf.table_configs(4, base)
    assert [t.stealth_pool for t in tables] == [(b,)] * 3 + [(a,)]

    buf.end_cycle()
    buf.add(c)
    buf.end_cycle()
    assert list(buf.old) == [b, c]          # capacidad 2: se olvida el más viejo


def test_online_loop_without_adversary(run, tmp_path):
    cfg = OnlineConfig(cycles=1, finetune_timesteps=128, n_envs=2, vec_env="dummy",
                       use_adversary=False, val_disguises=(0.3,), val_episodes=1)
    out = run_online(run, tmp_path / "online", cfg)
    history = json.loads((out / "history.json").read_text())
    assert [h["cycle"] for h in history] == [0, 1]
    assert (out / "best" / "model.zip").exists()
    load_supervisor(out).act(CasinoTableEnv().reset(seed=0)[0])


# ---------------------------------------------------------------- clasificador

def test_classifier_separates_counters():
    from supervisor.classifier import ClassifierSupervisor, collect_samples, fit_classifier

    x, y = collect_samples([0, 1, 2], rounds=200)
    assert len(x) == len(y) and 0 < y.mean() < 1
    for kind in ("logistic", "gboost"):
        model = fit_classifier(x, y, kind)
        proba = model.predict_proba(x)[:, 1]
        assert proba[y == 1].mean() > proba[y == 0].mean() + 0.3

    sup = ClassifierSupervisor(model, threshold=0.5, min_rounds=0)
    seats = x[:5].copy()
    action = sup.act({"seats": seats, "global": np.zeros(4)})
    assert action.shape == (5,)
