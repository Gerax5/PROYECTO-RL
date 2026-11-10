"""Explicabilidad: completitud de los gradientes integrados, oclusión y replay con XAI."""

import numpy as np
import pytest

from env import CasinoTableEnv
from env.features import F, SEAT_FEATURES
from xai import AttributionStats, make_explainer, top_features
from xai.tabular import occlusion_attributions


def test_occlusion_is_exact_for_additive_models():
    rng = np.random.default_rng(0)
    w = rng.normal(size=4)
    predict = lambda z: z @ w  # noqa: E731
    x, background = rng.normal(size=(3, 4)), rng.normal(size=(20, 4))
    p, attr = occlusion_attributions(predict, x, background)
    np.testing.assert_allclose(p, x @ w)
    np.testing.assert_allclose(attr, w * (x - background.mean(0)))


def test_attribution_stats_and_top_features():
    stats = AttributionStats()
    attr = {name: 0.0 for name in SEAT_FEATURES} | {"corr_bet_tc": 0.5, "bet_spread": -0.8}
    stats.add("g", attr)
    stats.add("g", attr)
    table = stats.table("g")
    assert stats.n["g"] == 2 and table[0]["feature"] == "bet_spread" and table[0]["mean"] == pytest.approx(-0.8)
    assert [n for n, _ in top_features(attr, 2)] == ["bet_spread", "corr_bet_tc"]


def test_heuristic_has_no_explainer():
    from supervisor import HeuristicSupervisor

    assert make_explainer(HeuristicSupervisor()) is None


# ------------------------------------------------------------ clasificador

@pytest.fixture(scope="module")
def classifier():
    pytest.importorskip("sklearn")
    from supervisor.classifier import ClassifierSupervisor, collect_samples, fit_classifier

    x, y = collect_samples([1, 2], rounds=200)
    return ClassifierSupervisor(fit_classifier(x, y, "logistic"), threshold=0.9), x


def test_classifier_explainer(classifier):
    sup, x = classifier
    explainer = make_explainer(sup, background=x[:30])
    env = CasinoTableEnv(rounds_per_episode=60)
    obs, _ = env.reset(seed=5)
    for _ in range(40):
        obs, *_ = env.step(np.zeros(5, dtype=int))
    out = explainer.explain(obs)
    occupied = obs["seats"][:, F["occupied"]] > 0
    assert [o is not None for o in out] == list(occupied)
    s = int(np.flatnonzero(occupied)[0])
    p = sup.model.predict_proba(obs["seats"][s:s + 1])[0, 1]
    assert out[s]["score"] == pytest.approx(p, abs=1e-4)
    assert set(out[s]["attr"]) == set(SEAT_FEATURES)


def test_replay_carries_explanations(classifier):
    from eval.replay import record_episode

    sup, x = classifier
    replay = record_episode(sup, seed=3, rounds=30, meta={"threshold": 0.9},
                            explainer=make_explainer(sup, background=x[:20]))
    assert replay["meta"]["xai"]["score_label"] == "P(contador)"
    seats = [s for f in replay["frames"] for s in f["seats"] if "xai" in s]
    assert seats and all(0 <= s["xai"]["score"] <= 1 for s in seats)


# ------------------------------------------------------------ política RL

@pytest.fixture(scope="module", params=["recurrent_ppo", "ppo"])
def trained(request, tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("sb3_contrib")
    from supervisor.train import TrainConfig, train

    cfg = TrainConfig(
        algo=request.param, total_timesteps=256, n_envs=2, vec_env="dummy", rounds_per_episode=40,
        n_steps=64, batch_size=64, n_epochs=1, eval_freq=10_000, eval_episodes=1,
        seat_encoder=(16,), global_encoder=(8,), lstm_hidden=16, head=(16,),
    )
    return train(cfg, tmp_path_factory.mktemp(request.param))


def _play(policy, explainer, steps=25, seed=4):
    env = CasinoTableEnv(rounds_per_episode=steps + 5)
    obs, _ = env.reset(seed=seed)
    policy.reset()
    actions, explanations = [], []
    for _ in range(steps):
        if explainer is not None:
            explanations.append(explainer.explain(obs))
        a = np.asarray(policy.act(obs))
        actions.append(a)
        obs, *_ = env.step(a)
    return actions, explanations


def test_policy_explainer_does_not_change_decisions(trained):
    from supervisor.train import load_supervisor

    policy = load_supervisor(trained, "final")
    plain, _ = _play(policy, None)
    explained, explanations = _play(policy, make_explainer(policy))
    assert all((a == b).all() for a, b in zip(plain, explained))
    # La acción explicada (más probable) es la que tomó el supervisor determinista
    for a, ex in zip(explained, explanations):
        for seat, x in enumerate(ex):
            if x is not None:
                assert x["action"] == a[seat]


def test_integrated_gradients_completeness(trained):
    from supervisor.train import load_supervisor

    policy = load_supervisor(trained, "final")
    explainer = make_explainer(policy, steps=256)
    _, explanations = _play(policy, explainer, steps=12)
    checked = 0
    for ex in explanations:
        for x in ex:
            if x is None:
                continue
            total = sum(x["attr"].values()) + x["other_seats"] + x["global"]
            assert total == pytest.approx(x["target"] - x["base"], abs=0.02 + 0.02 * abs(x["target"] - x["base"]))
            assert sum(x["probs"]) == pytest.approx(1, abs=1e-3)
            checked += 1
    assert checked > 0
