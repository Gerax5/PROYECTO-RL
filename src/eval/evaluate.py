"""Evalúa una política del supervisor en episodios con semillas fijas."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from env import CasinoTableEnv, RewardConfig, TableConfig

from .metrics import SupervisorMetrics


class Policy(Protocol):
    def reset(self) -> None: ...

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray: ...


def run_episode(env: CasinoTableEnv, policy: Policy, seed: int) -> SupervisorMetrics:
    metrics = SupervisorMetrics()
    obs, info = env.reset(seed=seed)
    policy.reset()
    metrics.update(info)
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(policy.act(obs))
        metrics.update(info, reward)
        done = terminated or truncated
    return metrics


def evaluate(
    policy: Policy,
    seeds: list[int],
    table_config: TableConfig | None = None,
    reward_config: RewardConfig | None = None,
    rounds_per_episode: int = 600,
) -> SupervisorMetrics:
    env = CasinoTableEnv(table_config, reward_config, rounds_per_episode)
    total = SupervisorMetrics()
    for seed in seeds:
        total = total.merge(run_episode(env, policy, seed))
    return total
