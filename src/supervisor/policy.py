"""Encoder de la política (sección 6.5): un MLP compartido por silla + uno global.

Los pesos de silla son los mismos para las 5, así que "contador" significa lo mismo en
cualquier asiento. Las salidas se concatenan en orden de silla para que la cabeza de
acción (o la LSTM de RecurrentPPO) sepa a qué silla corresponde cada parte.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


def mlp(n_in: int, hidden: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for n_out in hidden:
        layers += [nn.Linear(n_in, n_out), nn.ReLU()]
        n_in = n_out
    return nn.Sequential(*layers)


class SeatSetEncoder(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        seat_hidden: tuple[int, ...] = (64, 64),
        global_hidden: tuple[int, ...] = (32,),
    ):
        n_seats, n_features = observation_space["seats"].shape
        n_global = observation_space["global"].shape[0]
        super().__init__(observation_space, n_seats * seat_hidden[-1] + global_hidden[-1])
        self.seat_mlp = mlp(n_features, tuple(seat_hidden))
        self.global_mlp = mlp(n_global, tuple(global_hidden))

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        seats = self.seat_mlp(obs["seats"])            # (batch, sillas, oculto)
        return torch.cat([seats.flatten(1), self.global_mlp(obs["global"])], dim=1)
