#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
from brax.envs import _envs, create_gym_env
from brax.envs.to_torch import JaxToTorchWrapper

from salina import TAgent


def _torch_cat_dict(d):
    r = {}
    for k in d[0]:
        r[k] = torch.cat([dd[k] for dd in d], dim=0)
    return r


class BraxAgent(TAgent):
    def __init__(self, n_envs, env_name, input="action", output="env/", **args):
        super().__init__()
        self.args = args
        self.brax_env_name = env_name
        self.gym_env = None
        self._seed = None
        self.n_envs = n_envs
        self.output = output
        self.input = input
        self._device = None

    def _initialize_envs(self, n_envs):
        assert self._seed is not None, "[GymAgent] seeds must be specified"

        self.gym_env = create_gym_env(
            self.brax_env_name, batch_size=n_envs, seed=self._seed, **self.args
        )
        self.gym_env = JaxToTorchWrapper(self.gym_env)

    def _write(self, v, t):
        for k in v:
            self.set((self.output + k, t), v[k])

    def forward(self, t=0, **args):
        if self.gym_env is None:
            self._initialize_envs(self.n_envs)

        if t == 0:
            self.timestep = torch.zeros(self.n_envs, device=self._device).long()
            self.cumulated_reward = torch.zeros(
                self.n_envs, device=self._device
            ).float()
            o = self.gym_env.reset()
            if self._device is None:
                self._device = o.device
                print(" -- BRAX Device is ", self._device)
            ret = {
                "env_obs": o,
                "done": torch.tensor([False], device=self._device).repeat(self.n_envs),
                "initial_state": torch.tensor([True], device=self._device).repeat(
                    self.n_envs
                ),
                "reward": torch.zeros(self.n_envs, device=self._device).float(),
                "timestep": self.timesteps,
                "cumulated_reward": self.cumulated_reward,
            }
            self._write(ret, t)
            self.timestep += 1
            return
        else:
            action = self.get((self.input, t - 1))
            assert action.device == torch.device(self._device)
            obs, rewards, done, info = self.gym_env.step(action)
            self.cumulated_reward += rewards
            done = done.bool()
            ret = {
                "env_obs": obs,
                "done": done,
                "initial_state": torch.tensor([False], device=self._device).repeat(
                    self.n_envs
                ),
                "reward": rewards.float(),
                "timestep": self.timesteps,
                "cumulated_reward": self.cumulated_reward,
            }
            self.timestep += 1
            self.timestep = (1.0 - done.float()) * self.timestep
            self.cumulated_reward = (1.0 - done.float()) * self.cumulated_reward
            self._write(ret, t)

    def seed(self, seed):
        self._seed = seed
