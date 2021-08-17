#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
from brax.envs.to_torch import JaxToTorchWrapper
from brax.envs import _envs, create_gym_env

from salina import TAgent


def _torch_cat_dict(d):
    r = {}
    for k in d[0]:
        r[k] = torch.cat([dd[k] for dd in d], dim=0)
    return r


class BraxAgent(TAgent):
    def __init__(
        self, env_name = None, input="action", output="env/", **args
    ):
        super().__init__()
        self.args=args
        self.brax_env_name = env_name
        self.gym_env = None
        self.batch_size = None
        self._seed = None
        self.output = output
        self.input = input
        self._device=torch.device("cpu")

    def _initialize_envs(self, batch_size):
        assert self._seed is not None, "[GymAgent] seeds must be specified"
        assert self.batch_size is None

        self.gym_env=create_gym_env(self.brax_env_name,batch_size=batch_size,seed=self._seed,**self.args)
        self.gym_env=JaxToTorchWrapper(self.gym_env,device=self._device)
        print("Creating BRAX env on ",self._device)

    def to(self,device):
        super().to(device)
        self._device=torch.device(device)

    def _write(self,v,t):
        for k in v:
            self.set((self.output+k,t),v[k])

    def forward(self, t=0, **args):
        if self.gym_env is None:
            self._initialize_envs(self.workspace.batch_size())
            self.batch_size=self.workspace.batch_size()
            self.timestep=0

        if t == 0 or self.timestep==0:
            self.timestep=0
            o=self.gym_env.reset()
            self.cumulated_reward=torch.zeros(self.batch_size,device=self._device).float()
            ret = {
                "env_obs":o,
                "done": torch.tensor([False],device=self._device).repeat(self.batch_size),
                "initial_state": torch.tensor([True],device=self._device).repeat(self.batch_size),
                "reward": torch.zeros(self.batch_size,device=self._device).float(),
                "timestep": torch.zeros(self.batch_size,device=self._device).long(),
                "cumulated_reward": self.cumulated_reward,
            }
            self._write(ret,t)
            self.timestep+=1
            return
        else:

            action=self.get((self.input,t-1))
            assert action.device==self._device
            obs, rewards, done, info = self.gym_env.step(action)
            self.cumulated_reward+=rewards
            done=done.bool()
            ret = {
                "env_obs":obs,
                "done": done,
                "initial_state": torch.tensor([False],device=self._device).repeat(self.batch_size),
                "reward": rewards.float(),
                "timestep": torch.tensor([self.timestep],device=self._device).repeat(self.batch_size).long(),
                "cumulated_reward": self.cumulated_reward,
            }
            if done.any():
                assert done.all()
                self.timestep=0
            else:
                self.timestep+=1
            self._write(ret,t)

    def seed(self, seed):
        self._seed = seed
