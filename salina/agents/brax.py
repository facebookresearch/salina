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
from salina.agents import Agents

class EpisodesDone(TAgent):
    """
    If done is encountered at time t, then done=True for all timeteps t'>=t
    It allows to simulate a single episode agent based on an autoreset agent
    """

    def __init__(self, in_var="env/done", out_var="env/done"):
        super().__init__()
        self.in_var = in_var
        self.out_var = out_var

    def forward(self, t, **kwargs):
        d = self.get((self.in_var, t))
        if t == 0:
            self.state = torch.zeros_like(d).bool()
        self.state = torch.logical_or(self.state, d)
        self.set((self.out_var, t), self.state)


def _torch_cat_dict(d):
    r = {}
    for k in d[0]:
        r[k] = torch.cat([dd[k] for dd in d], dim=0)
    return r

class BraxAgent(TAgent):
    """An agent based on a brax environment, with autoreset

    The agent reads `action` at `t-1` and outputs `env/env_obs`,`env/reward`,`env/initial_state`,`env/done`,`env/timestep`,`env/cumulated_reward`
    """

    def __init__(self, n_envs, env_name, input="action", output="env/", **kwargs):
        """ Initialize the agent

        Args:
            n_envs ([int]): number of envs (batch dimension)
            env_name ([str]): [the BRAX environment name
            input (str, optional):  Defaults to "action".
            output (str, optional): Defaults to "env/".
        """
        super().__init__()
        self.args = kwargs
        self.brax_env_name = env_name
        self.gym_env = None
        self._seed = None
        self.n_envs = n_envs
        self.output = output
        self.input = input
        self.brax_device = None
        self.ghost_params = torch.nn.Parameter(torch.randn(()))

    def _initialize_envs(self, n_envs):
        assert self._seed is not None, "[GymAgent] seeds must be specified"

        self.gym_env = create_gym_env(
            self.brax_env_name, batch_size=n_envs, seed=self._seed, **self.args
        )
        self.gym_env = JaxToTorchWrapper(self.gym_env)

    def _write(self, v, t):
        for k in v:
            self.set((self.output + k, t), v[k])

    def forward(self, t=0, **kwargs):
        if self.gym_env is None:
            self._initialize_envs(self.n_envs)

        if t == 0:

            o = self.gym_env.reset()
            if self.brax_device is None:
                self.brax_device = o.device
                #print(" -- BRAX Device is ", self.brax_device)
                self.to(self.brax_device)

            my_device = self.ghost_params.device
            self.timestep = torch.zeros(self.n_envs, device=my_device).long()
            self.cumulated_reward = torch.zeros(self.n_envs, device=my_device).float()

            ret = {
                "env_obs": o,
                "done": torch.tensor([False], device=my_device).repeat(self.n_envs),
                "initial_state": torch.tensor([True], device=my_device).repeat(
                    self.n_envs
                ),
                "reward": torch.zeros(self.n_envs, device=my_device).float(),
                "timestep": self.timestep,
                "cumulated_reward": self.cumulated_reward,
            }
            self._write(ret, t)
            self.timestep += 1
            return
        else:
            my_device = self.ghost_params.device
            action = self.get((self.input, t - 1))
            assert action.device == torch.device(my_device)
            obs, rewards, done, info = self.gym_env.step(action)
            obs = obs.to(my_device)
            rewards = rewards.to(my_device)
            done = done.to(my_device)
            self.cumulated_reward += rewards
            done = done.bool()
            ret = {
                "env_obs": obs.float(),
                "done": done,
                "initial_state": torch.tensor([False], device=my_device).repeat(
                    self.n_envs
                ),
                "reward": rewards.float(),
                "timestep": self.timestep,

                "cumulated_reward": self.cumulated_reward,
            }
            if done.any():
                assert done.all()
            # Mettre des To(my_device) partout
            self.timestep += 1
            self.timestep = ((1.0 - done.float()) * self.timestep).long()
            self.cumulated_reward = (1.0 - done.float()) * self.cumulated_reward
            self._write(ret, t)

    def seed(self, seed):
        self._seed = seed


class AutoResetBraxAgent(BraxAgent):
    """The same than BraxAgent"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class NoAutoResetBraxAgent(Agents):
    """
    A BraxAgent without auto-reset
    """

    def __init__(self, **kwargs):
        agent1 = BraxAgent(**kwargs)
        agent2 = EpisodesDone(out_var="env/done")
        super().__init__(agent1, agent2)
