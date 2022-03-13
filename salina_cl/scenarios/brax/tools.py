#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#



import torch
from salina.agents.brax import BraxAgent
from brax.envs.to_torch import JaxToTorchWrapper
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

class AutoResetBraxAgent(BraxAgent):
   def __init__(self, n_envs, env_name = "", input = "action", output = "env/", **kwargs):
        super().__init__(n_envs,env_name)
        self.make_env_fn = kwargs["make_env_fn"]
        self.make_env_args = kwargs["make_env_args"]

   def _initialize_envs(self, n_envs):
        assert self._seed is not None, "[GymAgent] seeds must be specified"

        self.gym_env = self.make_env_fn(
            batch_size=n_envs,
            seed=self._seed,
           **self.make_env_args
        )
        self.gym_env = JaxToTorchWrapper(self.gym_env)

class NoAutoResetBraxAgent(Agents):
    """
    A BraxAgent without auto-reset
    """

    def __init__(self, **kwargs):
        agent1 = BraxAgent(**kwargs)
        agent2 = EpisodesDone(out_var="env/done")
        super().__init__(agent1, agent2)