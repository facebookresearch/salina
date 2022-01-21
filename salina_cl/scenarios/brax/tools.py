#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#



import torch
from salina.agents.brax import BraxAgent
from brax.envs.to_torch import JaxToTorchWrapper

class AutoResetBraxAgent(BraxAgent):
   def __init__(self, n_envs, env_name = "", input="action", output="env/", **kwargs):
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
