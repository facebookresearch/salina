#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from salina import Agent


class RewardToGoAgent(Agent):
    """
    Compute the reward to go based on a workspace where complete episodes are stored
    The reward to go value is avaialble for all timesteps before env/done is True (end of the episode)
    """

    def __init__(
        self,
        input_reward_name="env/reward",
        input_done_name="env/done",
        output_reward_name="reward_to_go",
        output_timestep_name="timesteps_to_go",
    ):
        super().__init__()
        self.i_r_name = input_reward_name
        self.i_d_name = input_done_name
        self.o_r_name = output_reward_name
        self.o_t_name = output_timestep_name

    def forward(self, scaling_factor=1.0, *args):
        reward = self.get(self.i_r_name)
        reward = torch.cat(
            [reward[1:], torch.zeros_like(reward[0]).unsqueeze(0)], dim=0
        )
        done = self.get(self.i_d_name)
        T, B = done.size()[0:2]
        length = done.float().argmax(0)
        arange = torch.arange(T).to(done.device).unsqueeze(-1).repeat(1, B)
        _length = length.unsqueeze(0).repeat(T, 1) + 1
        mask = arange.ge(_length).float()
        _zeros = torch.zeros_like(reward)
        _reward = (1 - mask) * reward + mask * _zeros

        _reward = torch.flip(_reward, (0,))
        _creward = torch.cumsum(_reward, 0)
        _creward = torch.flip(_creward, (0,))

        t = (1 - mask) * torch.ones_like(reward) + mask * _zeros
        t = torch.flip(t, (0,))
        ct = torch.cumsum(t, 0)
        ct = torch.flip(ct, (0,)).long()

        self.set(self.o_r_name, _creward * scaling_factor)
        self.set(self.o_t_name, ct)
