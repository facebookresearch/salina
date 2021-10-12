#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from gym.utils import seeding
from torch.utils.data import DataLoader

from salina import Agent


class ShuffledDatasetAgent(Agent):
    """An agent that read a dataset in a shuffle order, in an infinte way."""

    def __init__(
        self,
        dataset,
        batch_size,
        output_names=("x", "y"),
    ):
        super().__init__()
        self.output_names = output_names
        self.dataset = dataset
        self.batch_size = batch_size
        self.ghost_params = torch.nn.Parameter(torch.randn(()))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def forward(self, **args):
        vs = []
        for k in range(self.batch_size):
            idx = self.np_random.randint(len(self.dataset))
            x = self.dataset[idx]
            xs = []
            for xx in x:
                if isinstance(xx, torch.Tensor):
                    xs.append(xx.unsqueeze(0))
                else:
                    xs.append(torch.tensor(xx).unsqueeze(0))
            vs.append(xs)

        vals = []
        for k in range(len(vs[0])):
            val = [v[k] for v in vs]
            val = torch.cat(val, dim=0)
            vals.append(val)

        for name, value in zip(self.output_names, vals):
            self.set((name, 0), value.to(self.ghost_params.device))


class DataLoaderAgent(Agent):
    """An agent based on a DataLoader that read a single dataset
    It also output mask value to tell if finished

    Args:
        TAgent ([type]): [description]
    """

    def __init__(self, dataloader, output_names=("x", "y")):
        super().__init__()
        self.dataloader = dataloader
        self.iter = iter(self.dataloader)
        self.output_names = output_names
        self._finished = False
        self.ghost_params = torch.nn.Parameter(torch.randn(()))

    def reset(self):
        self.iter = iter(self.dataloader)
        self._finished = False

    def finished(self):
        return self._finished

    def forward(self, **args):
        try:
            output_values = next(self.iter)
        except StopIteration:
            self.iter = None
            self._finished = True
        else:
            for name, value in zip(self.output_names, output_values):
                self.set((name, 0), value.to(self.ghost_params.device))
