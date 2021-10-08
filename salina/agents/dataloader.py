#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from gym.utils import seeding
from torch.utils.data import DataLoader

from salina import TAgent


class ShuffledDataLoaderAgent(TAgent):
    """An agent that read a dataset in a shuffle order, in an infinte way."""

    def __init__(
        self,
        dataset,
        output_names,
        batch_size,
    ):
        super().__init__()
        self.output_names = output_names
        self.dataset = dataset
        self.batch_size = batch_size

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def forward(self, t, **args):
        vs = []
        for k in range(self.workspace.batch_size()):
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

        for name, value in zip(self.output, vals):
            self.set((name, t), value, use_workspace_device=True)


class DataLoaderAgent(TAgent):
    """An agent based on a DataLoader that read a single dataset
    It also output mask value to tell if finished

    Args:
        TAgent ([type]): [description]
    """

    def __init__(self, dataset, output, **args):
        super().__init__()
        self.output = output
        self.args = args
        self.iter = None
        self.dataloader = None
        self.dataset = dataset
        self.batch_size = None

    def forward(self, t, **args):

        if self.dataloader is None:
            print(
                "[DataLoaderAgent] Creating dataloader with batch size = ",
                self.workspace.batch_size(),
            )
            self.batch_size = self.workspace.batch_size()
            self.dataloader = DataLoader(
                self.dataset, batch_size=self.batch_size, **self.args
            )

        assert (
            self.batch_size == self.workspace.batch_size()
        ), "[DataLoaderAgent] Batch size cannot be changed."

        if self.iter is None:
            self.iter = iter(self.dataloader)

        try:
            output_values = next(self.iter)
        except StopIteration:
            self.iter = None
            mask = torch.zeros(self.batch_size, dtype=torch.bool)
        else:
            if len(output_values[0]) < self.batch_size:
                mask = torch.zeros(self.batch_size, dtype=torch.bool)
                mask[: len(output_values[0])] = True

                for idx, value in enumerate(output_values):
                    output_values[idx] = torch.cat(
                        (
                            value,
                            torch.zeros(
                                (self.batch_size - value.shape[0], *value.shape[1:]),
                                dtype=value.dtype,
                            ),
                        )
                    )
            else:
                mask = torch.ones(self.batch_size, dtype=torch.bool)

            for name, value in zip(self.output, output_values):
                self.set((name, t), value, use_workspace_device=True)

        self.set((self.output[0] + "/mask", t), mask, use_workspace_device=True)
