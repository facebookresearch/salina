#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import copy

import torch
import torch.nn as nn

import salina


class Agent(nn.Module):
    """The core class in salina. It describes an agent that read and write into a workspace"""

    def __init__(self):
        super().__init__()

    def seed(self, seed):
        print("[", type(self), "] Seed not implemented")

    def __call__(self, workspace, **args):
        assert not workspace is None, "[Agent.__call__] workspace must not be None"
        self.workspace = workspace
        self.forward(**args)
        w = self.workspace
        self.workspace = None
        return w

    def forward(self, **args):
        raise NotImplemetedError

    def clone(self):
        self.workspace = None
        self.zero_grad()
        return copy.deepcopy(self)

    def get(self, index):
        if salina.trace_workspace:
            _id = str(type(self)) + "_" + str(hex(id(self)))
            self.workspace._put_in_trace(("get", _id, index))

        if isinstance(index, str):
            return self.workspace[index]
        else:
            return self.workspace.get(index[0], index[1])

    def set(self, index, value, use_workspace_device=False):
        if salina.trace_workspace:
            _id = str(type(self)) + "_" + str(hex(id(self)))
            self.workspace._put_in_trace(("set", _id, index))

        if use_workspace_device:
            value = value.to(self.workspace.device())

        if isinstance(index, str):
            self.workspace._set_sequence(index, value)
        else:
            self.workspace.set(index[0], index[1], value)


class TAgent(Agent):
    """A specific agent that uses a timestep as an input"""

    def forward(self, t, **args):
        raise NotImplemetedError
