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
    def __init__(self,name=None):
        super().__init__()
        self._name=name

    def seed(self, seed):
        print("[", type(self), "] Seed not implemented")

    def set_name(self,n):
        self._name=n

    def get_name(self):
        return self._name

    def __call__(self, workspace, **args):
        assert not workspace is None, "[Agent.__call__] workspace must not be None"
        self.workspace = workspace
        self.forward(**args)
        w = self.workspace
        self.workspace = None

    def _asynchronous_call(self,workspace,**args):
        return self.__call__(workspace,**args)

    def is_running(self):
        return False

    def forward(self, **args):
        raise NotImplemetedError

    def clone(self):
        self.workspace = None
        self.zero_grad()
        return copy.deepcopy(self)

    def get(self, index):
        if isinstance(index, str):
            return self.workspace.get_full(index)
        else:
            return self.workspace.get(index[0], index[1])

    def set(self, index, value):
        if isinstance(index, str):
            self.workspace.set_full(index, value)
        else:
            self.workspace.set(index[0], index[1],value)

    def get_by_name(self,n):
        if n==self._name:
            return [self]
        return []

class TAgent(Agent):
    """A specific agent that uses a timestep as an input"""

    def forward(self, t, **args):
        raise NotImplemetedError
