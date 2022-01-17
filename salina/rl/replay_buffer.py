#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from salina import Workspace
import copy

class ReplayBuffer:
    def __init__(self, max_size, device=torch.device("cpu")):
        self.max_size = max_size
        self.variables = None
        self.position = 0
        self.is_full = False
        self.device = device

    def put(self, workspace, time_size=None, padding=None):
        assert (
            workspace._all_variables_same_time_size()
        ), "Only works with workspace where all variables have the same time_size"
        T = workspace.time_size()
        if not time_size is None:
            assert time_size <= T
            n = T - time_size + 1
            if padding is None:
                padding = 1
            for t in range(0, n, padding):
                nworkspace = workspace.subtime(t, t + time_size)
                self.put(nworkspace)
            return

        all_tensors = {
            k: workspace.get_full(k).detach().to(self.device) for k in workspace.keys()
        }
        if self.variables is None:
            self.variables = {}
            for k, v in all_tensors.items():
                s = list(v.size())
                s[1] = self.max_size
                _s=copy.deepcopy(s)
                s[0]=_s[1]
                s[1]=_s[0]

                tensor = torch.zeros(*s, dtype=v.dtype, device=self.device)
                print(
                    "[ReplayBuffer] Var ",
                    k,
                    " size=",
                    s,
                    " dtype=",
                    v.dtype,
                    " device=",
                    self.device,
                )
                self.variables[k] = tensor
            self.is_full = False
            self.position = 0

        B = None
        arange = None
        indexes = None
        for k, v in all_tensors.items():
            if B is None:
                B = v.size()[1]
            B = min(self.position + B, self.max_size)
            B = B - self.position
            if indexes is None:
                indexes = torch.arange(B) + self.position
                arange = torch.arange(B)
            indexes = indexes.to(v.device)
            arange = arange.to(v.device)
            self.variables[k][indexes] = v.detach().transpose(0,1)

        self.position = self.position + B
        if self.position >= self.max_size:
            self.position = 0
            self.is_full = True

    def size(self):
        if self.is_full:
            return self.max_size
        else:
            return self.position

    def get(self, B):
        who = torch.randint(low=0, high=self.size(), size=(B,), device=self.device)
        workspace = Workspace()
        for k in self.variables:
            workspace.set_full(k, self.variables[k][who].transpose(0,1))

        return workspace

    def to(self,device):
        n_vars={k:v.to(device) for k,v in self.variables.items()}
        self.variables=n_vars
