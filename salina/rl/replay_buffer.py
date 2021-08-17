#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from salina import Workspace


class ReplayBuffer:
    def __init__(self, max_size,device=torch.device("cpu")):
        self.max_size = max_size
        self.variables = {}
        self.position = 0
        self.is_full = False
        self.time_size = None
        self.device=device

    def put(self, workspace, time_size=None, padding=None):
        if not time_size is None and time_size != workspace.time_size():
            T = workspace.time_size()
            n = T - time_size + 1
            if padding is None:
                padding = 1
            for t in range(0, n, padding):
                nworkspace = workspace.subtime(t, t + time_size)
                self.put(nworkspace)
            return

        dict_variables = workspace.to_dict()
        self.time_size = workspace.time_size()
        B = None
        indexes = None
        arange = None
        for k, v in dict_variables.items():
            if not k in self.variables:
                s = v.size()[2:]
                print(
                    "\t[ReplayBuffer] Creating variable ",
                    k,
                    " of size ",
                    (self.time_size, self.max_size, *s),
                    " and type ",
                    v.dtype,
                )
                self.variables[k] = torch.zeros(
                    self.time_size, self.max_size, *s, dtype=v.dtype,device=self.device
                )
            B = v.size()[1]
            B = min(self.position + B, self.max_size)
            B = B - self.position
            if indexes is None:
                indexes = torch.arange(B) + self.position
                arange = torch.arange(B,device=self.device)
            self.variables[k][:, indexes] = v[:, arange].detach()

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
        who = torch.randint(low=0, high=self.size(), size=(B,),device=self.device)
        workspace = Workspace(batch_size=B, time_size=self.time_size)
        workspace = workspace.to(self.device)
        for k in self.variables:
            workspace.variables[k] = self.variables[k][:, who]

        return workspace
