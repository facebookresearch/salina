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

    def __init__(self, name=None):
        super().__init__()
        self._name = name
        self.__trace_file=None

    def seed(self, seed):
        pass
        # print("[", type(self), "] Seed not implemented")

    def set_name(self, n):
        self._name = n

    def get_name(self):
        return self._name

    def set_trace_file(self,filename):
        print("[TRACE]: Tracing agent in file "+filename)
        self.__trace_file=open(filename,"wt")

    def __call__(self, workspace, **kwargs):
        assert not workspace is None, "[Agent.__call__] workspace must not be None"
        self.workspace = workspace
        self.forward(**kwargs)
        w = self.workspace
        self.workspace = None

    def _asynchronous_call(self, workspace, **kwargs):
        return self.__call__(workspace, **kwargs)

    def is_running(self):
        return False

    def forward(self, **kwargs):
        raise NotImplementedError

    def clone(self):
        self.workspace = None
        self.zero_grad()
        return copy.deepcopy(self)

    def get(self, index):
        if not self.__trace_file is None:
            t=time.time()
            self.__trace_file.write(str(self)+" type = "+type(self)+" time = ",t," get ",index,"\n")
        if isinstance(index, str):
            return self.workspace.get_full(index)
        else:
            return self.workspace.get(index[0], index[1])

    def get_time_truncated(self,var_name,from_time,to_time):
        return self.workspace.get_time_truncated(var_name,from_time,to_time)

    def set(self, index, value):
        if not self.__trace_file is None:
            t=time.time()
            self.__trace_file.write(str(self)+" type = "+type(self)+" time = ",t," set ",index," = ",value.size(),"/",value.dtype,"\n")
        if isinstance(index, str):
            self.workspace.set_full(index, value)
        else:
            self.workspace.set(index[0], index[1], value)

    def get_by_name(self, n):
        if n == self._name:
            return [self]
        return []

    def __del__(self):
        if self.__trace_file is not None:
            self.__trace_file.close()


class TAgent(Agent):
    """A specific agent that uses a timestep as an input"""

    def forward(self, t, **kwargs):
        raise NotImplementedError
