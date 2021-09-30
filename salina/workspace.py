#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from __future__ import annotations

import copy
from typing import Dict, Iterable, List

import torch
from graphviz import Digraph

import salina

class SlicedTemporalTensor:
    def __init__(self):
        self.tensors=[]
        self.size=None
        self.device=None
        self.dtype=None

    def set(self,t,value):
        if self.size is None:
            self.size=value.size()
            self.device=value.device
            self.dtype=value.dtype

        assert self.size==value.size(),"Incompatible size"
        assert self.device==value.device,"Incompatible device"
        assert self.dtype==value.dtype,"Incompatible type"
        while (len(self.tensors)<=t):
            self.tensors.append(torch.zeros(*self.size,device=self.device,dtype=self.dtype))
        self.tensors[t]=value

    def get(self,t):
        assert t<len(self.tensors),"Temporal index out of bouds"
        return self.tensors[t]

    def get_all(self):
        return torch.cat([a.unsqueeze(0) for a in self.tensors],dim=0)

    def set_all(self,value):
        for t in range(value.size()[0]):
            self.set(t,value[t])

    def time_size(self):
        return len(self.tensors)

    def clear(self):
        self.tensors=[]
        self.size=None
        self.device=None
        self.dtype=None

    def copy_time(self,from_time,to_time,n_steps):
        for t in range(n_steps):
            v=self.get(from_time+t)
            self.set(to_time+t,v)

    def zero_grad(self):
        self.tensors=[v.detach() for v in self.tensors]

class CompactSharedTensor:
    def __init__(self,_tensor):
        self.tensor=_tensor.get_all().detach()
        self.tensor.share_memory_()

    def set(self,t,value):
        self.tensor[t]=value.detach()

    def get(self,t):
        assert t<self.tensor.size()[0],"Temporal index out of bouds"
        return self.tensor[t]

    def get_all(self):
        return self.tensor

    def time_size(self):
        return self.tensor.size()[0]

    def set_all(self,value):
        self.tensor=value.detach()

    def clear(self):
        assert False,"Cannot clear a shared tensor"

    def copy_time(self,from_time,to_time,n_steps):
        self.tensor[to_time:to_time+n_steps]=self.tensor[from_time:from_time+n_steps]


    def zero_grad(self):
        pass

class CompactTemporalTensor:
    def __init__(self,value=None):
        self.size=None
        self.device=None
        self.dtype=None
        self.tensor=None
        if not value is None:
            self.tensor=value
            self.device=value.device
            self.size=value.size()
            self.dtype=value.dtype

    def set(self,t,value):
        assert not self.tensor is None,"Tensor must be initialized"
        assert self.size[1:]==value.size(),"Incompatible size"
        assert self.device==value.device,"Incompatible device"
        assert self.dtype==value.dtype,"Incompatible type"
        assert t<self.tensor.size()[0],"Temporal index out of bounds"
        self.tensor[t]=value

    def get(self,t):
        assert t<self.tensor.size()[0],"Temporal index out of bouds"
        return self.tensor[t]

    def get_all(self):
        return self.tensor

    def time_size(self):
        return self.tensor.size()[0]


    def set_all(self,value):
        if self.tensor is None:
            self.size=value.size()
            self.dtype=value.dtype
            self.device=value.device
        self.tensor=value

    def clear(self):
        self.size=None
        self.device=None
        self.dtype=None
        self.tensor=None

    def copy_time(self,from_time,to_time,n_steps):
        self.tensor[to_time:to_time+n_steps]=self.tensor[from_time:from_time+n_steps]

    def zero_grad(self):
        self.tensor=self.tensor.detach()

class Workspace:
    """A workspace is a collection of tensors indexed by name and time. The first dimension of each tensor is the batch dimension"""
    def __init__(self):
        self.variables = {}
        self.is_shared=False

    def set(self, var_name, t, v):
        if not var_name in self.variables:
            assert not self.is_shared,"Cannot add new variable into a shared workspace"
            self.variables[var_name]=SlicedTemporalTensor()
        self.variables[var_name].set(t,v)

    def get(self, var_name, t):
        assert var_name in self.variables,"Unknoanw variable '"+var_name+"'"
        return self.variables[var_name].get(t)

    def clear(self):
        for k,v in self.variables.items():
            v.clear()

    def set_all(self, var_name, value):
        if not var_name in self.variables:
            assert not self.is_shared,"Cannot add new variable into a shared workspace"
            self.variables[var_name]=CompactTemporalTensor()
        self.variables[var_name].set_all(value)

    def get_all(self, var_name):
        assert var_name in self.variables, (
            "[Workspace.get_all] unnknown variable '" + var_name + "'"
        )
        return self.variables[var_name].get_all()

    def keys(self):
        return self.variables.keys()

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.get_all(key)
        else:
            return (self.get_all(k) for k in key)

    def time_size(self,var_name):
        return self.variables[var__name].time_size()

    def copy_time(self, from_time, to_time,n_steps,var_names=None):
        for k,v in self.variables.items():
            if var_names is None or k in var_names:
                v.copy_time(from_time,to_time,n_steps)

    def copy_n_last_steps(self, n,var_names=None):
        _ts=None
        for k,v in self.variables.items():
            if var_names is None or k in var_names:
                if _ts is None: _ts=v.time_size()
                assert _ts==v.time_size(),"Variables must have the same time size"

        for k,v in self.variables.items():
            if var_names is None or k in var_names:
                self.copy_time(_ts-n,0,n)

    def zero_grad(self):
        for k,v in self.variables.items():
            v.zero_grad()

    def _convert_to_shared_workspace(self):
        workspace=Workspace()
        for k,v in self.variables.items():
            workspace.variables[k]=CompactSharedTensor(v)
            workspace.is_shared=True
        return workspace

class WorkspaceArray:
    def __init__(self,workspaces):
        self.workspaces=workspaces

    def __getitem__(self,k):
        if isinstance(k,int):
            return self.workspaces[k]
        else:
            return [w[k] for w in self.workspaces]

    def __len__(self):
        return len(self.workspaces)

    def to_workspace(self):
        w=Workspace()
        for k,v in self.workspaces[0].variables.items():
            tensors=[w.variables[k].get_all().clone() for w in self.workspaces]
            tensor=torch.cat(tensors,dim=1)
            w.set_all(k,tensor)
        return w

    def copy_n_last_steps(self, n,var_names=None):
        [w.copy_n_last_steps(n,var_names=var_names) for w in self.workspaces]


def create_shared_workspaces_array(agent,n_workspaces=1,**args):
    workspace=Workspace()
    agent(workspace,**args)
    return WorkspaceArray([workspace._convert_to_shared_workspace() for k in range(n_workspaces)])
