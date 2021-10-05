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

    def set(self,t,value,batch_dims):
        assert batch_dims is None,"Unable to use batch dimensions with SlicedTemporalTensor"
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

    def to(self,device):
        s=SlicedTemporalTensor()
        for k in range(len(self.tensors)):
            s.set(k,self.tensors[k].to(device))
        return s

    def get(self,t,batch_dims):
        assert batch_dims is None,"Unable to use batch dimensions with SlicedTemporalTensor"
        assert t<len(self.tensors),"Temporal index out of bouds"
        return self.tensors[t]

    def get_full(self,batch_dims):
        assert batch_dims is None,"Unable to use batch dimensions with SlicedTemporalTensor"
        return torch.cat([a.unsqueeze(0) for a in self.tensors],dim=0)

    def set_full(self,value,batch_dims):
        assert batch_dims is None,"Unable to use batch dimensions with SlicedTemporalTensor"
        for t in range(value.size()[0]):
            self.set(t,value[t],batch_dims=batch_dims)

    def time_size(self):
        return len(self.tensors)

    def batch_size(self):
        return self.tensors[0].size()[0]

    def clear(self):
        self.tensors=[]
        self.size=None
        self.device=None
        self.dtype=None

    def copy_time(self,from_time,to_time,n_steps):
        for t in range(n_steps):
            v=self.get(from_time+t,batch_dims=None)
            self.set(to_time+t,v,batch_dims=None)

    def subtime(self,from_t,to_t):
        return CompactTemporalTensor(torch.cat([a.unsqueeze(0) for a in self.tensors[from_t:to_t]],dim=0))

    def zero_grad(self):
        self.tensors=[v.detach() for v in self.tensors]

class CompactSharedTensor:
    def __init__(self,_tensor):
        self.tensor=_tensor
        self.tensor.share_memory_()

    def set(self,t,value,batch_dims):
        if batch_dims is None:
            self.tensor[t]=value.detach()
        else:
            self.tensor[t,batch_dims[0]:batch_dims[1]]=value.detach()

    def get(self,t,batch_dims):
        assert t<self.tensor.size()[0],"Temporal index out of bouds"
        if batch_dims is None:
            return self.tensor[t]
        else:
            return self.tensor[t,batch_dims[0]:batch_dims[1]]

    def to(self,device):
        if device==self.tensor.device: return self
        t=self.tensor.to(device)
        t.share_memory_()
        return CompactSharedTensor(t)

    def get_full(self,batch_dims):
        if batch_dims is None:
            return self.tensor
        else:
            return self.tensor[:,batch_dims[0]:batch_dims[1]]

    def time_size(self):
        return self.tensor.size()[0]

    def batch_size(self):
        return self.tensor.size()[1]


    def set_full(self,value,batch_dims):
        if batch_dims is None:
            self.tensor=value.detach()
        else:
            self.tensor[:,batch_dims[0]:batch_dims[1]]=value.detach()

    def clear(self):
        assert False,"Cannot clear a shared tensor"

    def subtime(self,from_t,to_t):
        t=self.tensor[from_t:to_t]
        t.share_memory_()
        return CompactSharedTensor(t)

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

    def set(self,t,value,batch_dims):
        assert False
        assert not self.tensor is None,"Tensor must be initialized"
        assert self.size[1:]==value.size(),"Incompatible size"
        assert self.device==value.device,"Incompatible device"
        assert self.dtype==value.dtype,"Incompatible type"
        assert t<self.tensor.size()[0],"Temporal index out of bounds"
        if batch_dims is None:
            self.tensor[t]=value
        else:
            self.tensor[t,batch_dims[0]:batch_dims[1]]=value

    def to_sliced(self):
        v=SlicedTemporalTensor()
        for t in range(self.tensor.size()[0]):
            v.set(t,self.tensor[t],None)
        return v

    def to(self,device):
        if device==self.tensor.device: return self
        t=self.tensor.to(device)
        return CompactTemporalTensor(t)

    def get(self,t,batch_dims):
        assert t<self.tensor.size()[0],"Temporal index out of bouds"
        if batch_dims is None:
            return self.tensor[t]
        else:
            return self.tensor[t,batch_dims[0]:batch_dims[1]]

    def get_full(self,batch_dims):
        if batch_dims is None:
            return self.tensor
        else:
            return self.tensor[:,batch_dims[0]:batch_dims[1]]

    def time_size(self):
        return self.tensor.size()[0]

    def batch_size(self):
        return self.tensor.size()[1]


    def set_full(self,value,batch_dims):
        if self.tensor is None:
            assert batch_dims is None
            self.size=value.size()
            self.dtype=value.dtype
            self.device=value.device
        if batch_dims is None:
            self.tensor=value
        else:
            self.tensor[:,batch_dims[0]:batch_dims[1]]=value

    def subtime(self,from_t,to_t):
        return CompactTemporalTensor(self.tensor[from_t:to_t])


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
    def __init__(self,workspace=None):
        self.variables = {}
        self.is_shared=False
        if not workspace is None:
            for k in workspace.keys():
                self.set_full(k,workspace[k].clone())

    def set(self, var_name, t, v,batch_dims=None):
        if not var_name in self.variables:
            assert not self.is_shared,"Cannot add new variable into a shared workspace"
            self.variables[var_name]=SlicedTemporalTensor()
        elif isinstance(self.variables[var_name],CompactTemporalTensor):
            self.variables[var_name]=self.variables[var_name].to_sliced()
        self.variables[var_name].set(t,v,batch_dims=batch_dims)

    def get(self, var_name, t,batch_dims=None):
        assert var_name in self.variables,"Unknoanw variable '"+var_name+"'"
        return self.variables[var_name].get(t,batch_dims=batch_dims)

    def clear(self):
        for k,v in self.variables.items():
            v.clear()

    def set_full(self, var_name, value,batch_dims=None):
        if not var_name in self.variables:
            assert not self.is_shared,"Cannot add new variable into a shared workspace"
            self.variables[var_name]=CompactTemporalTensor()
        self.variables[var_name].set_full(value,batch_dims=batch_dims)

    def get_full(self, var_name,batch_dims=None):
        assert var_name in self.variables, (
            "[Workspace.get_full] unnknown variable '" + var_name + "'"
        )
        return self.variables[var_name].get_full(batch_dims=batch_dims)

    def keys(self):
        return self.variables.keys()

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.get_full(key,None)
        else:
            return (self.get_full(k,None) for k in key)

    def _all_variables_same_time_size(self):
        _ts=None
        for k,v in self.variables.items():
            if _ts is None: _ts=v.time_size()
            if _ts!=v.time_size(): return False
        return True

    def time_size(self):
        _ts=None
        for k,v in self.variables.items():
            if _ts is None: _ts=v.time_size()
            assert _ts==v.time_size(),"Variables must have the same time size"
        return _ts

    def batch_size(self):
        _bs=None
        for k,v in self.variables.items():
            if _bs is None: _bs=v.batch_size()
            assert _bs==v.batch_size(),"Variables must have the same batch size"
        return _bs

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

    def to(self,device):
        workspace=Workspace()
        for k,v in self.variables.items():
            workspace.variables[k]=v.to(device)
        return workspace

    def _convert_to_shared_workspace(self,n_repeat=1):
        workspace=Workspace()
        for k,v in self.variables.items():
            value=v.get_full(None).detach()
            ts=[value for t in range(n_repeat)]
            value=torch.cat(ts,dim=1)
            workspace.variables[k]=CompactSharedTensor(value)
            workspace.is_shared=True
        return workspace

    def subtime(self,from_t,to_t):
        assert self._all_variables_same_time_size(),"All variables must have the same time size"
        workspace=Workspace()
        for k,v in self.variables.items():
            workspace.variables[k]=v.subtime(from_t,to_t)
        return workspace

    def __str__(self):
        r=["Workspace:"]
        for k,v in self.variables.items():
            r.append("\t"+k+": time_size = "+str(v.time_size())+", batch_size = "+str(v.batch_size()))
        return  "\n".join(r)

class _SplitSharedWorkspace:
    def __init__(self,workspace,batch_dims):
        self.workspace=workspace
        self.batch_dims=batch_dims
        self.is_shared=self.workspace.is_shared

    def set(self, var_name, t, v):
        self.workspace.set(var_name,t,v,batch_dims=self.batch_dims)

    def get(self, var_name, t):
        return self.workspace.get(var_name,t,batch_dims=self.batch_dims)

    def set_full(self, var_name, value):
        self.workspace.set_full(var_name,value,batch_dims=self.batch_dims)

    def get_full(self, var_name):
        return self.workspace.get_full(var_name,batch_dims=self.batch_dims)
