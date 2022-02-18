#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from __future__ import annotations

import copy

import numpy as np
import torch
import salina
from typing import List, Set, Dict, Tuple, Optional

""" This module provides different ways to store tensors that are more flexible than the torch.Tensor class
It also defines the `Workspace` as a dictionary of tensors and a version of the workspace where tensors are in shared memory for multiprocessing
"""

class SlicedTemporalTensor:
    """A SlicedTemporalTensor represents a tensor of size TxBx... by using a list of tensors of size Bx...
    The interest is that this tensor automatically adapts its timestep dimension and does not need to have a predefined size.
    """

    def __init__(self):
        """ Initialize an empty tensor
        """
        self.tensors: list[torch.Tensor] = []
        self.size: torch.Size = None
        self.device: torch.device = None
        self.dtype: torch.dtype = None

    def set(self, t:int, value:torch.Tensor, batch_dims:Optional[tuple(int,int)]):
        """Set a value (dim Bx...) at time t
        """
        assert (
            batch_dims is None
        ), "Unable to use batch dimensions with SlicedTemporalTensor"
        if self.size is None:
            self.size = value.size()
            self.device = value.device
            self.dtype = value.dtype
        assert self.size == value.size(), "Incompatible size"
        assert self.device == value.device, "Incompatible device"
        assert self.dtype == value.dtype, "Incompatible type"
        while len(self.tensors) <= t:
            self.tensors.append(
                torch.zeros(*self.size, device=self.device, dtype=self.dtype)
            )
        self.tensors[t] = value

    def to(self, device:torch.device):
        """Move the tensor to a specific device"""
        s = SlicedTemporalTensor()
        for k in range(len(self.tensors)):
            s.set(k, self.tensors[k].to(device))
        return s

    def get(self, t:int, batch_dims:Optional[tuple(int,int)]):
        """Get the value of the tensor at time t"""

        assert (
            batch_dims is None
        ), "Unable to use batch dimensions with SlicedTemporalTensor"
        assert t < len(self.tensors), "Temporal index out of bouds"
        return self.tensors[t]

    def get_full(self, batch_dims):
        """Returns the complete tensor of size TxBx..."""

        assert (
            batch_dims is None
        ), "Unable to use batch dimensions with SlicedTemporalTensor"
        return torch.cat([a.unsqueeze(0) for a in self.tensors], dim=0)

    def get_time_truncated(self, from_time:int, to_time:int, batch_dims:Optional[tuple(int,int)]):
        """Returns tensor[from_time:to_time]"""
        assert from_time >= 0 and to_time >= 0 and to_time > from_time
        assert batch_dims is None
        return torch.cat(
            [
                self.tensors[k].unsqueeze(0)
                for k in range(from_time, min(len(self.tensors), to_time))
            ],
            dim=0,
        )

    def set_full(self, value:torch.Tensor, batch_dims:Optional[tuple(int,int)]):
        """ Set the tensor given a BxTx... tensor. The input tensor is cut into slices that are stored in a list of tensors
        """
        assert (
            batch_dims is None
        ), "Unable to use batch dimensions with SlicedTemporalTensor"
        for t in range(value.size()[0]):
            self.set(t, value[t], batch_dims=batch_dims)

    def time_size(self):
        """
        Return the size of the time dimension
        """
        return len(self.tensors)

    def batch_size(self):
        """Return the size of the batch dimesion

        """
        return self.tensors[0].size()[0]

    def select_batch(self, batch_indexes:torch.LongTensor):
        """ Return the tensor where the batch dimension has been selected by the index

        """
        var = SlicedTemporalTensor()
        for t, v in enumerate(self.tensors):
            batch_indexes=batch_indexes.to(v.device)
            var.set(t, v[batch_indexes], None)
        return var

    def clear(self):
        """ Clear the tensor
        """
        self.tensors = []
        self.size = None
        self.device = None
        self.dtype = None

    def copy_time(self, from_time:int, to_time:int, n_steps:int):
        """ Copy temporal slices of the tensor from from_time:from_time+n_steps to to_time:to_time+n_steps
        """
        for t in range(n_steps):
            v = self.get(from_time + t, batch_dims=None)
            self.set(to_time + t, v, batch_dims=None)

    def subtime(self, from_t:int, to_t:int):
        """
            Return tensor[from_t:to_t]

        """
        return CompactTemporalTensor(
            torch.cat([a.unsqueeze(0) for a in self.tensors[from_t:to_t]], dim=0)
        )

    def zero_grad(self):
        """Clear any gradient information in the tensor
        """
        self.tensors = [v.detach() for v in self.tensors]

class CompactTemporalTensor:
    """ A CompactTemporalTensor is a tenosr of size TxBx... It behaves like the `SlicedTemporalTensor` but has a fixed size that cannot change. It is faster than the SlicedTemporalTensor.
        See `SlicedTemporalTensor`
    """
    def __init__(self, value: torch.Tensor=None):
        self.size = None
        self.device = None
        self.dtype = None
        self.tensor = None
        if not value is None:
            self.tensor = value
            self.device = value.device
            self.size = value.size()
            self.dtype = value.dtype

    def set(self, t, value, batch_dims):
        assert False
        assert not self.tensor is None, "Tensor must be initialized"
        assert self.size[1:] == value.size(), "Incompatible size"
        assert self.device == value.device, "Incompatible device"
        assert self.dtype == value.dtype, "Incompatible type"
        assert t < self.tensor.size()[0], "Temporal index out of bounds"
        if batch_dims is None:
            self.tensor[t] = value
        else:
            self.tensor[t, batch_dims[0] : batch_dims[1]] = value

    def select_batch(self, batch_indexes):
        v = CompactTemporalTensor(self.tensor[:, batch_indexes])
        return v

    def to_sliced(self) -> SlicedTemporalTensor :
        """ Transform the tensor to a s`SlicedTemporalTensor`
        """
        v = SlicedTemporalTensor()
        for t in range(self.tensor.size()[0]):
            v.set(t, self.tensor[t], None)
        return v

    def to(self, device):
        if device == self.tensor.device:
            return self
        t = self.tensor.to(device)
        return CompactTemporalTensor(t)



    def get(self, t, batch_dims):
        assert t < self.tensor.size()[0], "Temporal index out of bouds"
        if batch_dims is None:
            return self.tensor[t]
        else:
            return self.tensor[t, batch_dims[0] : batch_dims[1]]

    def get_full(self, batch_dims):
        if batch_dims is None:
            return self.tensor
        else:
            return self.tensor[:, batch_dims[0] : batch_dims[1]]

    def time_size(self):
        return self.tensor.size()[0]

    def batch_size(self):
        return self.tensor.size()[1]

    def set_full(self, value, batch_dims):
        if self.tensor is None:
            assert batch_dims is None
            self.size = value.size()
            self.dtype = value.dtype
            self.device = value.device
        if batch_dims is None:
            self.tensor = value
        else:
            self.tensor[:, batch_dims[0] : batch_dims[1]] = value

    def subtime(self, from_t, to_t):
        return CompactTemporalTensor(self.tensor[from_t:to_t])

    def clear(self):
        self.size = None
        self.device = None
        self.dtype = None
        self.tensor = None

    def copy_time(self, from_time, to_time, n_steps):
        self.tensor[to_time : to_time + n_steps] = self.tensor[
            from_time : from_time + n_steps
        ]

    def zero_grad(self):
        self.tensor = self.tensor.detach()

class CompactSharedTensor:
    """ It corresponds to a tensor in shared memory and is used when building a workspace shared by multiple processes.
        All the methods behaves like the methods of `SlicedTemporalTensor`
    """
    def __init__(self, _tensor:torch.Tensor):
        self.tensor = _tensor
        self.tensor.share_memory_()

    def set(self, t, value, batch_dims):
        if batch_dims is None:
            self.tensor[t] = value.detach()
        else:
            self.tensor[t, batch_dims[0] : batch_dims[1]] = value.detach()

    def get(self, t, batch_dims):
        assert t < self.tensor.size()[0], "Temporal index out of bouds"
        if batch_dims is None:
            return self.tensor[t]
        else:
            return self.tensor[t, batch_dims[0] : batch_dims[1]]

    def to(self, device):
        if device == self.tensor.device:
            return self
        t = self.tensor.to(device)
        t.share_memory_()
        return CompactSharedTensor(t)

    def select_batch(self, batch_indexes):
        v = CompactSharedTensor(self.tensor[:, batch_indexes])
        return v

    def get_full(self, batch_dims):
        if batch_dims is None:
            return self.tensor
        else:
            return self.tensor[:, batch_dims[0] : batch_dims[1]]

    def time_size(self):
        return self.tensor.size()[0]

    def batch_size(self):
        return self.tensor.size()[1]

    def set_full(self, value, batch_dims):
        if batch_dims is None:
            self.tensor = value.detach()
        else:
            self.tensor[:, batch_dims[0] : batch_dims[1]] = value.detach()

    def clear(self):
        assert False, "Cannot clear a shared tensor"

    def subtime(self, from_t, to_t):
        t = self.tensor[from_t:to_t]
        return CompactSharedTensor(t)

    def copy_time(self, from_time, to_time, n_steps):
        self.tensor[to_time : to_time + n_steps] = self.tensor[
            from_time : from_time + n_steps
        ]

    def zero_grad(self):
        pass

def take_per_row_strided(A, indx, num_elem=2):
    # TODO: Optimize this function
    all_indx = indx
    arange = torch.arange(A.size()[1], device=A.device)
    return torch.cat(
        [A[all_indx + t, arange].unsqueeze(0) for t in range(num_elem)], dim=0
    )


class Workspace:
    """ Workspace is the most important class in `SaLinA`. It correponds to a collection of tensors ('SlicedTemporalTensor`, `CompactTemporalTensor` or ` CompactShareTensor`).
        In the majority of cases, we consider that all the tensors have the same time and batch sizes (but it is not mandatory for most of the functions)
    """


    def __init__(self, workspace:Optional[Workspace]=None):
        """ Create an empty workspace

        Args:
            workspace (Workspace, optional): If specified, it creates a copy of the workspace (where tensors are cloned as CompactTemporalTensors)
        """
        self.variables = {}
        self.is_shared = False
        if not workspace is None:
            for k in workspace.keys():
                self.set_full(k, workspace[k].clone())

    def set(self, var_name:str, t:int, v:torch.Tensor, batch_dims:Optional[tuple(int,int)]=None):
        """ Set the variable var_name at time t
        """
        if not var_name in self.variables:
            assert not self.is_shared, "Cannot add new variable into a shared workspace"
            self.variables[var_name] = SlicedTemporalTensor()
        elif isinstance(self.variables[var_name], CompactTemporalTensor):
            self.variables[var_name] = self.variables[var_name].to_sliced()

        self.variables[var_name].set(t, v, batch_dims=batch_dims)

    def get(self, var_name:str, t:int, batch_dims:Optional[tuple(int,int)]=None) -> torch.Tensor:
        """ Get the variable var_name at time t
        """
        assert var_name in self.variables, "Unknoanw variable '" + var_name + "'"
        return self.variables[var_name].get(t, batch_dims=batch_dims)

    def clear(self,name=None):
        """ Remove all the variables from the workspace
        """
        if name is None:
            for k, v in self.variables.items():
                v.clear()
            self.variables={}
        else:
            self.variables[name].clear()
            del(self.variables[name])

    def contiguous(self) -> Workspace:
        """ Generates a workspace where all tensors are stored in the Compact format.
        """
        workspace=Workspace()
        for k in self.keys():
            workspace.set_full(k,self.get_full(k))
        return workspace

    def set_full(self, var_name:str, value:torch.Tensor, batch_dims:Optional[tuple(int,int)]=None):
        """ Set variable var_name with a complete tensor (TxBx...)
        """
        if not var_name in self.variables:
            assert not self.is_shared, "Cannot add new variable into a shared workspace"
            self.variables[var_name] = CompactTemporalTensor()
        self.variables[var_name].set_full(value, batch_dims=batch_dims)

    def get_full(self, var_name:str, batch_dims:Optional[tuple(int,int)]=None) -> torch.Tensor:
        """ Return the complete tensor for var_name
        """
        assert var_name in self.variables, (
            "[Workspace.get_full] unnknown variable '" + var_name + "'"
        )
        return self.variables[var_name].get_full(batch_dims=batch_dims)

    def keys(self):
        """ Return an interator over the variables names
        """
        return self.variables.keys()

    def __getitem__(self, key):
        """ if key is a string, then it returns a torch.Tensor
        if key is a list of string, it returns a tuple of torch.Tensor
        """
        if isinstance(key, str):
            return self.get_full(key, None)
        else:
            return (self.get_full(k, None) for k in key)

    def _all_variables_same_time_size(self) -> bool:
        """ Check that all variables have the same time size
        """
        _ts = None
        for k, v in self.variables.items():
            if _ts is None:
                _ts = v.time_size()
            if _ts != v.time_size():
                return False
        return True

    def time_size(self) -> int :
        """ Return the time size of the variables in the workspace
        """
        _ts = None
        for k, v in self.variables.items():
            if _ts is None:
                _ts = v.time_size()
            assert _ts == v.time_size(), "Variables must have the same time size"
        return _ts

    def batch_size(self) -> int :
        """ Return the batch size of the variables in the workspace
        """
        _bs = None
        for k, v in self.variables.items():
            if _bs is None:
                _bs = v.batch_size()
            assert _bs == v.batch_size(), "Variables must have the same batch size"
        return _bs

    def select_batch(self, batch_indexes:torch.LongTensor) -> Workspace:
        """ Given a tensor of indexes, it returns a new workspace with the select elements (over the batch dimension)
        """
        _bs = None
        for k, v in self.variables.items():
            if _bs is None:
                _bs = v.batch_size()
            assert _bs == v.batch_size(), "Variables must have the same batch size"

        workspace = Workspace()
        for k, v in self.variables.items():
            v = v.select_batch(batch_indexes)
            workspace.variables[k] = v
        return workspace

    def select_batch_n(self, n):
        """ Return a new Workspace of batch_size==n by randomly sampling over the batch dimension
        """
        who = torch.randint(low=0, high=self.batch_size(), size=(n,))
        return self.select_batch(who)

    def copy_time(self, from_time:int, to_time:int, n_steps:int, var_names:Optional[list[str]]=None):
        """ Copy all the variables values from time `from_time` to `from_time+n_steps` to `to_time` to `to_time+n_steps`
        It can be restricted to specific variables uusing `var_names`
        """
        for k, v in self.variables.items():
            if var_names is None or k in var_names:
                v.copy_time(from_time, to_time, n_steps)

    def get_time_truncated(self, var_name:str, from_time:int, to_time:int, batch_dims:Optional[tuple(int,int)]=None) -> torch.Tensor:
        """ Return workspace[var_name][from_time:to_time]
        """
        assert from_time >= 0 and to_time >= 0 and to_time > from_time

        v = self.variables[var_name]
        if isinstance(v, SlicedTemporalTensor):
            return v.get_time_truncated(from_time, to_time, batch_dims)
        else:
            return v.get_full(batch_dims)[from_time:to_time]

    def get_time_truncated_workspace(self,from_time:int, to_time:int) -> Workspace:
        """ Return a workspace where all variables are truncated between from_time and to_time
        """
        workspace=Workspace()
        for k in self.keys():
            workspace.set_full(k,self.get_time_truncated(k,from_time,to_time,None))
        return workspace

    #Static function
    def cat_batch(workspaces:list[Workspace]) -> Workspace:
        """ Concatenate multiple workspaces over the batch dimension. The workspaces must have the same time dimension.
        """

        ts = None
        for w in workspaces:
            if ts is None:
                ts = w.time_size()
            assert ts == w.time_size(), "Workspaces must have the same time_size"

        workspace = Workspace()
        for k in workspaces[0].keys():
            vals = [w[k] for w in workspaces]
            v = torch.cat(vals, dim=1)
            workspace.set_full(k, v)
        return workspace

    def copy_n_last_steps(self, n:int, var_names:Optional[list(str)]=None):
        """ Copy the n last timesteps of each variables to the n first timesteps.
        """
        _ts = None
        for k, v in self.variables.items():
            if var_names is None or k in var_names:
                if _ts is None:
                    _ts = v.time_size()
                assert _ts == v.time_size(), "Variables must have the same time size"

        for k, v in self.variables.items():
            if var_names is None or k in var_names:
                self.copy_time(_ts - n, 0, n)

    def zero_grad(self):
        """ Remove any gradient information
        """
        for k, v in self.variables.items():
            v.zero_grad()

    def to(self, device:torch.device) -> Workspace:
        """ Return a workspace where all tensors are on a particular device
        """
        workspace = Workspace()
        for k, v in self.variables.items():
            workspace.variables[k] = v.to(device)
        return workspace

    def _convert_to_shared_workspace(self, n_repeat=1, time_size=None):
        """ INTERNAL METHOD. It converts a workspace to a shared worspace, by repeating this workspace n times on the batch dimension
        It also automatically adapts the time_size if specified (used in NRemoteAgent.create)
        """

        with torch.no_grad():
            workspace = Workspace()
            for k, v in self.variables.items():
                value = v.get_full(None).detach()
                if not time_size is None:
                    s = value.size()
                    value = torch.zeros(
                        time_size, *s[1:], dtype=value.dtype, device=value.device
                    )
                ts = [value for t in range(n_repeat)]
                value = torch.cat(ts, dim=1)
                workspace.variables[k] = CompactSharedTensor(value)
                workspace.is_shared = True
        return workspace

    def subtime(self, from_t:int, to_t:int) -> Workspace:
        """
        Return a workspace restricted to a subset of the time dimension
        """
        assert (
            self._all_variables_same_time_size()
        ), "All variables must have the same time size"
        workspace = Workspace()
        for k, v in self.variables.items():
            workspace.variables[k] = v.subtime(from_t, to_t)
        return workspace

    def remove_variable(self, var_name:str):
        """ Remove a variable from the Workspace
        """
        del self.variables[var_name]

    def __str__(self):
        r = ["Workspace:"]
        for k, v in self.variables.items():
            r.append(
                "\t"
                + k
                + ": time_size = "
                + str(v.time_size())
                + ", batch_size = "
                + str(v.batch_size())
            )
        return "\n".join(r)

    def select_subtime(self, t: torch.LongTensor, window_size:int) -> Workspace:
        """
        `t` is a tensor of size `batch_size` that provides one time index for each element of the workspace.
        Then the function returns a new workspace by aggregating `window_size` timesteps starting from index `t`
        This methods allows to sample multiple windows in the Workspace.
        Note that the function may be quite slow.
        """
        _vars = {k: v.get_full(batch_dims=None) for k, v in self.variables.items()}
        workspace = Workspace()
        for k, v in _vars.items():
            workspace.set_full(
                k, take_per_row_strided(v, t, num_elem=window_size), batch_dims=None
            )
        return workspace

    #Static
    def sample_subworkspace(self,n_times,n_batch_elements,n_timesteps):
        """ Sample a workspace from the  workspace. The process is the following:
                * Let us consider that workspace batch_size is B and time_size is T
                * For n_times iterations:
                    * We sample a time window of size n_timesteps
                    * We then sample a n_batch_elements elements on the batch size
                    * =>> we obtain a worspace of size n_batch_elements x n_timesteps
                * We concatenate all the workspaces collected (over the batch diimension)

        Args:
            n_times ([type]): The number of sub workspaces to sample (and concatenate)
            n_batch_elements ([type]): <=workspace.batch_size() : the number of batch elements to sample for each sub workspace
            n_timesteps ([type]): <=workspace.time_size() : the number of tiimesteps to keep

        Returns:
            [Workspace]: The resulting workspace
        """
        B=self.batch_size()
        T=self.time_size()
        to_aggregate=[]
        for _ in range(n_times):
            assert not n_timesteps>T
            mini_workspace=self
            if n_timesteps<T:
                t=np.random.randint(T-n_timesteps)
                mini_workspace=self.subtime(t,t+n_timesteps)

            # Batch sampling
            if n_batch_elements<B:
                idx_envs=torch.randperm(B)[:n_batch_elements]
                mini_workspace=mini_workspace.select_batch(idx_envs)
            to_aggregate.append(mini_workspace)

        if len(to_aggregate)>1:
            mini_workspace=Workspace.cat_batch(to_aggregate)
        else:
            mini_workspace=to_aggregate[0]
        return mini_workspace


class _SplitSharedWorkspace:
    """ This is a view over a Workspace, restricted to particular batch dimensions. It is used when multiple agents are reading/writing in the same workspace but for specific batch dimensions (see NRemoteAgent)
    """
    def __init__(self, workspace, batch_dims):
        self.workspace = workspace
        self.batch_dims = batch_dims
        self.is_shared = self.workspace.is_shared

    def set(self, var_name, t, v):
        self.workspace.set(var_name, t, v, batch_dims=self.batch_dims)

    def get(self, var_name, t):
        return self.workspace.get(var_name, t, batch_dims=self.batch_dims)

    def keys(self):
        return self.workspace.keys()

    def get_time_truncated(self, var_name, from_time, to_time):
        assert from_time >= 0 and to_time >= 0 and to_time > from_time
        return self.workspace.get_time_truncated(
            var_name, from_time, to_time, batch_dims=self.batch_dims
        )

    def set_full(self, var_name, value):
        self.workspace.set_full(var_name, value, batch_dims=self.batch_dims)

    def get_full(self, var_name):
        return self.workspace.get_full(var_name, batch_dims=self.batch_dims)
