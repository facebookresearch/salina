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


class Workspace:
    """A workspace is a collection of tensors indexed by name and time. The first dimension of each tensor is the batch dimension"""

    def __init__(self, batch_size, time_size, device=torch.device("cpu")):
        self.variables = {}
        self._batch_size = batch_size
        self._time_size = time_size
        self._device = torch.device(device)
        self.trace = []

    def split(self):
        """Returns a list of workspace of batch_size==1"""
        ws = []
        for b in range(self._batch_size):
            w = Workspace(1, self._time_size, device=self._device)
            _v = {}
            for k, v in self.variables.items():
                if isinstance(v, torch.Tensor):
                    _v[k] = v[:, b].unsqueeze(1)
                else:
                    _v[k] = {}
                    for kk, vv in v.items():
                        _v[k][kk] = vv[b].unsqueeze(0)
            w.variables = _v
            ws.append(w)
        return ws

    def set(self, var_name, t, v):
        """Set a variable in the workspace, and erase previously existing vaariable if any
        :param var_name: the name of the variable
        :type var_name: str
        :param t: time index in 0,workspace.time_size()-1
        :type t: int
        :param v: a Bx.... torch Tensor
        :type v: torch.Tensor
        """
        assert v.device == self._device
        assert isinstance(var_name, str) and isinstance(t, int)
        assert (
            v.size()[0] == self._batch_size
        ), "[workspace.set] batch_size is not matching"
        assert t < self._time_size

        if not var_name in self.variables:
            self.variables[var_name] = {}
        if isinstance(self.variables[var_name], torch.Tensor):
            # print(" [DEBUG] Unfolding variable '"+var_name+"' for time size "+str(self.variables[var_name].size()[0]))
            assert self.variables[var_name].size()[0] == self._time_size
            self.variables[var_name] = {
                t: self.variables[var_name][t]
                for t in range(self.variables[var_name].size()[0])
            }

        self.variables[var_name][t] = v

    def clear(self):
        """Remove all variables in the workspace"""
        self.variables = {}

    def _put_in_trace(self, v):
        self.trace.append(v)
        if len(self.trace) == salina.trace_maximum_size:
            self.trace.pop(0)

    def _save_trace_graph(self, filename=None, timestep=None):
        dot = Digraph()
        for t in self.trace:
            if timestep is None or len(t[2]) == 2 and t[2][1] == timestep:
                var = str(t[2])
                dot.node(var, var)
                agent = str(t[1])
                dot.node(agent)
                if t[0] == "get":
                    dot.edge(var, agent)
                else:
                    dot.edge(agent, var)
        if not filename is None:
            f = open(filename, "wt")
            f.write(dot.source)
            f.close()

        return dot.source

    def get(self, var_name, t):
        """Get the value of a variable

        :param var_name: the name of the variable
        :type var_name: str
        :param t: time index in 0,workspace.time_size()-1
        :type t: int
        :rtype: torch.Tensor  (of size batch_size x .....)
        """
        assert isinstance(var_name, str) and isinstance(t, int)
        assert t < self._time_size
        assert var_name in self.variables and t < self._time_size, (
            "[workspace.get] variable (" + var_name + "," + str(t) + ") does not exist"
        )
        return self.variables[var_name][t]

    def _set_sequence(self, var_name, value):
        assert value.device == self._device

        assert value.size()[0] == self.time_size()
        assert value.size()[1] == self.batch_size()
        self.variables[var_name] = value

    def _get_sequence(self, var_name, length=None):
        assert var_name in self.variables, (
            "[Workspace.get_sequence] unnknown variable '" + var_name + "'"
        )
        if length is None:
            length = self._time_size
        if isinstance(self.variables[var_name], torch.Tensor):
            assert self.variables[var_name].size()[0] == self._time_size
            return self.variables[var_name][:length]

        v = next(self.variables[var_name].values().__iter__())
        empty = torch.zeros_like(v).unsqueeze(0)
        array = []
        for t in range(length):
            if t in self.variables[var_name]:
                array.append(self.variables[var_name][t].unsqueeze(0))
            else:
                array.append(empty)

        value = torch.cat(array, dim=0)
        return value

    def batch_size(self):
        return self._batch_size

    def time_size(self):
        return self._time_size

    def clone(self):
        return copy.deepcopy(self)

    def keys(self):
        return self.variables.keys()

    def to_dict(self, *keys, length=None):
        if length is None:
            length = self._time_size
        return {
            k: self._get_sequence(k, length)
            for k in self.variables
            if len(keys) == 0 or k in keys
        }

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._get_sequence(key)
        else:
            return (self._get_sequence(k) for k in key)

    def to_workspace(self):
        return self

    def convert_to_workspace(self):
        return self

    def to(self, device):
        device = torch.device(device)
        if device == self._device:
            return self
        else:
            workspace = Workspace(
                time_size=self.time_size(), batch_size=self.batch_size(), device=device
            )
            for k, v in self.variables.items():
                if isinstance(v, torch.Tensor):
                    workspace.variables[k] = v.to(device)
                else:
                    workspace.variables[k] = {}
                    for kk, vv in v.items():
                        workspace.variables[k][kk] = vv.to(device)
            return workspace

    def to_multiple_devices(self, devices):
        if len(devices) == 1:
            return [self.to(devices[0])]

        n_devices = len(devices)
        assert self.batch_size() % n_devices == 0
        bs = int(self.batch_size() / n_devices)
        pos = 0
        workspaces = []
        for nd in range(n_devices):
            workspace = Workspace(
                time_size=self.time_size(), batch_size=bs, device=devices[nd]
            )
            for k, v in self.variables.items():
                if isinstance(v, torch.Tensor):
                    workspace.variables[k] = v[:, pos : pos + bs].to(devices[nd])
                else:
                    workspace.variables[k] = {}
                    for kk, vv in v.items():
                        workspace.variables[k][kk] = vv[pos : pos + bs].to(devices[nb])
            workspaces.append(workspace)
            pos += bs
        return workspaces

    def subtime(self, from_t, to_t, clone=False):
        workspace = Workspace(self.batch_size(), to_t - from_t)
        for k, v in self.variables.items():
            if isinstance(v, torch.Tensor):
                if not clone:
                    workspace.variables[k] = v[from_t:to_t]
                else:
                    workspace.variables[k] = v[from_t:to_t].clone()
            else:
                workspace.variables[k] = {}
                if not clone:
                    for kk, vv in v.items():
                        if kk >= from_t and kk < to_t:
                            workspace.variables[k][kk] = vv
                else:
                    for kk, vv in v.items():
                        if kk >= from_t and kk < to_t:
                            workspace.variables[k][kk] = vv.clone()

        workspace._device = self.device()
        return workspace

    def copy_time(self, from_time, to_time):
        if from_time == -1:
            from_time = self._time_size - 1
        if to_time == -1:
            to_time = self._time_size - 1
        for k in self.variables:
            self.variables[k][to_time] = self.variables[k][from_time].detach()

    def copy_n_last_steps(self, n):
        ts = self.time_size()
        for tt in range(n):
            self.copy_time(
                from_time=ts - n + tt,
                to_time=tt,
            )

    def device(self):
        return self._device


class SharedWorkspace:
    def __init__(self, workspace):
        self.variables = {}
        self._batch_size = workspace.batch_size()
        self._time_size = workspace.time_size()
        print("[SharedWorkspace] Building Shared workspace")
        for k in workspace.variables:
            v = workspace._get_sequence(k, self._time_size)
            print(
                "[SharedWorkspace]\t Variable ",
                k,
                " of size ",
                v.size(),
                " type=",
                v.dtype,
            )
            self.variables[k] = v.detach().clone().share_memory_()
        self.trace = []
        self._device = workspace.device()

    def split(self):
        """Returns a list of workspace of batch_size==1"""

        ws = []
        for b in range(self._batch_size):
            w = Workspace(1, self._time_size, device=self._device)
            _v = {}
            for k, v in self.variables.items():
                _v[k] = v[:, b].unsqueeze(1)
            w.variables = _v
            ws.append(w)
        return ws

    def subtime(self, from_t, to_t, clone=False):
        workspace = Workspace(self.batch_size(), to_t - from_t)
        for k in self.variables:
            if not clone:
                workspace.variables[k] = self.variables[k][from_t:to_t]
            else:
                workspace.variables[k] = self.variables[k][from_t:to_t].clone()
        workspace._device = self.device()
        return workspace

    def _put_in_trace(self, v):
        pass

    def set(self, var_name, t, v):
        assert v.device == self._device

        assert isinstance(var_name, str) and isinstance(t, int)
        assert (
            v.size()[0] == self._batch_size
        ), "[workspace.set] batch_size is not matching"
        assert t < self._time_size
        assert var_name in self.variables, (
            "[SharedWorkspace.set] variable '" + var_name + "' is unkwnown"
        )
        self.variables[var_name][t].copy_(v.detach())

    def _set_batch_slice(self, var_name, t, v, batch_from, batch_to):
        assert v.device == self._device

        assert isinstance(var_name, str) and isinstance(t, int)
        assert (
            v.size()[0] == batch_to - batch_from
        ), "[workspace.set] batch_size is not matching"
        assert t < self._time_size
        assert var_name in self.variables, (
            "[SharedWorkspace.set] variable '" + var_name + "' is unkwnown"
        )
        self.variables[var_name][t, batch_from:batch_to].copy_(v.detach())

    def get(self, var_name, t):
        assert isinstance(var_name, str) and isinstance(t, int)
        assert t < self._time_size
        assert var_name in self.variables and t < self._time_size, (
            "[workspace.get] variable (" + var_name + "," + str(t) + ") does not exist"
        )
        return self.variables[var_name][t]

    def _get_batch_slice(self, var_name, t, batch_from, batch_to):
        assert isinstance(var_name, str) and isinstance(t, int)
        assert t < self._time_size
        assert var_name in self.variables and t < self._time_size, (
            "[workspace.get] variable (" + var_name + "," + str(t) + ") does not exist"
        )
        return self.variables[var_name][t, batch_from:batch_to]

    def _get_sequence(self, var_name, length=None):
        if length is None:
            length = self._time_size
        assert var_name in self.variables, (
            "[Workspace.get_as_tensor] unnknown variable '" + var_name + "'"
        )
        return self.variables[var_name][:length]

    def keys(self):
        return self.variables.keys()

    def batch_size(self):
        return self._batch_size

    def time_size(self):
        return self._time_size

    def clear(self):
        pass

    def to_workspace(self):
        print(
            "[SharedWorkspace] to_workspace is deprecated. use 'convert_to_workspace'"
        )
        return self.convert_to_workspace()

    def convert_to_workspace(self):
        workspace = Workspace(batch_size=self._batch_size, time_size=self._time_size)
        for k in self.variables:
            workspace.variables[k] = self.variables[k].clone()
        workspace._device = self.device()
        return workspace

    def to_dict(self, *keys, length=None):
        if length is None:
            length = self._time_size
        return {
            k: self._get_sequence(k, length)
            for k in self.variables
            if len(keys) == 0 or k in keys
        }

    def copy_time(self, from_time, to_time):
        if from_time == -1:
            from_time = self._time_size - 1
        if to_time == -1:
            to_time = self._time_size - 1
        for k in self.variables:
            self.variables[k][to_time] = self.variables[k][from_time]

    def copy_n_last_steps(self, n):
        for k in self.variables:
            self.variables[k][:n] = self.variables[k][-n:].detach()

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._get_sequence(key)
        else:
            return (self._get_sequence(k) for k in key)

    def device(self):
        return self._device


class SharedSubWorkspace:
    def __init__(self, workspace, batch_idx, batch_size):
        assert isinstance(workspace, SharedWorkspace)
        self.workspace = workspace
        self.batch_idx = batch_idx
        self._batch_size = batch_size
        self.batch_to = batch_idx + batch_size
        assert self.batch_idx + self._batch_size <= workspace.batch_size()

    def _put_in_trace(self, v):
        pass

    def get(self, var_name, t):
        return self.workspace._get_batch_slice(
            var_name, t, self.batch_idx, self.batch_to
        )

    def set(self, var_name, t, v):
        self.workspace._set_batch_slice(var_name, t, v, self.batch_idx, self.batch_to)

    def batch_size(self):
        return self._batch_size

    def time_size(self):
        return self.workspace.time_size()

    def keys(self):
        return self.workspace.keys()

    def device(self):
        return self.workspace.device()
