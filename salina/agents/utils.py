#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

from salina import Agent, TAgent


class Agents(Agent):
    """An agent composed of multiple agents. The agents are executed in sequence"""

    def __init__(self, *agents, name=None):
        super().__init__(name=name)
        for a in agents:
            assert isinstance(a, Agent)
        self.agents = nn.ModuleList(agents)

    def __call__(self, workspace, **args):
        for a in self.agents:
            a(workspace, **args)

    def forward(**args):
        raise NotImplementedError

    def seed(self, seed):
        for a in self.agents:
            a.seed(seed)

    def __getitem__(self, k):
        return self.agents[k]

    def get_by_name(self, n):
        r = []
        for a in self.agents:
            r = r + a.get_by_name(n)
        if n == self._name:
            r = r + [self]
        return r


class TemporalAgent(Agent):
    """Execute an agent over multiple time steps
    the stop_variable (if any) is used to force the stop of the agent when all the values are equal to True at the particular timestep
    """

    def __init__(self, agent, name=None):
        super().__init__(name=name)
        self.agent = agent

    def __call__(self, workspace, t=0, n_steps=None, stop_variable=None, **args):
        """
        :param t: The start timestep
        :type t: int, optional
        :param n_steps: number of timesteps to execute the agent on (None means 'until the end of the workspace')
        :type n_steps: [type], optional
        """
        assert not (n_steps is None and stop_variable is None)
        _t = t
        while True:
            self.agent(workspace, t=_t, **args)
            if not stop_variable is None:
                s = workspace.get(stop_variable, _t)
                if s.all():
                    break
            _t += 1
            if not n_steps is None:
                if _t >= t + n_steps:
                    break

    def forward(self, **args):
        raise NotImplementedError

    def seed(self, seed):
        self.agent.seed(seed)

    def get_by_name(self, n):
        r = self.agent.get_by_name(n)
        if n == self._name:
            r = r + [self]
        return r


class CopyTAgent(TAgent):
    """a TAgent that copies one variable to another. The variable can be copied with or without gradient."""

    def __init__(self, input_name, output_name, detach=False, name=None):
        super().__init__(name=name)
        self.input_name = input_name
        self.output_name = output_name
        self.detach = detach

    def forward(self, t, **args):
        x = self.get((self.input_name, t))
        if not self.detach:
            self.set((self.output_name, t), x)
        else:
            self.set((self.output_name, t), x.detach())


class IfTAgent(TAgent):
    """A 'If' Agent"""

    def __init__(
        self,
        input_true,
        input_false,
        output_name,
        detach=False,
        condition_name=None,
        name=None,
    ):
        super().__init__(name=name)
        self.input_true = input_true
        self.input_false = input_false
        self.output_name = output_name
        self.detach = detach
        self.condition_name = condition_name

    def forward(self, t, switches, **args):
        s = switches[self.condition_name]
        x = None
        if s:
            x = self.get((self.input_true, t))
        else:
            x = self.get((self.input_false, t))

        if not self.detach:
            self.set((self.output_name, t), x)
        else:
            self.set((self.output_name, t), x.detach())


class PrintAgent(Agent):
    """A TAgent that print variables values to console"""

    def __init__(self, *names, name=None):
        super().__init__(name=name)
        self.names = names

    def forward(self, t, **args):
        for n in self.names:
            print(n, " = ", self.get((n, t)))
