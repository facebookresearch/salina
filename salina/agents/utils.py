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

    def __init__(self, *agents):
        super().__init__()
        for a in agents:
            assert isinstance(a, Agent)
        self.agents = nn.ModuleList(agents)

    def __call__(self, workspace, **args):
        for a in self.agents:
            workspace = a(workspace, **args)
        return workspace

    def forward(**args):
        raise NotImplementedError

    def seed(self, seed):
        for a in self.agents:
            a.seed(seed)


class TemporalAgent(Agent):
    """Execute an agent over multiple time steps
    the stop_variable (if any) is used to force the stop of the agent when all the values are equal to True at the particular timestep
    """

    def __init__(self, agent, stop_variable=None):
        super().__init__()
        self.agent = agent
        self.stop_variable = stop_variable

    def __call__(self, workspace, t=0, n_steps=None, **args):
        """
        :param t: The start timestep
        :type t: int, optional
        :param n_steps: number of timesteps to execute the agent on (None means 'until the end of the workspace')
        :type n_steps: [type], optional
        """
        if n_steps is None:
            n_steps = workspace.time_size() - t
        for _t in range(t, t + n_steps):
            workspace = self.agent(workspace, t=_t, **args)
            if not self.stop_variable is None:
                s = workspace.get(self.stop_variable, _t)
                if s.all():
                    break
        return workspace

    def forward(self, **args):
        raise NotImplementedError

    def seed(self, seed):
        self.agent.seed(seed)

class CopyTAgent(TAgent):
    """a TAgent that copies one variable to another. The variable can be copied with or without gradient."""

    def __init__(self, input_name, output_name, detach=False):
        super().__init__()
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
        self, input_true, input_false, output_name, detach=False, condition_name=None
    ):
        super().__init__()
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

    def __init__(self, *names):
        super().__init__()
        self.names = names

    def forward(self, t, **args):
        for n in self.names:
            print(n, " = ", self.get((n, t)))
