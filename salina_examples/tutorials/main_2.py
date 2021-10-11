import torch

from salina import Agent, Workspace
from salina.agents import Agents, NRemoteAgent, TemporalAgent

# a_t = rand() if t==0, and b_{t-1}*1.2 elsewhere
# b_t = a_t+epsilon
# c_t = a_t+b_t


class A(Agent):
    def __init__(self):
        super().__init__(self)

    def forward(self, t, **args):
        if t == 0:
            value = torch.rand(32)
            self.set(("a", t), value)
        else:
            b = self.get(("b", t - 1))
            value = b * 1.2
            self.set(("a", t), value)


class B(Agent):
    def __init__(self):
        super().__init__(self)

    def forward(self, t, epsilon, **args):
        a = self.get(("a", t))
        b = a + epsilon
        self.set(("b", t), b)


class C(Agent):
    def __init__(self):
        super().__init__(self)

    def forward(self, t, **args):
        a = self.get(("a", t))
        b = self.get(("b", t))
        c = a + b
        self.set(("c", t), c)


class C2(Agent):
    def __init__(self):
        super().__init__(self)

    def forward(self, **args):
        a = self.get("a")
        b = self.get("b")
        c = a + b
        self.set("c", c)


if __name__ == "__main__":

    agent_A = A()
    agent_B = B()

    agent_AB = Agents(agent_A, agent_B)
    tagent_AB = TemporalAgent(agent_AB)

    remote_agent, remote_workspace = NRemoteAgent.create(
        tagent_AB, num_processes=4, t=0, n_steps=100, epsilon=-0.3
    )
    remote_agent.seed(0)
    remote_agent(remote_workspace, t=0, n_steps=100, epsilon=1.2)

    print(list(remote_workspace.keys()))
    print(remote_workspace["a"].size(), remote_workspace["b"].size())
    del remote_agent
