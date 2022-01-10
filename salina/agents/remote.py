#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import time

import torch
import torch.multiprocessing as mp

from salina import Agent
from salina.workspace import Workspace, _SplitSharedWorkspace

def f(agent, in_queue, out_queue, seed):
    """The function that is executed in a single process"""
    out_queue.put("ok")
    running = True
    old_workspace = None
    print("Seeding remote agent with ", seed)
    agent.seed(seed)
    while running:
        command = in_queue.get()
        if command[0] == "go_new_workspace":
            _, workspace, kwargs = command
            old_workspace = workspace
            agent(workspace, **kwargs)
            out_queue.put("ok")
        elif command[0] == "go_reuse_workspace":
            _, _, kwargs = command
            agent(old_workspace, **kwargs)
            out_queue.put("ok")
        elif command[0] == "exit":
            out_queue.put("ok")
            return
        elif command[0] == "eval_mode":
            agent.eval()
            out_queue.put("ok")
        elif command[0] == "train_mode":
            agent.train()
            out_queue.put("ok")


class RemoteAgent(Agent):
    """It corresponds to an agent that is executed in another process

    Args:
        Agent ([salina.Agent]): the agent ot execute in another process
    """

    def __init__(self, agent, name=None):
        super().__init__(name=name)
        self.agent = agent
        self._is_running = False
        self.process = None
        self.last_workspace = None
        self.train_mode = True

    def get_by_name(self, n):
        if self._name == n:
            return [self] + self.agent.get_by_name(n)
        else:
            return self.agent.get_by_name(n)

    def forward(self, **kwargs):
        raise NotImplementedError

    def _create_process(self):
        print("[RemoteAgent] starting process...")
        self.i_queue = mp.Queue()
        self.o_queue = mp.Queue()
        self.i_queue.cancel_join_thread()
        self.o_queue.cancel_join_thread()
        self.process = mp.Process(
            target=f, args=(self.agent, self.i_queue, self.o_queue, self._seed)
        )
        self.process.daemon = False
        self.process.start()
        r = self.o_queue.get()

    def __call__(self, workspace, **kwargs):
        with torch.no_grad():
            assert (
                workspace.is_shared
            ), "You must use a shared workspace when using a Remote Agent"
            if self.process is None:
                self._create_process()
                self.train(self.train_mode)
            if not workspace == self.last_workspace:
                self.i_queue.put(("go_new_workspace", workspace, kwargs))
                self.last_workspace = workspace
                r = self.o_queue.get()
                assert r == "ok"
            else:
                self.i_queue.put(("go_reuse_workspace", workspace, kwargs))
                r = self.o_queue.get()
                assert r == "ok"

    def _asynchronous_call(self, workspace, **kwargs):
        """Non-blocking forward. To use together with `is_running`"""
        with torch.no_grad():
            self._is_running = True
            assert (
                workspace.is_shared
            ), "You must use a shared workspace when using a Remote Agent"
            if self.process is None:
                self._create_process()
            if not workspace == self.last_workspace:
                self.i_queue.put(("go_new_workspace", workspace, kwargs))
                self.last_workspace = workspace
            else:
                self.i_queue.put(("go_reuse_workspace", workspace, kwargs))

    def train(self, f=True):
        self.train_mode = f
        if self.process is None:
            return
        if f:
            self.i_queue.put(("train_mode",))
            a = self.o_queue.get()
            assert a == "ok"
        else:
            self.eval()

    def eval(self):
        self.train_mode = False
        if self.process is None:
            return
        self.i_queue.put(("eval_mode",))
        a = self.o_queue.get()
        assert a == "ok"

    def seed(self, _seed):
        self._seed = _seed

    def _running_queue(self):
        return self.o_queue

    def is_running(self):
        if self._is_running:
            try:
                r = self.o_queue.get(False)
                assert r == "ok"
                self._is_running = False
            except:
                pass
        return self._is_running

    def close(self):
        if self.process is None:
            return

        print("[RemoteAgent] closing process")
        self.i_queue.put(("exit",))
        self.o_queue.get()
        time.sleep(0.1)
        self.process.terminate()
        self.process.join()
        self.i_queue.close()
        self.o_queue.close()
        time.sleep(0.1)
        del self.i_queue
        del self.o_queue
        self.process = None

    def __del__(self):
        self.close()


class NRemoteAgent(Agent):
    """Multiple agents executed in different processes. Use the `NRemoteAgent.create` function to create such an agent"""

    def __init__(self, agents, batch_dims):
        super().__init__()
        self.agents = agents
        self.batch_dims = batch_dims

    def get_by_name(self, name):
        r = []
        if self._name == name:
            r = [self]
        for a in self.agents:
            r = r + a.get_by_name(name)
        return r

    @staticmethod
    def create(agent, num_processes=0, time_size=None, **extra_kwargs):
        """Returns a NRemote agent with num_processes copies of agent in different processes
        Also returns the specific workspace to use with such an agent

        Args:
            agent ([salina.Agent]): The agent to execute in multiple processes
            num_processes (int, optional): Number of processes to create. If 0, then no processes are created (for debugging). Defaults to 0.
            time_size ([type], optional): If specified, it forces the created Workspace to have this particular time_size. Defaults to None.

        Returns:
            [salina.Agent,salina.SharedWorkspace]: The NRemoteAgent and the corresponding workspace
        """
        agent.seed(0)
        if num_processes == 0:
            workspace = Workspace()
            _agent = copy.deepcopy(agent)
            agent(workspace, **extra_kwargs)
            shared_workspace = workspace._convert_to_shared_workspace(
                n_repeat=1, time_size=time_size
            )
            return _agent, shared_workspace

        workspace = Workspace()
        agents = [copy.deepcopy(agent) for t in range(num_processes)]
        agent(workspace, **extra_kwargs)
        b = workspace.batch_size()
        batch_dims = [(k * b, k * b + b) for k, a in enumerate(agents)]
        shared_workspace = workspace._convert_to_shared_workspace(
            n_repeat=num_processes, time_size=time_size
        )
        agents = [RemoteAgent(a) for a in agents]
        return NRemoteAgent(agents, batch_dims), shared_workspace

    def __call__(self, workspace, **kwargs):
        assert workspace.is_shared
        for k in range(len(self.agents)):
            _workspace = _SplitSharedWorkspace(workspace, self.batch_dims[k])
            self.agents[k]._asynchronous_call(_workspace, **kwargs)
        for a in self.agents:
            ok = a._running_queue().get()
            assert ok == "ok"

    def seed(self, seed, inc=1):
        s = seed
        for a in self.agents:
            a.seed(s)
            s += inc

    def _asynchronous_call(self, workspace, **kwargs):
        assert workspace.is_shared
        for k in range(len(self.agents)):
            _workspace = _SplitSharedWorkspace(workspace, self.batch_dims[k])
            self.agents[k]._asynchronous_call(_workspace, **kwargs)

    def is_running(self):
        for a in self.agents:
            if a.is_running():
                return True
        return False

    def train(self, f=True):
        for a in self.agents:
            a.train(f)

    def eval(self):
        for a in self.agents:
            a.eval()

    def close(self):
        for a in self.agents:
            a.close()
