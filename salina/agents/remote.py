#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import time

import torch
import torch.multiprocessing as mp

from salina import Agent
from salina.workspace import SharedSubWorkspace, SharedWorkspace, Workspace


def f(agent, in_queue, out_queue, seed):
    out_queue.put("ok")
    running = True
    old_workspace = None
    print("Seeding remote agent with ", seed)
    agent.seed(seed)
    while running:
        command = in_queue.get()
        if command[0] == "go":
            _, workspace, start_position, batch_size, args = command
            if workspace is None:
                sub_workspace = old_workspace
            else:
                sub_workspace = SharedSubWorkspace(
                    workspace, start_position, batch_size
                )
                old_workspace = sub_workspace
            agent(sub_workspace, **args)
            out_queue.put("ok")
        elif command[0] == "exit":
            out_queue.put("ok")
            return


class RemoteAgent(Agent):
    """A agent that executes an agent over multiple threads
    :param Agent: [description]
    :type Agent: [type]
    """

    def __init__(self, agent, num_processes):
        super().__init__()
        self.num_processes = num_processes
        self.agent = agent
        self.agent.share_memory()
        self.seeds = []
        self.last_workspace = None
        if self.num_processes > 0:
            self.processes = []
            self.running = [False for k in range(num_processes)]
            self.num_processes = num_processes
            self.workspace = None

    def _forward(self, k, renv, start_position, batch_size, args):
        self.processes[k][1].put(("go", renv, start_position, batch_size, args))

    def seed(self, seed, _inc_seed=64):
        self.agent.seed(seed)
        self.seeds = [seed + _inc_seed * k for k in range(self.num_processes)]

    def forward(self, **args):
        raise NotImplementedError

    def asynchronous_forward_(self, workspace, **args):
        if self.num_processes == 0:
            print(
                "[RemoteAgent] call with asynchronuous forward and num_processes==0: Executing in sychronous mode"
            )
            return self.__call__(workspace, **args)

        if not isinstance(workspace, SharedWorkspace):
            return self._first_forward(workspace, **args)

        if self.num_processes > 0:
            assert isinstance(
                workspace, SharedWorkspace
            ), "[RemoteAgent.asynchronous_forward_] Must be on a SharedWorkspace"

            assert not self.is_running()
            assert workspace.batch_size() % self.num_processes == 0
            bs = int(workspace.batch_size() / self.num_processes)
            sp = [bs * k for k in range(self.num_processes)]
            self.running = [False for k in range(self.num_processes)]

            if workspace == self.last_workspace:
                for k in range(self.num_processes):
                    self._forward(k, None, sp[k], bs, args)
                    self.running[k] = True
            else:
                self.last_workspace = workspace
                for k in range(self.num_processes):
                    self._forward(k, workspace, sp[k], bs, args)
                    self.running[k] = True

            return workspace
        else:
            assert (
                False
            ), "[RemoteAgent] asynchronous_forward_ can be sed only if num_processes>0"

    def _first_forward(self, workspace, **args):
        with torch.no_grad():
            for k in range(self.num_processes):
                print("Creating process ", k, " over ", self.num_processes)
                i_queue = mp.Queue()
                o_queue = mp.Queue()
                i_queue.cancel_join_thread()
                o_queue.cancel_join_thread()
                p = mp.Process(
                    target=f, args=(self.agent, i_queue, o_queue, self.seeds[k])
                )
                p.daemon = False
                p.start()
                r = o_queue.get()
                self.processes.append((p, i_queue, o_queue))

            assert isinstance(workspace, Workspace)
            print("[RemoteAgent] First call in main process for lazy initialization")
            workspace = self.agent(workspace, **args)
            self.workspace = SharedWorkspace(workspace)
            assert (
                len(self.seeds) == self.num_processes
            ), "[RemoteAgent] Seeding needed before calling the agent"

        return self.workspace

    def __call__(self, workspace, **args):
        assert not workspace is None
        if self.num_processes == 0:
            return self.agent(workspace, **args)

        if self.num_processes > 0:
            assert not self.is_running()
            if not isinstance(workspace, SharedWorkspace):
                return self._first_forward(workspace, **args)

            assert workspace.batch_size() % self.num_processes == 0
            bs = int(workspace.batch_size() / self.num_processes)
            sp = [bs * k for k in range(self.num_processes)]
            self.running = [False for k in range(self.num_processes)]

            if workspace == self.last_workspace:
                for k in range(self.num_processes):
                    self._forward(k, None, sp[k], bs, args)
            else:
                self.last_workspace = workspace
                for k in range(self.num_processes):
                    self._forward(k, workspace, sp[k], bs, args)

            for k in range(self.num_processes):
                r = self.processes[k][2].get()
                assert (
                    r == "ok"
                ), "[RemoteAgent.forward] Invalid answer from the process"
            return workspace

    def is_running(self):
        if self.num_processes > 0:
            for k in range(self.num_processes):
                if self.running[k]:
                    try:
                        r = self.processes[k][2].get(False)
                        self.running[k] = False
                    except:
                        pass

            for r in self.running:
                if r:
                    return True
            return False
        else:
            return False

    def close(self):
        print("[Remoteagent] closing processes")
        if self.processes is None: return
        if self.num_processes > 0:
            for p, inq, outq in self.processes:
                inq.put(("exit",))
                outq.get()
                time.sleep(0.1)
                p.terminate()
                p.join()
                inq.close()
                outq.close()
                time.sleep(0.1)
                del inq
                del outq
        self.processes=None

    def __del__(self):
        self.close()
