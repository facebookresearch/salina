#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import time

import torch
import torch.multiprocessing as mp

from salina import Agent, AgentArray
from salina.workspace import Workspace,WorkspaceArray


def f(agent, in_queue, out_queue, seed):
    out_queue.put("ok")
    running = True
    old_workspace = None
    print("Seeding remote agent with ", seed)
    agent.seed(seed)
    while running:
        command = in_queue.get()
        if command[0] == "go_new_workspace":
            _, workspace, args = command
            old_workspace=workspace
            agent(workspace, **args)
            out_queue.put("ok")
        elif command[0] == "go_reuse_workspace":
            _,_,args = command
            agent(old_workspace, **args)
            out_queue.put("ok")
        elif command[0] == "exit":
            out_queue.put("ok")
            return


class RemoteAgent(Agent):
    def __init__(self,agent,name=None):
        super().__init__(name=name)
        self.agent=agent
        self._is_running=False
        self.process=None
        self.last_workspace=None

    def forward(self, **args):
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

    def __call__(self,workspace,**args):
        assert workspace.is_shared,"You must use a shared workspace when using a Remote Agent"
        if self.process is None:
            self._create_process()
        if not workspace==self.last_workspace:
            self.i_queue.put(("go_new_workspace",workspace,args))
            self.last_workspace=workspace
            r = self.o_queue.get()
            assert r=="ok"
        else:
            self.i_queue.put(("go_reuse_workspace",workspace,args))
            r = self.o_queue.get()
            assert r=="ok"

    def _asynchronous_call(self,workspace,**args):
        self._is_running=True
        assert workspace.is_shared,"You must use a shared workspace when using a Remote Agent"
        if self.process is None:
            self._create_process()
        if not workspace==self.last_workspace:
            self.i_queue.put(("go_new_workspace",workspace,args))
            self.last_workspace=workspace
        else:
            self.i_queue.put(("go_reuse_workspace",workspace,args))

    def seed(self,_seed):
        self._seed=_seed

    def is_running(self):
        if self._is_running:
            try:
                r = self.o_queue.get(False)
                assert r=="ok"
                self._is_running = False
            except:
                pass
        return self._is_running

    def close(self):
        if self.process is None: return

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
        self.process=None

    def __del__(self):
        self.close()

class RemoteAgentArray(AgentArray):
    """A set of multiple agents"""
    def __init__(self,agents_list):
        self.agents=[RemoteAgent(a) for a in agents_list]

    def __call__(self,workspaces,**args):
        assert isinstance(workspaces,WorkspaceArray) and len(workspaces)==len(self.agents)
        for k in range(len(self.agents)):
            self.agents[k]._asynchronous_call(workspaces[k],**args)
        for a in self.agents:
            while a.is_running():
                pass


    def seed(self, seed, inc=1):
        s=seed
        for a in self.agents:
            a.seed(s)
            s+=inc
