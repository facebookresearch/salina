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

def f(agent,in_queue,out_queue):
    while True:
        args=in_queue.get()
        if args=="exit":
            out_queue.put("ok")
            return
        workspace=Workspace()
        with torch.no_grad():
            agent(workspace,**args)
        out_queue.put("ok")
        for k in workspace.keys():
            out_queue.put((k,workspace.get_full(k)))
        out_queue.put("ok")

class AsynchronousAgent:
    """ Implement an agent that is executed aynchronously, and that returns its own workspace
    """
    def __init__(self,agent):
        self._is_running=False
        self.process=None
        self._workspace=None
        self.agent=agent

    def __call__(self,**kwargs):
        assert not self._is_running
        if self.process is None:
            self.o_queue = mp.Queue()
            self.o_queue.cancel_join_thread()
            self.i_queue = mp.Queue()
            self.i_queue.cancel_join_thread()
            self.process = mp.Process(
                target=f, args=(self.agent, self.i_queue,self.o_queue)
            )
            self.process.daemon = False
            self.process.start()
        self._is_running=True
        self.i_queue.put(kwargs)

    def is_running(self):
        if self._is_running:
            try:
                r = self.o_queue.get(False)
                assert r == "ok"
                self._is_running = False
                print("========")
                r = self.o_queue.get()
                workspace=Workspace()
                while(r!="ok"):
                    key,val=r
                    workspace.set_full(key,val)
                    r = self.o_queue.get()
                self._workspace=workspace
            except:
                pass
        return self._is_running

    def get_workspace(self):
        if self.is_running():
            return None
        return self._workspace

    def close(self):
        if self.process is None:
            return

        print("[AsynchronousAgent] closing process")
        self.i_queue.put("exit")
        self.o_queue.get()
        self.process.terminate()
        self.process.join()
        self.i_queue.close()
        self.o_queue.close()
        del self.i_queue
        del self.o_queue
        self.process = None

    def __del__(self):
        self.close()
