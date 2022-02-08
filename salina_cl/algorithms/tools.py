#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

def compute_time_unit(device):
    """ Compute a time unit as the time in seconds used to do a given number of forward bacward over random data
    This time unit aims at normalizing the learning time over different computers
    """    
    x=torch.randn(512,28*28).to(device)
    y=torch.randint(low=0,high=10,size=(512,)).to(device)
    m=nn.Sequential(nn.Linear(28*28,10))
    m.to(device)
    optimizer=torch.optim.Adam(m.parameters(),lr=0.001)
    _st=time.time()
    for k in range(2000):
            optimizer.zero_grad()
            py=m(x)
            loss=F.cross_entropy(py,y)
            loss.backward()
            optimizer.step()
    _et=time.time()
    ref_time=(_et-_st)
    return ref_time

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def _state_dict(agent, device):
    sd = agent.state_dict()
    for k, v in sd.items():
        sd[k] = v.to(device)
    return sd

def clip_grad(parameters, grad):
    return (torch.nn.utils.clip_grad_norm_(parameters, grad) if grad > 0 else torch.Tensor([0.0]))