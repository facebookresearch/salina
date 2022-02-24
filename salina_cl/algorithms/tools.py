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
import pandas as pd
import numpy as np
import PIL.Image
from torchvision.transforms import ToTensor
pd.options.mode.chained_assignment = None
from matplotlib.patches import RegularPolygon, Rectangle
import matplotlib.pyplot as plt
import numpy as np
import io

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

def display_kshot(alphas,rewards):
    fig, ax = plt.subplots(figsize = (10,8))
    plt.axis('off')
    n_anchors = alphas.shape[1]
    radius = 0.5
    center = (0.5,0.5)

    subspace = RegularPolygon((0.5,0.5),n_anchors,radius = radius, fc=(1,1,1,0), edgecolor="black")
    anchors = subspace.get_path().vertices[:-1] * radius + center

    for i,anchor in enumerate(anchors):
        x = anchor[0] + (anchor[0]-center[0]) * 0.1
        y = anchor[1] + (anchor[1]-center[1]) * 0.1
        ax.text(x,y,"θ"+str(i+1),fontsize="x-large")

    coordinates = (alphas @ anchors).T
    ax.add_artist(subspace)
    points = ax.scatter(coordinates[0],coordinates[1],c=rewards, cmap="RdYlGn", s=50)
    ax.scatter(coordinates[0][rewards.argmax()],coordinates[1][rewards.argmax()], s=300, color="darkgreen", marker="x")
    ax.set_xlim(0.,1.)
    ax.set_ylim(0.,1.)

    cbar = fig.colorbar(points, ax=ax, pad=0.1)
    minVal = int(rewards.min().item())
    maxVal = int(rewards.max().item())
    cbar.set_ticks([minVal, maxVal])
    cbar.set_ticklabels([minVal, maxVal])

    return fig

def kshot_evolution_plot(rewards, n_samples = 10, steps = 100):
    fig, ax = plt.subplots(figsize = (12,6))
    n = rewards.shape[0]
    max_rewards = []
    ks = np.linspace(start = 1, stop = n, num=steps,dtype="int")
    for k in ks:
        max_reward = []
        for _ in range(n_samples):
            idx = torch.randperm(n)[:k]
            max_reward.append(rewards[idx].max().item())
        max_rewards.append(max_reward)
    max_rewards = np.array(max_rewards)
    mean_rewards = max_rewards.mean(1)
    std_rewards = max_rewards.std(1)
    ax.plot(ks,mean_rewards)
    ax.set_ylim(rewards.max().item()*0.95,rewards.max().item()*1.05)
    ax.set_xlim(0.,n+1)
    ax.set_ylabel("max_reward")
    ax.set_xlabel("nb shots")
    ax.set_title("k-shot evolution")
    ax.fill_between(ks, (mean_rewards-std_rewards), (mean_rewards+std_rewards), color='b', alpha=.1)
    return fig
