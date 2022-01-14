import time
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_time_unit(device):
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
