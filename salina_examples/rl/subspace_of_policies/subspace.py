#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class Linear(nn.Module):
    def __init__(self, n_anchors, in_channels, out_channels, bias = True, same_init = False):
        super().__init__()
        self.n_anchors = n_anchors
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_bias = bias

        if same_init:
            anchor = nn.Linear(in_channels,out_channels,bias=self.is_bias)
            anchors = [copy.deepcopy(anchor) for _ in range(n_anchors)]
        else:
            anchors = [nn.Linear(in_channels,out_channels,bias=self.is_bias) for _ in range(n_anchors)]
        self.anchors = nn.ModuleList(anchors)

    def forward(self, x, alpha):
        xs = [anchor(x) for anchor in self.anchors]
        xs = torch.stack(xs,dim=-1)

        alpha = torch.stack([alpha] * self.out_channels, dim=-2)
        xs = (xs * alpha).sum(-1)
        return xs

class Sequential(nn.Sequential):
    def __init__(self,*args):
        super().__init__(*args)

    def forward(self, input, t):
        for module in self:
            input = module(input,t) if isinstance(module,Linear) else module(input)
        return input

