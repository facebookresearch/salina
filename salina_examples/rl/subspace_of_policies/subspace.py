#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, n_models, in_channels, out_channels, bias = True, same_init = False):
        super().__init__()
        self.n_models = n_models
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_bias = bias

        if same_init:
            weight = torch.nn.init.xavier_uniform_(torch.zeros((1,out_channels,in_channels)))
            weight = weight.repeat(n_models,1,1)
        else:
            weight = torch.nn.init.xavier_uniform_(torch.zeros((n_models,out_channels,in_channels)))
        self.weight = nn.Parameter(weight)

        if self.is_bias:
            if same_init:
                bias = torch.nn.init.xavier_uniform_(torch.zeros((1,out_channels)))
                bias = bias.repeat(n_models,1)
            else:
                bias = torch.nn.init.xavier_uniform_(torch.zeros((n_models,out_channels)))
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter('bias_n', None)

    def forward(self, x, t):
        xs=[F.linear(x, self.weight[k], self.bias[k]) for k in range(self.n_models)]
        xs=torch.stack(xs,dim=-1)
        t = torch.stack([t] * self.out_channels,dim=-2)
        xs = xs * t
        xs = xs.sum(-1)
        return xs

class Sequential(nn.Sequential):
    def __init__(self,*args):
        super().__init__(*args)

    def forward(self, input, t):
        for module in self:
            input = module(input,t) if isinstance(module,Linear) else module(input)
        return input

