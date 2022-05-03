#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# Credits to Denis Yarats for the Squashed normal and TanhTransform 
# https://github.com/denisyarats/pytorch_sac
from urllib.parse import non_hierarchical
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd, hspmm
import torch.nn.init as init
import math
import copy

class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

class LinearSubspace(nn.Module):
    def __init__(self, n_anchors, in_channels, out_channels, bias = True, same_init = False, freeze_anchors = True):
        super().__init__()
        self.n_anchors = n_anchors
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_bias = bias
        self.freeze_anchors = freeze_anchors

        if same_init:
            anchor = nn.Linear(in_channels,out_channels,bias=self.is_bias)
            anchors = [copy.deepcopy(anchor) for _ in range(n_anchors)]
        else:
            anchors = [nn.Linear(in_channels,out_channels,bias=self.is_bias) for _ in range(n_anchors)]
        self.anchors = nn.ModuleList(anchors)

    def forward(self, x, alpha):
        #print("---anchor:",max(x.abs().max() for x in self.anchors.parameters()))
        #check = (not torch.is_grad_enabled()) and (alpha[0].max() == 1.)
        xs = [anchor(x) for anchor in self.anchors]
        #if check:
        #    copy_xs = xs
        #    argmax = alpha[0].argmax()
        xs = torch.stack(xs,dim=-1)

        alpha = torch.stack([alpha] * self.out_channels, dim=-2)
        xs = (xs * alpha).sum(-1)
        #if check:
        #    print("sanity check:",(copy_xs[argmax] - xs).sum().item())
        return xs

    def add_anchor(self,alpha = None):
        if self.freeze_anchors:
            for param in self.parameters():
                param.requires_grad = False

        # Midpoint by default
        if alpha is None:
            alpha = torch.ones((self.n_anchors,)) / self.n_anchors

        new_anchor = nn.Linear(self.in_channels,self.out_channels,bias=self.is_bias)
        new_weight = torch.stack([a * anchor.weight.data for a,anchor in zip(alpha,self.anchors)], dim = 0).sum(0)
        new_anchor.weight.data.copy_(new_weight)
        if self.is_bias:
            new_bias = torch.stack([a * anchor.bias.data for a,anchor in zip(alpha,self.anchors)], dim = 0).sum(0)
            new_anchor.bias.data.copy_(new_bias)
        self.anchors.append(new_anchor)
        self.n_anchors +=1

    def L2_norms(self):
        L2_norms = {}
        with torch.no_grad():
            for i in range(self.n_anchors):
                for j in range(i+1,self.n_anchors):
                    w1 = self.anchors[i].weight
                    w2 = self.anchors[j].weight
                    L2_norms["θ"+str(i+1)+"θ"+str(i+2)] = torch.norm(w1 - w2, p=2).item()
        return L2_norms

    def cosine_similarities(self):
        cosine_similarities = {}
        with torch.no_grad():
            for i in range(self.n_anchors):
                for j in range(i+1,self.n_anchors):
                    w1 = self.anchors[i].weight
                    w2 = self.anchors[j].weight
                    p = ((w1 * w2).sum() / max(((w1 ** 2).sum().sqrt() * (w2 ** 2).sum().sqrt()),1e-8)) ** 2
                    cosine_similarities["θ"+str(i+1)+"θ"+str(i+2)] = p.item()
        return cosine_similarities
                    





class LayerNormSubspace(nn.Module):
    def __init__(self, n_anchors, hs, same_init = False, freeze_anchors = True):
        super().__init__()
        self.n_anchors = n_anchors
        self.hs = hs
        self.freeze_anchors = freeze_anchors

        if same_init:
            anchor = nn.LayerNorm(self.hs)
            anchors = [copy.deepcopy(anchor) for _ in range(n_anchors)]
        else:
            anchors = [nn.LayerNorm(self.hs) for _ in range(n_anchors)]
        self.anchors = nn.ModuleList(anchors)

    def forward(self, x, alpha):
        xs = [anchor(x) for anchor in self.anchors]
        xs = torch.stack(xs,dim=-1)

        alpha = torch.stack([alpha] * self.hs, dim=-2)
        xs = (xs * alpha).sum(-1)
        return xs

    def add_anchor(self,alpha = None):
        if self.freeze_anchors:
            for param in self.parameters():
                param.requires_grad = False

        # Midpoint by default
        if alpha is None:
            alpha = torch.ones((self.n_anchors,)) / self.n_anchors

        new_anchor = nn.LayerNorm(self.hs)
        new_weight = torch.stack([a * anchor.weight.data for a,anchor in zip(alpha,self.anchors)], dim = 0).sum(0)
        new_anchor.weight.data.copy_(new_weight)
        new_bias = torch.stack([a * anchor.bias.data for a,anchor in zip(alpha,self.anchors)], dim = 0).sum(0)
        new_anchor.bias.data.copy_(new_bias)
        self.anchors.append(new_anchor)
        self.n_anchors +=1

class Sequential(nn.Sequential):
    def forward(self, input, alpha):
        for module in self:
            input = module(input,alpha) if (isinstance(module,LinearSubspace) or isinstance(module,LayerNormSubspace)) else module(input)
        return input