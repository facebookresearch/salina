#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import torch.nn as nn

from salina import Agent, Workspace
from salina.agents import Agents
from xformers.components.attention import ScaledDotProduct, maybe_sparsify
from xformers.components.feedforward import MLP
from xformers.components import MultiHeadDispatch, Activation
from typing import Optional, Tuple


def _layer_norm(module, x):
    if len(x.size()) == 2:
        return module(x)
    else:
        s = x.size()
        x = x.reshape(s[0] * s[1], s[2])
        x = module(x)
        return x.reshape(*s)


class Id(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class xFormerBlockAgent(Agent):
    def __init__(
        self,
        embedding_size: int,
        n_heads: int,
        max_context_length: int = 1024,
        n_steps: Optional[int] = None,
        input_name: str = "attn_in/x",
        output_name: str = "attn_out/x",
        use_layer_norm: bool = False,
    ):
        """[summary]

        Args:
            embedding_size ([type]): size of the embeddings (input and output)
            n_heads ([type]):number of heads
            max_sequence_length: the maximum sequence length which can be stored
            n_steps ([type], optional): Number of previous timesteps to consider. None = all previous timesteps
            input_name (str, optional):  Defaults to "attn_in/x".
            output_name (str, optional):  Defaults to "attn_out/x".
            use_layer_norm (bool, optional):  Defaults to False.
        """
        super().__init__()
        self.n_steps = n_steps
        self.input_name = input_name
        self.output_name = output_name
        self.embedding_size = embedding_size
        self.n_heads = n_heads

        # The sparse-aware attention mechanism
        self.multiheadattention = MultiHeadDispatch(
            seq_len=max_context_length,
            dim_model=embedding_size,
            residual_dropout=0.0,
            num_heads=n_heads,
            attention=ScaledDotProduct(dropout=0.0),
        )

        # The rest of the model, nothing fancy here
        if use_layer_norm:
            self.ln1: nn.Module = nn.LayerNorm(embedding_size)
            self.ln2: nn.Module = nn.LayerNorm(embedding_size)
        else:
            self.ln1 = Id()
            self.ln2 = Id()

        self.mlp = MLP(dim_model=embedding_size, dropout=0.0, activation=Activation.GeLU, hidden_layer_multiplier=4)

        self._cached_mask: Optional[torch.Tensor] = None
        self._cached_mask_params: Tuple[int, Optional[int]] = (-1, None)

    def _get_mask(self, context: int, steps: Optional[int], device: torch.device):
        """
        boolean mask convention:
        true means compute, and false means skip the computation
        """

        if (context, steps) == self._cached_mask_params:
            return self._cached_mask

        if steps is None or steps == 0:
            # No time span, consider all the past tokens
            attn_mask = torch.tril(torch.ones(context, context), diagonal=0).bool()
        else:
            # Time span specified, mask out the older results
            attn_mask = torch.tril(torch.ones(context, context), diagonal=0)
            attn_mask2 = torch.tril(torch.ones(context, context), diagonal=-steps)
            attn_mask = (attn_mask - attn_mask2).bool()

        attn_mask = maybe_sparsify(attn_mask)

        # Cache the generated mask
        self._cached_mask = attn_mask.to(device)
        self._cached_mask_params = (context, steps)

        return self._cached_mask

    def forward(self, t: Optional[int] = None, **_):
        """ "
        There are 4 possible cases here, given two axes:
        - t labels the point in time which is of interest.
        - n_steps labels the number of steps to consider prior to this point

        t can be None (we're interested in all the points in time), or some reference relative to current
        n_steps can be None (we're interested in all the backlog) or a given time span

        The 4 cases handled here are thus:
        - t and all the backlog
        - all t and all the backlog
        - t and a preset time scope
        - all t and a preset time scope
        """

        if t is not None:
            # In this case we have a reference in time to look into
            if self.n_steps is None or self.n_steps == 0:
                # No time span specified, use all the prior tokens
                tokens = self.get(self.input_name)[: t + 1]
            else:
                # A time span is specified, limit the lookback
                from_time = max(0, t + 1 - self.n_steps)
                to_time = t + 1
                tokens = self.get_time_truncated(self.input_name, from_time, to_time)

            ln_tokens = _layer_norm(self.ln1, tokens).transpose(1, 0)  # B x T x E
            previous_tokens = ln_tokens[:]

            keys, values = previous_tokens, previous_tokens
            queries = ln_tokens[:, -1:, :]  # B x T x E

            attn_output = self.multiheadattention(queries, keys, values)

            attn_output = attn_output.squeeze(1)
            x = tokens[-1] + attn_output  # Now  B x E
            nx = _layer_norm(self.ln2, x)
            x = x + self.mlp(nx)
            self.set((self.output_name, t), x)

        else:
            # No reference in time, consider all the results
            tokens = self.get(self.input_name)
            tokens = _layer_norm(self.ln1, tokens)
            tokens = tokens.transpose(1, 0)
            keys, values, queries = tokens, tokens, tokens

            T = queries.size()[1]

            attn_mask = self._get_mask(T, self.n_steps, tokens.device)  # n_steps x n_steps

            if not attn_mask.is_sparse:
                attn_mask = attn_mask.unsqueeze(0).expand(
                    queries.shape[0] * self.n_heads, -1, -1
                )  # (batch * heads) x n_steps x n_steps

            attn_output = self.multiheadattention(queries, keys, values, att_mask=attn_mask)
            x = tokens + attn_output

            x = x.transpose(1, 0)

            nx = _layer_norm(self.ln2, x)
            x = x + self.mlp(nx)
            self.set(self.output_name, x)


class xFormerMultiBlockAgent(Agents):
    def __init__(
        self,
        n_layers: int,
        embedding_size: int,
        n_heads: int,
        max_context_length: int = 1024,
        n_steps: Optional[int] = None,
        prefix: str = "attn_",
        use_layer_norm: bool = False,
    ):
        agents = []
        for k in range(n_layers):
            in_prefix = prefix + str(k + 1)
            out_prefix = prefix + str(k + 2)
            if k == n_layers - 1:
                out_prefix = prefix + "out"
            if k == 0:
                in_prefix = prefix + "in"
            agents.append(
                xFormerBlockAgent(
                    embedding_size=embedding_size,
                    n_heads=n_heads,
                    n_steps=n_steps,
                    max_context_length=max_context_length,
                    input_name=in_prefix + "/x",
                    output_name=out_prefix + "/x",
                    use_layer_norm=use_layer_norm,
                )
            )
        super().__init__(*agents)


if __name__ == "__main__":
    print("Check that transformers and batch transformers are computing the same output")
    a = torch.randn(5, 3, 4).cuda()  # Time x Batch x Embedding
    workspace = Workspace()
    workspace.set_full("x", a)
    agent = xFormerBlockAgent(
        embedding_size=4,
        n_heads=1,
        n_steps=2,
        input_name="x",
        output_name="y",
        use_layer_norm=False,
    ).cuda()
    for t in range(5):
        agent(workspace, t=t)
    y1 = workspace.get_full("y")
    print(y1)

    workspace = Workspace()
    workspace.set_full("x", a)
    agent(workspace)
    y2 = workspace.get_full("y")
    print(y2)
    assert ((y1 - y2) ** 2).lt(0.0000001).all(), "Problem..."
