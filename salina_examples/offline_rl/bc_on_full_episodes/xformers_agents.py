#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import d4rl
import d4rl_atari
import gym
import torch
import torch.nn as nn
from gym.wrappers import TimeLimit

from salina import Agent, TAgent, instantiate_class
from salina.agents import Agents
from salina.agents.xformers_transformers import *
from salina_examples.offline_rl.bc_on_full_episodes.agents import *


def xformers_transition_transformers(encoder, transformer, decoder):
    ns=None
    if "n_steps" in transformer and transformer.n_steps>0:
        ns=transformer.n_steps


    _encoder = TransitionEncoder(**dict(encoder))
    mblock =xFormerMultiBlockAgent(
        n_layers= transformer.n_layers,
        embedding_size= encoder.embedding_size,
        n_heads= transformer.n_heads,
        max_context_length= encoder.max_episode_steps+1,
        n_steps= ns,
        prefix = "attn_",
        use_layer_norm= transformer.use_layer_norm,
    )

    internal_action_agent = ActionMLPAgentFromTransformer(
        decoder.env, decoder.n_layers, decoder.hidden_size, encoder.embedding_size
    )
    action_agent = Agents(_encoder, mblock, internal_action_agent)
    return action_agent


def observation_mlp(**kwargs):
    return ActionMLPAgentFromObservation(**kwargs)
