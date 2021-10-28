#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


import collections

import d4rl
import gym
import numpy as np
import torch
from gym.wrappers import TimeLimit
from tqdm import tqdm

from salina import Workspace
from salina_examples.rl.atari_wrappers import make_atari, wrap_deepmind, wrap_pytorch


def make_d4rl_env(**env_args):
    e = gym.make(env_args["env_name"], stack=True)
    e = TimeLimit(e, max_episode_steps=env_args["max_episode_steps"])
    return e


def make_d4rl_atari_env(**env_args):
    import d4rl_atari

    e = gym.make(env_args["env_name"], stack=True)
    e = TimeLimit(e, max_episode_steps=env_args["max_episode_steps"])
    return e


def _fixed_sequence_dataset(
    env, dataset=None, max_steps=None, max_episodes=None, **kwargs
):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    # TODO: Some serious performance issues.
    # TODO: Randomize the episode selection without extracting all of them.
    # TODO: Adding discounted reward returns.

    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    total_steps = dataset["rewards"].shape[0]
    if max_steps is None:
        max_steps = total_steps

    assert (
        max_steps <= dataset["rewards"].shape[0]
    ), '"max_steps ={} " should be smaller (or equal) than total number of steps = {}.'.format(
        max_steps, total_steps
    )

    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if "timeouts" in dataset:
        use_timeouts = True

    key_list = []
    for k_index in dataset:
        if (
            isinstance(dataset[k_index], np.ndarray)
            and dataset[k_index].shape[0] == total_steps
        ):
            key_list.append(k_index)

    episode_step = 0
    for i in range(max_steps):
        done_bool = bool(dataset["terminals"][i])
        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
        else:
            final_timestep = episode_step == env._max_episode_steps - 1

        for k in key_list:
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            yield episode_data
            data_ = collections.defaultdict(list)
            if max_episodes:
                max_episodes -= 1
                if max_episodes < 1:
                    break

        episode_step += 1

    if max_episodes is not None and max_episodes > 0:
        import warnings

        warnings.warn(
            "[WARNING] Not enough steps in the dataset to generate the requested number of episodes"
        )


def d4rl_transition_buffer(d4rl_env, time_size=2, padding=1):
    # Import the dataset associated with a d4rl environment to a Workspace as transitions (including transitions from the final state of one episode to the initial state of a new episode)
    dataset = d4rl_env.get_dataset()
    print("[d4rl_transition_buffer] Loading transitions")
    if isinstance(dataset["observations"], list):
        obs = np.array(dataset["observations"])
    else:
        obs = dataset["observations"]
    T = obs.shape[0]
    # T = int(T * proportion)
    print("\tUsing ", T, " transitions...")
    _obs = torch.tensor(obs[:T]).float()
    if "timeouts" in dataset:
        _done = (
            torch.tensor(dataset["terminals"][:T])
            + torch.tensor(dataset["timeouts"][:T])
        ).bool()
    else:
        _done = torch.tensor(dataset["terminals"][:T]).bool()

    _initial_state = torch.zeros_like(_done).bool()
    _initial_state[1:] = _done[:-1]
    _initial_state[0] = True
    _action = torch.tensor(dataset["actions"][:T])
    _action = torch.tensor(dataset["actions"][:T])
    _reward = torch.tensor(dataset["rewards"][:T]).float()
    _reward[_initial_state] = 0.0
    _reward = torch.cat([torch.tensor([0.0]), _reward], dim=0)[:-1]

    _timestep = torch.zeros_like(_reward).long()
    _creward = torch.zeros_like(_reward).float()
    cr = 0.0
    ts = 0
    for t in range(T):
        if _initial_state[t]:
            cr = 0.0
            ts = 0
        _timestep[t] = ts
        cr = cr + _reward[t]
        _creward[t] = cr
        ts += 1

    buffer_size = T - time_size - 1

    _obs = _obs.unfold(0, time_size, padding)
    _done = _done.unfold(0, time_size, padding)
    _timestep = _timestep.unfold(0, time_size, padding)
    _creward = _creward.unfold(0, time_size, padding)
    _action = _action.unfold(0, time_size, padding)
    _reward = _reward.unfold(0, time_size, padding)
    _initial_state = _initial_state.unfold(0, time_size, padding)

    arange = torch.arange(len(_obs.size())).tolist()
    _obs = _obs.permute(-1, 0, *arange[1:-1])
    print(_obs.size())
    arange = torch.arange(len(_done.size())).tolist()
    _done = _done.permute(-1, 0, *arange[1:-1])
    arange = torch.arange(len(_timestep.size())).tolist()
    _timestep = _timestep.permute(-1, 0, *arange[1:-1])
    arange = torch.arange(len(_creward.size())).tolist()
    _creward = _creward.permute(-1, 0, *arange[1:-1])
    arange = torch.arange(len(_initial_state.size())).tolist()
    _initial_state = _initial_state.permute(-1, 0, *arange[1:-1])
    arange = torch.arange(len(_reward.size())).tolist()
    _reward = _reward.permute(-1, 0, *arange[1:-1])
    arange = torch.arange(len(_action.size())).tolist()
    _action = _action.permute(-1, 0, *arange[1:-1])
    if _action.dtype == torch.int32:
        _action = _action.long()

    workspace = Workspace()
    workspace.set_full("env/env_obs", _obs)
    workspace.set_full("env/done", _done)
    workspace.set_full("action", _action)
    workspace.set_full("env/reward", _reward)
    workspace.set_full("env/initial_state", _initial_state)
    workspace.set_full("env/timestep", _timestep)
    workspace.set_full("env/cumulated_reward", _creward)
    print("\t Resulting workspace:")
    for k in workspace.keys():
        print("\t\t", k, " => ", workspace[k].size(), " (", workspace[k].dtype, ")")
    return workspace


def d4rl_episode_buffer(d4rl_env):
    # Import the dataset associated with a d4rl environment to a Workspace as full episodes (of different length)
    print("[d4rl_episode_buffer] Reading dataset")
    sequence_dataset = _fixed_sequence_dataset(d4rl_env)

    episodes = []
    current_episode = []
    cumulated_reward = 0.0

    episodes = []
    for s in sequence_dataset:
        episode = {k: torch.tensor(v).unsqueeze(1) for k, v in s.items()}
        nepisode = {}
        for k, v in episode.items():
            if k.endswith("s"):
                nepisode["env/" + k[:-1]] = v
            else:
                nepisode["env/" + k] = v
        if "env/timeout" in nepisode:
            nepisode["env/done"] = (
                nepisode["env/terminal"] + nepisode["env/timeout"]
            ).bool()
        else:
            nepisode["env/done"] = (nepisode["env/terminal"]).bool()

        if "env/observation" in nepisode:
            nepisode["env/env_obs"] = nepisode.pop("env/observation")
        nepisode["env/done"][-1] = True
        nepisode["env/initial_state"] = nepisode["env/done"].clone()
        nepisode["env/initial_state"].fill_(False)
        nepisode["env/initial_state"][0] = True
        nepisode["action"] = nepisode.pop("env/action")
        nepisode["env/timestep"] = torch.arange(
            nepisode["env/done"].size()[0]
        ).unsqueeze(1)
        nepisode["env/reward"][1:] = nepisode["env/reward"][:-1].clone()
        nepisode["env/reward"][0] = 0.0
        nepisode["env/cumulated_reward"] = torch.zeros(
            nepisode["env/done"].size()[0], 1
        )
        cr = 0.0
        for t in range(nepisode["env/done"].size()[0]):
            cr += nepisode["env/reward"][t].item()
            nepisode["env/cumulated_reward"][t] = cr
        episodes.append(nepisode)

    max_length = max([e["env/reward"].size()[0] for e in episodes])
    print("\t max episode length = ", max_length)
    print("\t n episodes = ", len(episodes))

    n_skip=0
    for e in episodes:
        l=e["env/reward"].size()[0]
        if l==0:
            n_skip+=1
            continue
        for k,v in e.items():
            ts=v.size()[0]
            if ts<max_length:
                v.resize_(max_length,*(v.size()[1:]))
                v[ts:]=0
    print("\tSkip ",n_skip," trajectories of size = 0")
    workspace=Workspace()
    f_episode={}
    for k in episodes[0]:
        vals=[e[k] for e in episodes]
        workspace.set_full(k,torch.cat(vals,dim=1))

    return workspace
