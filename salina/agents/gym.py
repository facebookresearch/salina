#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch

from salina import TAgent


def _format_frame(frame):
    if isinstance(frame, dict):
        r = {}
        for k in frame:
            r[k] = _format_frame(frame[k])
        return r
    elif isinstance(frame, list):
        t = torch.tensor(frame).unsqueeze(0)
        if t.dtype != torch.float32:
            t = t.float()
        return t
    elif isinstance(frame, np.ndarray):
        t = torch.from_numpy(frame).unsqueeze(0)
        if t.dtype != torch.float32:
            t = t.float()
        return t
    elif isinstance(frame, torch.Tensor):
        return frame.unsqueeze(0)  # .float()
    elif isinstance(frame, bool):
        return torch.tensor([frame]).bool()
    elif isinstance(frame, int):
        return torch.tensor([frame]).long()
    elif isinstance(frame, float):
        return torch.tensor([frame]).float()

    else:
        try:
            # Check if its a LazyFrame from OpenAI Baselines
            o = torch.from_numpy(frame.__array__()).unsqueeze(0).float()
            return o
        except:
            assert False


def _torch_cat_dict(d):
    r = {}
    for k in d[0]:
        r[k] = torch.cat([dd[k] for dd in d], dim=0)
    return r


class GymAgent(TAgent):
    def __init__(
        self, make_env_fn=None, make_env_args={}, input="action", output="env/"
    ):
        super().__init__()
        self.envs = None
        self.env_args = make_env_args
        self._seed = 0
        self.n_envs = None
        self.output = output
        self.input = input
        self.make_env_fn = make_env_fn

    def _initialize_envs(self, n):
        assert self._seed is not None, "[GymAgent] seeds must be specified"
        self.envs = [self.make_env_fn(**self.env_args) for k in range(n)]
        for k in range(n):
            self.envs[k].seed(self._seed + k)
        self.n_envs = n
        self.timestep = 0
        self.finished = torch.tensor([True for e in self.envs])
        self.timestep = torch.tensor([0 for e in self.envs])
        self.last_frame = {}
        self.cumulated_reward = {}

    def _reset(self, k, save_render):
        env = self.envs[k]
        self.cumulated_reward[k] = 0.0
        o = env.reset()
        self.cumulated_reward[k] = 0.0
        observation = _format_frame(o)
        if isinstance(observation, torch.Tensor):
            observation = {"env_obs": observation}
        else:
            assert isinstance(observation, dict)
        if save_render:
            image = env.render(mode="image").unsqueeze(0)
            observation["rendering"] = image

        self.last_frame[k] = observation
        done = torch.tensor([False])
        initial_state = torch.tensor([True])
        self.finished[k] = False
        finished = torch.tensor([False])
        reward = torch.tensor([0.0]).float()
        self.timestep[k] = 0
        timestep = torch.tensor([self.timestep[k]])
        ret = {
            **observation,
            "done": done,
            "initial_state": initial_state,
            "reward": reward,
            "timestep": timestep,
            "cumulated_reward": torch.tensor([self.cumulated_reward[k]]).float(),
        }
        return ret

    def _step(self, k, action, save_render):
        if self.finished[k]:
            assert k in self.last_frame
            return {
                **self.last_frame[k],
                "done": torch.tensor([True]),
                "initial_state": torch.tensor([False]),
                "reward": torch.tensor([0.0]).float(),
                "cumulated_reward": torch.tensor([self.cumulated_reward[k]]),
                "timestep": torch.tensor([-1]),
                "cumulated_reward": torch.tensor([self.cumulated_reward[k]]),
            }
        self.timestep[k] += 1
        env = self.envs[k]
        if len(action.size()) == 0:
            action = action.item()
            assert isinstance(action, int)
        else:
            action = np.array(action.tolist())

        o, r, d, _ = env.step(action)
        self.cumulated_reward[k] += r
        observation = _format_frame(o)
        if isinstance(observation, torch.Tensor):
            observation = {"env_obs": observation}
        else:
            assert isinstance(observation, dict)
        if save_render:
            image = env.render(mode="image").unsqueeze(0)
            observation["rendering"] = image

        self.last_frame[k] = observation
        if d:
            self.finished[k] = True
        ret = {
            **observation,
            "done": torch.tensor([d]),
            "initial_state": torch.tensor([False]),
            "reward": torch.tensor([r]).float(),
            "cumulated_reward": torch.tensor([self.cumulated_reward[k]]),
            "timestep": torch.tensor([self.timestep[k]]),
            "cumulated_reward": torch.tensor([self.cumulated_reward[k]]),
        }
        return ret

    def forward(self, t=0, save_render=False, **args):
        if self.envs is None:
            self._initialize_envs(self.workspace.batch_size())
        assert (
            self.n_envs == self.workspace.batch_size()
        ), "[GymEnv] cannot be used on workspace of different batch size"

        if t == 0:
            self.timestep = torch.tensor([0 for e in self.envs])
            observations = []
            for k, e in enumerate(self.envs):
                obs = self._reset(k, save_render)
                observations.append(obs)
            observations = _torch_cat_dict(observations)
            for k in observations:
                self.set(
                    (self.output + k, t), observations[k], use_workspace_device=True
                )
        else:
            assert t > 0
            action = self.get((self.input, t - 1))
            observations = []
            for k, e in enumerate(self.envs):
                obs = self._step(k, action[k], save_render)
                observations.append(obs)
            observations = _torch_cat_dict(observations)
            for k in observations:
                self.set(
                    (self.output + k, t), observations[k], use_workspace_device=True
                )

    def seed(self, seed):
        self._seed = seed
        if not self.envs is None:
            for k, e in enumerate(self.envs):
                e.seed(self._seed + k)


class AutoResetGymAgent(TAgent):
    def __init__(
        self, make_env_fn=None, make_env_args={}, input="action", output="env/"
    ):
        super().__init__()
        self.envs = None
        self.env_args = make_env_args
        self._seed = None
        self.n_envs = None
        self.output = output
        self.input = input
        self.make_env_fn = make_env_fn

    def _initialize_envs(self, n):
        assert self._seed is not None, "[GymAgent] seeds must be specified"
        self.envs = [self.make_env_fn(**self.env_args) for k in range(n)]
        for k in range(n):
            self.envs[k].seed(self._seed + k)
        self.n_envs = n
        self.timestep = 0
        self.finished = torch.tensor([True for e in self.envs])
        self.timestep = torch.tensor([0 for e in self.envs])
        self.is_running = [False for k in range(n)]
        self.cumulated_reward = {}

    def _reset(self, k, save_render):
        env = self.envs[k]
        self.cumulated_reward[k] = 0.0
        o = env.reset()
        self.cumulated_reward[k] = 0
        observation = _format_frame(o)
        if isinstance(observation, torch.Tensor):
            observation = {"env_obs": observation}
        else:
            assert isinstance(observation, dict)
        done = torch.tensor([False])
        initial_state = torch.tensor([True])
        self.finished[k] = False
        finished = torch.tensor([False])
        reward = torch.tensor([0.0]).float()
        self.timestep[k] = 0
        timestep = torch.tensor([self.timestep[k]])
        self.is_running[k] = True

        if save_render:
            image = env.render(mode="image").unsqueeze(0)
            observation["rendering"] = image

        ret = {
            **observation,
            "done": done,
            "initial_state": initial_state,
            "reward": reward,
            "timestep": timestep,
            "cumulated_reward": torch.tensor([self.cumulated_reward[k]]).float(),
        }
        return ret

    def _step(self, k, action, save_render):
        self.timestep[k] += 1
        env = self.envs[k]
        if len(action.size()) == 0:
            action = action.item()
            assert isinstance(action, int)
        else:
            action = np.array(action.tolist())

        o, r, d, _ = env.step(action)
        self.cumulated_reward[k] += r
        observation = _format_frame(o)
        if isinstance(observation, torch.Tensor):
            observation = {"env_obs": observation}
        else:
            assert isinstance(observation, dict)
        if d:
            self.is_running[k] = False

        if save_render:
            image = env.render(mode="image").unsqueeze(0)
            observation["rendering"] = image
        ret = {
            **observation,
            "done": torch.tensor([d]),
            "initial_state": torch.tensor([False]),
            "reward": torch.tensor([r]).float(),
            "timestep": torch.tensor([self.timestep[k]]),
            "cumulated_reward": torch.tensor([self.cumulated_reward[k]]).float(),
        }
        return ret

    def forward(self, t=0, save_render=False, **args):
        if self.envs is None:
            self._initialize_envs(self.workspace.batch_size())
        assert (
            self.n_envs == self.workspace.batch_size()
        ), "[GymEnv] cannot be used on workspace of different batch size"

        observations = []
        for k, env in enumerate(self.envs):
            if not self.is_running[k]:
                observations.append(self._reset(k, save_render))
            else:
                assert t > 0
                action = self.get((self.input, t - 1))
                observations.append(self._step(k, action[k], save_render))

        observations = _torch_cat_dict(observations)
        for k in observations:
            self.set((self.output + k, t), observations[k], use_workspace_device=True)

    def seed(self, seed):
        self._seed = seed
        assert (
            self.envs is None
        ), "[GymAgent.seed] Seeding only possible before running the agent"
