#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import gym
import metaworld
import numpy as np
import random
from gym.wrappers import TimeLimit
from typing import List


TASK_SEQS = {
    "CW10": [
        "hammer-v1",
        "push-wall-v1",
        "faucet-close-v1",
        "push-back-v1",
        "stick-pull-v1",
        "handle-press-side-v1",
        "push-v1",
        "shelf-place-v1",
        "window-close-v1",
        "peg-unplug-side-v1",
    ],
    "T1": [
        "push-v1",
        "window-close-v1",
        "hammer-v1",
    ],
    "T2": [
        "hammer-v1",
        "window-close-v1",
        "faucet-close-v1",
    ],
    "T3": [
        "stick-pull-v1",
        "push-back-v1",
        "push-wall-v1",
    ],
    "T4": [
        "push-wall-v1",
        "shelf-place-v1",
        "push-back-v1",
    ],
    "T5": [
        "faucet-close-v1",
        "shelf-place-v1",
        "push-back-v1",
    ],
    "T6": [
        "stick-pull-v1",
        "peg-unplug-side-v1",
        "stick-pull-v1",
    ],
    "T7": [
        "window-close-v1",
        "handle-press-side-v1",
        "peg-unplug-side-v1",
    ],
    "T8": [
        "faucet-close-v1",
        "shelf-place-v1",
        "peg-unplug-side-v1",
    ],
}
TASK_SEQS["CW20"] = TASK_SEQS["CW10"] + TASK_SEQS["CW10"]
META_WORLD_TIME_HORIZON = 200
def get_mt50() -> metaworld.MT50:
    saved_random_state = np.random.get_state()
    np.random.seed(1)
    MT50 = metaworld.MT50()
    np.random.set_state(saved_random_state)
    return MT50

MT50 = get_mt50()
def get_subtasks(name: str) -> List[metaworld.Task]:
    return [s for s in MT50.train_tasks if s.env_name == name]

class RandomizationWrapper(gym.Wrapper):
    """Manages randomization settings in MetaWorld environments."""

    ALLOWED_KINDS = [
        "deterministic",
        "random_init_all",
        "random_init_fixed20",
        "random_init_small_box",
    ]

    def __init__(self, env: gym.Env, subtasks: List[metaworld.Task], kind: str) -> None:
        assert kind in RandomizationWrapper.ALLOWED_KINDS
        super().__init__(env)
        self.subtasks = subtasks
        self.kind = kind

        env.set_task(subtasks[0])
        if kind == "random_init_all":
            env._freeze_rand_vec = False

        if kind == "random_init_fixed20":
            assert len(subtasks) >= 20

        if kind == "random_init_small_box":
            diff = env._random_reset_space.high - env._random_reset_space.low
            self.reset_space_low = env._random_reset_space.low + 0.45 * diff
            self.reset_space_high = env._random_reset_space.low + 0.55 * diff

    def reset(self, **kwargs) -> np.ndarray:
        if self.kind == "random_init_fixed20":
            self.env.set_task(self.subtasks[random.randint(0, 19)])
        elif self.kind == "random_init_small_box":
            rand_vec = np.random.uniform(
                self.reset_space_low, self.reset_space_high, size=self.reset_space_low.size
            )
            self.env._last_rand_vec = rand_vec

        return self.env.reset(**kwargs)

class MetaWorldWrapper(gym.Wrapper):
    def __init__(self,e):
        super().__init__(e)
        # self.env=e

    def reset(self):
        return {"env_obs":self.env.reset(),"success":0.0}

    def step(self,a):
        o,r,d,i=self.env.step(a)
        o={"env_obs":o,"success":i["success"],"goalDist":i["goalDist"]}
        return o,r,d,i