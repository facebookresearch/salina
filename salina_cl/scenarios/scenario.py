#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from salina_cl.core import Scenario
from brax.envs import wrappers
from salina_cl.scenarios.brax.halfcheetah import Halfcheetah
from salina_cl.scenarios.brax.ant import Ant
from salina_cl.core import Task

domains = {
    "halfcheetah": Halfcheetah,
    "ant": Ant
}

def make_env(seed = 0,
            batch_size = None,
            max_episode_steps = 1000,
            action_repeat = 1,
            backend = None,
            auto_reset = True,
            domain = "halfcheetah",
            env_task = "normal",
            **kwargs):

    env = domains[domain](env_task, **kwargs)
    if max_episode_steps is not None:
        env = wrappers.EpisodeWrapper(env, max_episode_steps, action_repeat)
    if batch_size:
        env = wrappers.VectorWrapper(env, batch_size)
    if auto_reset:
        env = wrappers.AutoResetWrapper(env)
    if batch_size is None:
        return wrappers.GymWrapper(env, seed=seed, backend=backend)
    return wrappers.VectorGymWrapper(env, seed=seed, backend=backend)

class BraxScenario(Scenario):
    def __init__(self,n_train_envs,n_evaluation_envs,n_steps,domain,tasks, **kwargs):
        print("Domain:",domain)
        print("Scenario:",tasks)
        self._train_tasks=[]
        for k,task in enumerate(tasks):
            agent_cfg={
                "classname":"salina.agents.brax.AutoResetBraxAgent",
                "make_env_fn":make_env,
                "make_env_args":{
                                "domain":domain,
                                "max_episode_steps":1000,
                                "env_task":task},
                "n_envs":n_train_envs
            }
            self._train_tasks.append(Task(agent_cfg,k,n_steps))

        self._test_tasks=[]
        for k,task in enumerate(tasks):
            agent_cfg={
                "classname":"salina.agents.brax.NoAutoResetBraxAgent",
                "make_env_fn":make_env,
                "make_env_args":{"max_episode_steps":1000,
                                 "env_task":task},
                "n_envs":n_evaluation_envs
            }
            self._test_tasks.append(Task(agent_cfg,k))

    def train_tasks(self):
        return self._train_tasks

    def test_tasks(self):
        return self._test_tasks