#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from salina_cl.core import Scenario
from brax.envs import wrappers
import random
import metaworld
from salina_cl.scenarios.metaworld.continual_world import *
from salina_cl.scenarios.brax.halfcheetah import Halfcheetah
from salina_cl.scenarios.brax.ant import Ant
from salina_cl.core import Task

brax_domains = {
    "halfcheetah": Halfcheetah,
    "ant": Ant
}

def make_brax_env(seed = 0,
            batch_size = None,
            max_episode_steps = 1000,
            action_repeat = 1,
            backend = None,
            auto_reset = True,
            domain = "halfcheetah",
            env_task = "normal",
            **kwargs):

    env = brax_domains[domain](env_task, **kwargs)
    if max_episode_steps is not None:
        env = wrappers.EpisodeWrapper(env, max_episode_steps, action_repeat)
    if batch_size:
        env = wrappers.VectorWrapper(env, batch_size)
    if auto_reset:
        env = wrappers.AutoResetWrapper(env)
    if batch_size is None:
        return wrappers.GymWrapper(env, seed=seed, backend=backend)
    return wrappers.VectorGymWrapper(env, seed=seed, backend=backend)

def make_mt(name,seed = 0):
    print("Building environment ",name)
    random.seed(seed)
    # ml1 = metaworld.ML1(name)
    # env = metaworld.MT50() #.train_classes[name]()  # Create an environment with task `pick_place`
    # task = random.choice(ml1.train_tasks)
    
    env = metaworld.MT50().train_classes[name]()
    env = RandomizationWrapper(env, get_subtasks(name), randomization = "random_init_all")
    
    
    # env.set_task(name)
    env = MetaWorldWrapper(env)
    env = TimeLimit(env,max_episode_steps = META_WORLD_TIME_HORIZON)
    # import ipdb;ipdb.set_trace()
    return env

class BraxScenario(Scenario):
    def __init__(self,n_train_envs,n_evaluation_envs,n_steps,domain,tasks, **kwargs):
        super.__init__()
        print("Domain:",domain)
        print("Scenario:",tasks)

        for k,task in enumerate(tasks):
            agent_cfg={
                "classname":"salina.agents.brax.AutoResetBraxAgent",
                "make_env_fn":make_brax_env,
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
                "make_env_fn":make_brax_env,
                "make_env_args":{"max_episode_steps":1000,
                                 "env_task":task},
                "n_envs":n_evaluation_envs
            }
            self._test_tasks.append(Task(agent_cfg,k))

class CWScenario(Scenario):
    def __init__(self,scenario,n_train_envs,n_evaluation_envs,n_steps,seed=0):
        super.__init__()
        tasks = [n for n in TASK_SEQS[scenario]]
        input_dimension = 12
        output_dimension = 4

        for k,task in enumerate(tasks):
           
            agent_cfg={
                "classname":"salina.agents.gyma.AutoResetGymAgent",
                "make_env_fn":make_mt,
                "make_env_args":{"name":task,"seed":seed},
                "n_envs":n_train_envs
            }
            self._train_tasks.append(Task(agent_cfg,input_dimension,output_dimension,k,n_steps))
            self._test_tasks.append(Task(agent_cfg.update({"n_envs":n_evaluation_envs}),input_dimension,output_dimension,k,n_steps))