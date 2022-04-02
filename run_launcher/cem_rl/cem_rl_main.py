import sys, os
import torch

import gym
from gym.wrappers import TimeLimit

import hydra
from omegaconf import DictConfig

from salina import instantiate_class,Workspace
from salina.agents import Agents,TemporalAgent,NRemoteAgent
from salina.agents.gyma import AutoResetGymAgent
from salina.agents.gyma import NoAutoResetGymAgent, GymAgent
from salina.logger import TFLogger
from salina.agents.asynchronous import AsynchronousAgent

sys.path.append(os.getcwd())

from algorithms.cem_rl import CemRl
HYDRA_FULL_ERROR=1

# TODO: clean remove this function with better env creation
def make_gym_env(max_episode_steps,env_name,verbose = False):
    gym_env = TimeLimit(gym.make(env_name),max_episode_steps=max_episode_steps)
    if verbose:
        print(f'for {env_name}, the low action is {gym_env.action_space.low} and hight is {gym_env.action_space.high}')
    return gym_env
    

def synchronized_train_multi(cfg):
    # init 
    cem_rl = CemRl(cfg)
    logger = instantiate_class(cfg.logger)

    n_processes=min(cfg.algorithm.num_processes,cfg.algorithm.es_algorithm.pop_size)
    pop_size = cfg.algorithm.es_algorithm.pop_size

    acquisition_agents = []
    acquisition_actors = []
    for i in range(n_processes): 
        env_agent  = AutoResetGymAgent(make_gym_env,{'max_episode_steps':cfg.env.max_episode_steps,
                                            'env_name':cfg.env.env_name},
                                            n_envs=cfg.algorithm.n_envs)
        action_agent = cem_rl.get_acquisition_actor(i)
        acquisition_actors.append(action_agent)
        temporal_agent=TemporalAgent(Agents(env_agent, action_agent))
        temporal_agent.seed(cfg.algorithm.env_seed)
        agent=AsynchronousAgent(temporal_agent)
        acquisition_agents.append(agent)

    n_interactions = 0
    for _ in range(cfg.algorithm.max_epochs):
        acquisition_workspaces = []
        nb_agent_finished = 0
        while(nb_agent_finished < pop_size):
            n_to_launch = min(pop_size-nb_agent_finished, n_processes)
            for idx_agent in range(n_to_launch):        
                idx_weight = idx_agent + nb_agent_finished
                cem_rl.update_acquisition_actor(acquisition_actors[idx_agent],idx_weight)
                
                # TODO: add noise args to agents interaction with env ? Alois does not. 
                acquisition_agents[idx_agent](t=0,stop_variable="env/done")

            #Wait for agents execution
            running=True
            while running:
                are_running = [a.is_running() for a in acquisition_agents[:n_to_launch]]
                running = any(are_running)
            nb_agent_finished += n_to_launch
            acquisition_workspaces += [a.get_workspace() for a in acquisition_agents[:n_to_launch]]

        ## Logging rewards:
        for acquisition_worspace in acquisition_workspaces:
            n_interactions += (
                acquisition_worspace.time_size() - 1
            ) * acquisition_worspace.batch_size()

        agents_creward = torch.zeros(len(acquisition_workspaces))
        for i,acquisition_worspace in enumerate(acquisition_workspaces):
            done = acquisition_worspace['env/done']
            cumulated_reward = acquisition_worspace['env/cumulated_reward']
            creward = cumulated_reward[done]
            agents_creward[i] = creward.mean()

        logger.add_scalar(f"monitor/n_interactions", n_interactions, n_interactions)
        logger.add_scalar(f"monitor/reward", agents_creward.mean().item(), n_interactions)
        logger.add_scalar(f"monitor/reward_best", agents_creward.max().item(), n_interactions)

        cem_rl.train(acquisition_workspaces,n_interactions,logger)

    for a in acquisition_agents:
        a.close()



def debug_train(cfg):
    """Train function without multi processing."""
    # init 
    cem_rl = CemRl(cfg)
    logger = instantiate_class(cfg.logger)

    pop_size = cfg.algorithm.es_algorithm.pop_size
    env_agent = env_agent = AutoResetGymAgent(make_gym_env,{'max_episode_steps':cfg.env.max_episode_steps,
                                        'env_name':cfg.env.env_name},
                                        n_envs=cfg.algorithm.n_envs//cfg.algorithm.num_processes)
    acquisition_actor = cem_rl.get_acquisition_actor(0)
    acquisition_agent = TemporalAgent(Agents(env_agent, acquisition_actor))
    acquisition_agent.seed(cfg.algorithm.env_seed)

    n_interactions = 0
    for epoch in range(cfg.algorithm.max_epochs):
        acquisition_workspaces = []
        for i in range(pop_size):
            workspace = Workspace()
            cem_rl.update_acquisition_actor(acquisition_actor,i)
            acquisition_agent(workspace,t=0,stop_variable="env/done")
            acquisition_workspaces.append(workspace)
        ## Logging rewards:
        for acquisition_worspace in acquisition_workspaces:
            n_interactions += (
                acquisition_worspace.time_size() - 1
            ) * acquisition_worspace.batch_size()

        agents_creward = torch.zeros(len(acquisition_workspaces))
        for i,acquisition_worspace in enumerate(acquisition_workspaces):
            done = acquisition_worspace['env/done']
            cumulated_reward = acquisition_worspace['env/cumulated_reward']
            creward = cumulated_reward[done]
            agents_creward[i] = creward.mean()

        logger.add_scalar(f"monitor/n_interactions", n_interactions, n_interactions)
        logger.add_scalar(f"monitor/reward", agents_creward.mean().item(), n_interactions)
        logger.add_scalar(f"monitor/reward_best", agents_creward.max().item(), n_interactions)
            
        cem_rl.train(acquisition_workspaces,n_interactions,logger)


@hydra.main(config_path=os.path.join(os.getcwd(),'run_launcher/configs/'), config_name="cem_rl_only_td3.yaml")
def main(cfg : DictConfig):
    import torch.multiprocessing as mp
    mp.set_start_method("spawn")
    synchronized_train_multi(cfg)
    # debug_train(cfg)

if __name__=='__main__':
    # print(os.getcwd())
    # sys.path.append(os.getcwd())
    main()
