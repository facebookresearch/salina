import os
import sys
import gym
from gym.wrappers import TimeLimit
import torch
import hydra
from omegaconf import DictConfig
from salina import instantiate_class, get_arguments
from salina.agents import Agents, TemporalAgent, NRemoteAgent
from salina.agents.gyma import AutoResetGymAgent
sys.path.append(os.getcwd())
from pbt_utils import PBT_Agent


# TODO: clean remove this function with better env creation
def make_gym_env(max_episode_steps, env_name):
    return TimeLimit(gym.make(env_name), max_episode_steps=max_episode_steps)


def get_crewards(workspace):
    """Return the mean cumulative reward of each trajectories"""
    done = workspace['env/done']
    cumulated_reward = workspace['env/cumulated_reward']
    fitness = cumulated_reward[done]
    return fitness


def update_actors(actor_containers,acq_actors):
    for actor_container,acq_actor in zip(actor_containers,acq_actors):
        actor_container.agents = torch.nn.ModuleList([acq_actor])


def synchronous_pbt(cfg):
    # Init 
    logger = instantiate_class(cfg.logger)
    env_args = get_arguments(cfg.env)
    n_env_per_process = cfg.algorithm.n_train_envs // cfg.algorithm.n_process
    acq_env_agent = AutoResetGymAgent(make_gym_env, env_args,
                                      n_envs=n_env_per_process)
    n_acq_steps = cfg.algorithm.n_timesteps
    pop_size = cfg.algorithm.es_algorithm.pop_size
    rl_population = [PBT_Agent(cfg) for _ in range(pop_size)]

    acq_agent = TemporalAgent(Agents(acq_env_agent,
                                    (rl_population[0].get_acq_actor())))
    
    remote_agent, workspace = NRemoteAgent.create(acq_agent,
                                                  num_processes=cfg.algorithm.n_process,
                                                  t=0, n_steps=n_acq_steps,
                                                  **rl_population[0].get_acq_args())
    remote_agent.seed(cfg.seed)
    es_algorithm = instantiate_class(cfg.algorithm.es_algorithm)
    es_algorithm.init(rl_population)
    total_timesteps = 0
    agent_time_step = torch.zeros(pop_size)
    for epoch in range(cfg.algorithm.max_epochs):
        agent_fitness = torch.zeros(pop_size)
        ########    (1) Train Phase         ########
        # Train pop for train_budget steps
        for i,pbt_a in enumerate(rl_population):
            n_train_it_timesteps = 0
            kwargs = pbt_a.get_acq_args()
            fitness = []
            while n_train_it_timesteps < cfg.algorithm.train_budget:
                # Transitions acquisition (multiple processes)
                pbt_a.update_acquisition_agent(remote_agent)
                if epoch == 0:
                    remote_agent(workspace, t=0,
                                 n_steps=n_acq_steps, **kwargs)
                else:
                    workspace.copy_n_last_steps(1)
                    remote_agent(workspace, t=1,
                                 n_steps=n_acq_steps - 1, **kwargs)
                steps = (workspace.time_size() - 1) * workspace.batch_size()
                n_train_it_timesteps += steps
                total_timesteps += steps
                agent_time_step[i] += steps
                
                # RL gradient steps (main processes)
                pbt_a.train(workspace, n_train_it_timesteps, total_timesteps, logger)

                # Compute fitness 
                new_fitnesses = get_crewards(workspace)
                fitness = [*fitness, *new_fitnesses]
                if len(new_fitnesses) > 0:
                    logger.add_scalars('monitor/reward_pop',
                                       {f'{i}': new_fitnesses.mean().item()},
                                       agent_time_step[i])
                # TODO : clarify absence of usage of discount factor.
                # because it means that RL and ES won't optimize the
                # same objective function.
            
            #######    (2) Evaluation      ########
            # The agent final fitness before evolutions strategies is
            # the mean of it's last 10 episodic rewards.
            agent_fitness[i] = torch.mean(torch.tensor(fitness)[-10:])
            max_id = torch.argmax(agent_fitness)
            logger.add_scalar('monitor/reward_best', agent_fitness[max_id].item(),
                              agent_time_step[max_id])
                
        #######    (3) Select & Mutate     ########
        es_algorithm.tell(rl_population, agent_fitness)
        rl_population = es_algorithm.ask()

@hydra.main(config_path=os.path.join(os.getcwd(),'run_launcher/configs/'), config_name="pbt.yaml")
def main(cfg : DictConfig):
    import torch.multiprocessing as mp
    mp.set_start_method("spawn")
    synchronous_pbt(cfg)
    

if __name__=='__main__':
    main()