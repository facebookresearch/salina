# -*- coding: utf-8 -*-
"""xCPU_CEM_SALINA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EndNzK-cKDTz0Lzn7jnqMIGqaYhEUTo1

# Outlook

In this colab we code a multi CPU version of the Cross-Entropy Method (CEM) using SaLinA, so as to better understand the inner mechanisms.

### Installation

The SaLinA library is [here](https://github.com/facebookresearch/salina).

Note the trick: we first try to import, if it fails we install the github repository and import again.
"""

import time
import numpy as np
import copy # used for multiprocessing

import gym
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra

import salina
from salina import Agent, get_arguments, instantiate_class, Workspace, get_class, instantiate_class
from salina.agents import Agents, RemoteAgent, TemporalAgent, NRemoteAgent
from salina.agents.asynchronous import AsynchronousAgent
from salina.agents.gyma import NoAutoResetGymAgent, GymAgent
from omegaconf import DictConfig, OmegaConf

class ActionAgent(Agent):
    def __init__(self, observation_size, hidden_size1, hidden_size2, action_size):
        super().__init__(name="action_agent") # Note that we passed the name to retrieve the agent by its name
        self.model = nn.Sequential(
            nn.Linear(observation_size, hidden_size1),
            nn.ReLU(), #is this one useful ???
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, action_size),
        )

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs", t))
        action = self.model(observation)
        self.set(("action", t), action)

class EnvAgent(NoAutoResetGymAgent):
  # Create the environment agent
  # This agent implements N gym environments with auto-reset
  def __init__(self, cfg, n_envs):
    super().__init__(
      get_class(cfg.env),
      get_arguments(cfg.env),
      n_envs=n_envs
    )
    env = instantiate_class(cfg.env)
    self.observation_space=env.observation_space
    self.action_space=env.action_space
    del(env)

  def get_obs_and_actions_sizes(self):
    return self.observation_space.shape[0], self.action_space.shape[0]

  def get_obs_and_actions_sizes_discrete(self):
    return self.observation_space.shape[0], self.action_space.n

class CovMatrix():
    def __init__(self, centroid: torch.Tensor, sigma, noise_multiplier):
        policy_dim = centroid.size()[0]
        self.noise = torch.diag(torch.ones(policy_dim) * sigma)
        self.cov = torch.diag(torch.ones(policy_dim) * torch.var(centroid)) + self.noise
        self.noise_multiplier = noise_multiplier

    def update_noise(self) -> None:
        self.noise = self.noise * self.noise_multiplier

    def generate_weights(self, centroid, pop_size):
      dist = torch.distributions.MultivariateNormal(centroid, covariance_matrix=self.cov)
      weights = [dist.sample() for _ in range(pop_size)]
      return weights

    def update_covariance(self, elite_weights) -> None:
      self.cov = torch.cov(elite_weights.T) + self.noise

def run_cem(cfg):
  # 1)  Build the  logger
  torch.manual_seed(cfg.algorithm.env_seed)
  logger = instantiate_class(cfg.logger)
  
  pop_size = cfg.algorithm.pop_size

  temporal_agents=[]
  for _ in range(cfg.algorithm.n_processes): 
    env_agent = EnvAgent(cfg,n_envs=cfg.algorithm.n_envs)
    observation_size, action_size = env_agent.get_obs_and_actions_sizes()
    action_agent = ActionAgent(observation_size, cfg.algorithm.architecture.hidden_size1, cfg.algorithm.architecture.hidden_size2, action_size)  
    temporal_agent=TemporalAgent(Agents(env_agent, action_agent))
    temporal_agent.seed(cfg.algorithm.env_seed)
    agent=AsynchronousAgent(temporal_agent)
    temporal_agents.append(agent)
  
  # This agent is used to convert parameters to vectors since the vector_to_parameters does not seem to work when moduule are in different processes
  env_agent = EnvAgent(cfg,n_envs=cfg.algorithm.n_envs)
  observation_size, action_size = env_agent.get_obs_and_actions_sizes()    
  action_agent = ActionAgent(observation_size, cfg.algorithm.architecture.hidden_size1, cfg.algorithm.architecture.hidden_size2, action_size)  
  temporal_agent=TemporalAgent(Agents(env_agent, action_agent))

  centroid = torch.nn.utils.parameters_to_vector(temporal_agent.parameters())
  matrix = CovMatrix(centroid, cfg.algorithm.sigma, 
                     cfg.algorithm.noise_multiplier, 
                     )

  best_score = -np.inf

  # 7) Training loop
  for epoch in range(cfg.algorithm.max_epochs):
    matrix.update_noise()
    scores = []
    weights = matrix.generate_weights(centroid, pop_size) 

    position=0
    while(position<pop_size):
      n_to_launch=min(pop_size-position,cfg.algorithm.n_processes)
      # Execute the agents
      for idx_agent in range(n_to_launch):        
        idx_weight=idx_agent+position
        torch.nn.utils.vector_to_parameters(weights[idx_weight], temporal_agent.parameters())
        temporal_agents[idx_agent].agent.load_state_dict(temporal_agent.state_dict())
        temporal_agents[idx_agent](t=0,stop_variable="env/done")
      
      #Wait for agents execution
      running=True
      while running:
        are_running=[a.is_running() for a in temporal_agents[:n_to_launch]]
        running=any(are_running)

      #Compute collected reward
      workspaces=[a.get_workspace() for a in temporal_agents[:n_to_launch]]
      for k,workspace in enumerate(workspaces):      
        episode_lengths=workspace["env/done"].float().argmax(0)+1
        arange=torch.arange(cfg.algorithm.n_envs) 
        mean_reward=workspace["env/cumulated_reward"][episode_lengths-1,arange].mean().item()
        scores.append(mean_reward)
        if mean_reward >= best_score:
          best_score = mean_reward
          best_params=weights[k+position]
    
        if cfg.verbose > 0: 
          print(f"Indiv: {k+position + 1} score {scores[k+position]:.2f}")

      position+=n_to_launch

    print("Best score: ", best_score)
    # Keep only best individuals to compute the new centroid
    elites_idxs=np.argsort(scores)[-cfg.algorithm.elites_nb :]
    elites_weights = [weights[k] for k in elites_idxs]
    elites_weights = torch.cat([torch.tensor(w).unsqueeze(0) for w in elites_weights],dim=0)
    centroid = elites_weights.mean(0)

    # Update covariance
    matrix.update_noise()
    matrix.update_covariance(elites_weights)
  
  for a in temporal_agents:
    a.close()

@hydra.main(config_path=".", config_name="main.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    run_cem(cfg)

if __name__ == "__main__":
    main()
