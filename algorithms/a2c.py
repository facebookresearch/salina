from typing import Union
from algorithms.learner import learner
import copy
import torch
from torch import nn
from salina import Workspace, get_class, get_arguments
from utils import get_env_dimensions
from salina.agents import TemporalAgent, Agents
from omegaconf import DictConfig, ListConfig, OmegaConf


def _index(tensor_3d, tensor_2d):
    """This function is used to index a 3d tensors using a 2d tensor"""
    x, y, z = tensor_3d.size()
    t = tensor_3d.reshape(x * y, z)
    tt = tensor_2d.reshape(x * y)
    v = t[torch.arange(x * y), tt]
    v = v.reshape(x, y)
    return v


class A2C(learner):

    def __init__(self,**kwargs)-> None:
        
        self.h_params = DictConfig(kwargs)
        self.device = self.h_params.device 
        if self.device == 'cuda':
            assert torch.cuda.is_available()

        obs_dim, n_action = get_env_dimensions(self.h_params.env)
        # create agents
        common_nn_args = {'state_dim': obs_dim, 'n_action': n_action}
        # To merge dict_a and dict_b in one line
        # you can write :  {**dict_a, **dict_b}
        nn_args = {**dict(self.h_params.prob_agent), **common_nn_args}
        self.prob_agent = get_class(self.h_params.prob_agent)(**nn_args)
        self.action_agent = get_class(self.h_params.action_agent)(**nn_args)

        nn_args = {**dict(self.h_params.v_agent), **common_nn_args}
        self.v_agent = get_class(self.h_params.v_agent)(**nn_args)

        # create temporal agents
        self.t_prob_agent = TemporalAgent(self.prob_agent).to(self.device)
        self.t_action_agent = TemporalAgent(self.action_agent)
        self.t_v_agent = TemporalAgent(self.v_agent).to(self.device)

        # create optimizers
        optimizer_args = get_arguments(self.h_params.optimizer)
        parameters = nn.Sequential(self.prob_agent, self.v_agent).parameters()
        self.optimizer = get_class(self.h_params.optimizer)(parameters,
                                                            **optimizer_args)
    
    def get_hyper_params(self) -> Union[DictConfig, ListConfig] :
        return self.h_params 

    def apply_hyper_params(self, params: DictConfig):
        self.h_params = OmegaConf.merge(self.h_params,params,str)
        # Caution with hyper-parameters contained in sub-classes: 
        # ex : self.optimizer.lr = self.h_params.lr
        
    def get_acquisition_actor(self):
        '''
        Returns the agents used to gather experiments in the environment.
        '''
        self.prob_agent.set_name("prob_agent")
        acquisition_actor = copy.deepcopy(Agents(self.prob_agent, self.action_agent))
        acquisition_actor.set_name("acquisition_actor")
        return acquisition_actor

    def update_acquisition_agent(self, acquisition_agent):
        for a in acquisition_agent.get_by_name("prob_agent"):
            a.load_state_dict(self.prob_agent.state_dict())

    def get_acquisition_args(self):
        return {'stochastic':True}
        
    def get_evaluation_args(self):
        return {'stochastic':False}
        
    def train(self, acq_workspace, actor_steps, total_actor_steps, logger=None):
        replay_workspace = Workspace(acq_workspace).to(self.device)
        n_timesteps = replay_workspace.time_size()
        self.t_prob_agent(replay_workspace, t=0, n_steps=n_timesteps)
        self.t_v_agent(replay_workspace, t=0, n_steps=n_timesteps)
        critic, done, action_probs, reward, action = replay_workspace[
            "critic", "env/done", "action_probs", "env/reward", "action"
        ]
        target = reward[1:] + self.h_params.discount_factor * critic[1:].detach() * (
            1 - done[1:].float()
        )
        td = target - critic[:-1]
        td_error = td ** 2
        critic_loss = td_error.mean()

        entropy_loss = torch.distributions.Categorical(action_probs).entropy().mean()

        action_logp = _index(action_probs, action).log()
        a2c_loss = action_logp[:-1] * td.detach()
        a2c_loss = a2c_loss.mean()
        if logger:
            logger.add_scalar("learner/entropy_loss", entropy_loss.item(), total_actor_steps)
            logger.add_scalar("learner/critic_loss", critic_loss.item(), total_actor_steps)
            logger.add_scalar("learner/a2c_loss", a2c_loss.item(), total_actor_steps)
        loss = (- self.h_params.entropy_coef * entropy_loss
                + self.h_params.critic_coef * critic_loss
                - self.h_params.a2c_coef * a2c_loss
                )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
