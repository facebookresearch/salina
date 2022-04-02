      

import copy

from salina import Agent,instantiate_class,get_class
from salina.agents import TemporalAgent
from salina.rl.replay_buffer import ReplayBuffer
from collections.abc import Iterable
from torch.nn.utils.convert_parameters import parameters_to_vector,vector_to_parameters

import torch

from algorithms.learner import learner
from utils import get_env_dimensions,soft_param_update
from models.utils import Salina_Actor_Decorator,Salina_Qcritic_Decorator


class ddpg(learner):

    def __init__(self,cfg) -> None:

        self.cfg=cfg
        self.device = cfg.algorithm.device 
        if self.device == 'cuda':
            assert torch.cuda.is_available()
        
        obs_dim,action_dim,max_action = get_env_dimensions(self.cfg.env)
        # create agents
        common_nn_args = {'state_dim':obs_dim,'action_dim':action_dim,'max_action':max_action}
        # To merge dict_a and dict_b in one line you can write :  {**dict_a,**dict_b}
        nn_args= {**dict(cfg.algorithm.action_agent),**common_nn_args}
        self.action_agent =  get_class(cfg.algorithm.action_agent)(**nn_args)
        if not isinstance(self.action_agent,Agent):
            self.action_agent = Salina_Actor_Decorator(self.action_agent)

        nn_args= {**dict(cfg.algorithm.q_agent),**common_nn_args}
        self.q_agent = get_class(cfg.algorithm.q_agent)(**nn_args)
        if not isinstance(self.q_agent,Agent):
            self.q_agent = Salina_Qcritic_Decorator(self.q_agent)
        # create target agents :
        self.target_q_agent = copy.deepcopy(self.q_agent)
        self.target_action_agent = copy.deepcopy(self.action_agent)

        # create temporal agents
        self.t_action_agent = TemporalAgent(self.action_agent).to(self.device)
        self.t_target_action_agent = TemporalAgent(self.target_action_agent).to(self.device)
        self.t_q_agent = TemporalAgent(self.q_agent).to(self.device)
        self.t_target_q_agent = TemporalAgent(self.target_q_agent).to(self.device)

        # create optimizers
        self.create_optimizers()

        self.replay_buffer = ReplayBuffer(self.cfg.algorithm.buffer_size)
    
    def create_optimizers(self):
        self.optimizer_actor_agent = torch.optim.Adam(self.action_agent.parameters(),lr=self.cfg.algorithm.optimizer.lr)
        self.optimizer_critic_agent = torch.optim.Adam(self.q_agent.parameters(),lr=self.cfg.algorithm.optimizer.lr)  

    def set_actor_params(self, weight):
        ''' Overrite the parameters of the actor and the target actor '''
        vector_to_parameters(weight.detach().clone().to('cuda'),self.action_agent.parameters())
        vector_to_parameters(weight.detach().clone().to('cuda'),self.target_action_agent.parameters())
        # reset action optimizer: 
        self.optimizer_actor_agent = torch.optim.Adam(self.action_agent.parameters(),lr=self.cfg.algorithm.optimizer.lr)

    def get_parameters(self):
        return self.action_agent.parameters()
    
    def get_acquisition_actor(self):
        '''
        Returns the agents used to gather experiments in the environment.
        '''
        self.action_agent.set_name("action_agent")
        acquisition_action_actor = copy.deepcopy(self.action_agent)
        return acquisition_action_actor.to('cpu')

    def get_acquisition_args(self):
        return {'epsilon':self.cfg.algorithm.action_noise}

    def update_acquisition_agent(self,acquisitionAgent):
        for a in acquisitionAgent.get_by_name("action_agent"):
            a.load_state_dict(self.action_agent.state_dict())

    def workspace_to_replay_buffer(self,acq_worspace): # might will find better way to handle multi-workspace. 
        ''' Add to the replay buffer an acquisition workspace
            or a list of acquisition workspaces'''
        if isinstance(acq_worspace, Iterable):
            for workspace in acq_worspace:
                self.replay_buffer.put(workspace,time_size=self.cfg.algorithm.time_size)
        else: 
            self.replay_buffer.put(acq_worspace,time_size=self.cfg.algorithm.time_size)
        
    def train(self,acq_workspace,n_actor_steps,n_total_actor_steps,logger):

        self.workspace_to_replay_buffer(acq_workspace)
        logger.add_scalar("monitor/self.replay_buffer_size", self.replay_buffer.size(), n_total_actor_steps)
        
        if self.replay_buffer.size() < self.cfg.algorithm.initial_buffer_size:
                return
        self.train_critic_and_actor(n_actor_steps,n_total_actor_steps,logger)

    def train_critic_and_actor(self,n_actor_steps,n_total_actor_steps,logger):
        for i in range(n_actor_steps):
            grad_step_id = n_total_actor_steps-n_actor_steps+i
            train_workspace =  self.replay_buffer.get(self.cfg.algorithm.batch_size)
            train_workspace = train_workspace.to(self.device)
            
            self.train_critic(train_workspace,grad_step_id,logger)
            self.train_actor(train_workspace,grad_step_id,logger)

    def train_critic(self,train_workspace,n_interactions,logger):
        train_workspace = train_workspace.to(self.device)
        done, reward = train_workspace["env/done", "env/reward"]
            
        # Train the critic : 
        ## Compute q(s,a) into the workspace: 
        self.t_q_agent(train_workspace,t=0,detach_action=True,n_steps=self.cfg.algorithm.time_size)
        q_values = train_workspace['q_value'].squeeze(-1)

        with torch.no_grad():
            ## Compute q_target(s,pi_target(s)) into the workspace:
            self.t_target_action_agent(train_workspace,t=0,n_steps=self.cfg.algorithm.time_size)
            self.t_target_q_agent(train_workspace,t=0,n_steps=self.cfg.algorithm.time_size)
            target_q_value = train_workspace['q_value'].squeeze(-1)
                
            target = reward[1:] + self.cfg.algorithm.discount_factor * target_q_value[1:]*(~done[1:])
        td_error = (target - q_values[:-1])
        critic_loss = torch.mean(td_error**2)
        logger.add_scalar("learner/critic_loss", critic_loss.mean().item(), n_interactions)
        
        self.optimizer_critic_agent.zero_grad()
        critic_loss.backward()
        if self.cfg.algorithm.clip_grad !=None:
            n= torch.nn.utils.clip_grad_norm_(self.q_agent.parameters(), self.cfg.algorithm.clip_grad)
            logger.add_scalar("monitor/grad_norm_q_critic", n.item(), n_interactions)
        self.optimizer_critic_agent.step()
        soft_param_update(self.t_target_q_agent,self.q_agent,self.cfg.algorithm.update_target_rho)

    def train_actor(self,train_workspace,n_interactions,logger):
        train_workspace = train_workspace.to(self.device)
        # Compute q(s,pi(s)) into the workspace: 
                
        self.t_action_agent(train_workspace,t=0,n_steps=self.cfg.algorithm.time_size)
        self.t_q_agent(train_workspace,t=0,n_steps=self.cfg.algorithm.time_size) # TODO : strange using algorithm.time_size, is like having double batch_size.
        target_q_value = train_workspace['q_value']

        actor_loss = -torch.mean(target_q_value) 
        logger.add_scalar("learner/actor_loss", actor_loss.mean().item(), n_interactions)
        
        self.optimizer_actor_agent.zero_grad()
        actor_loss.backward()
        if self.cfg.algorithm.clip_grad !=None:
            n = torch.nn.utils.clip_grad_norm_(self.action_agent.parameters(), self.cfg.algorithm.clip_grad)
            logger.add_scalar("monitor/grad_norm_actor", n.item(), n_interactions)
        self.optimizer_actor_agent.step()

        soft_param_update(self.target_action_agent,self.action_agent,self.cfg.algorithm.update_target_rho)