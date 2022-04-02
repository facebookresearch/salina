

from algorithms.ddpg import ddpg,get_env_dimensions,soft_param_update
from salina import Workspace,Agent,instantiate_class,get_class
from salina.agents import Agents,TemporalAgent,NRemoteAgent
from salina.agents.gyma import AutoResetGymAgent
from salina.logger import TFLogger
from salina.rl.replay_buffer import ReplayBuffer
import copy
import torch
from models.utils import Salina_Actor_Decorator,Salina_Qcritic_Decorator


class td3(ddpg):
    def __init__(self,cfg) -> None:

        self.cfg = cfg
        self.device = cfg.algorithm.device 
        if self.device == 'cuda':
            assert torch.cuda.is_available()

        obs_dim,action_dim,max_action = get_env_dimensions(self.cfg.env)
        
        self.n_q_agents = 2
        # create agents
        common_nn_args = {'state_dim':obs_dim,'action_dim':action_dim,'max_action':max_action}
        # To merge dict_a and dict_b in one line you can write :  {**dict_a,**dict_b}
        nn_args= {**dict(cfg.algorithm.action_agent),**common_nn_args}
        self.action_agent =  get_class(cfg.algorithm.action_agent)(**nn_args).to(self.device)
        if not isinstance(self.action_agent,Agent):
            self.action_agent = Salina_Actor_Decorator(self.action_agent)
            

        nn_args= {**dict(cfg.algorithm.q_agent),**common_nn_args}        
        self.q_agents = [get_class(cfg.algorithm.q_agent)(**nn_args).to(self.device) 
                         for _ in range(self.n_q_agents)]
        for i,_ in enumerate(self.q_agents):
            if not isinstance(self.q_agents[i],Agent):
                self.q_agents[i] = Salina_Qcritic_Decorator(self.q_agents[i])
            
        
        self.target_action_agent = copy.deepcopy(self.action_agent)
        self.target_q_agents = [copy.deepcopy(q_agent) for q_agent in self.q_agents]

        # create temporal agents
        self.t_action_agent = TemporalAgent(self.action_agent)
        self.t_target_action_agent = TemporalAgent(self.target_action_agent)
        self.t_q_agents = [TemporalAgent(q_agent) for q_agent in self.q_agents] 
        self.t_target_q_agents = [TemporalAgent(target_q_agent) for target_q_agent in self.target_q_agents] 
        self.t_q_agent = self.t_q_agents[0] # t_q_agent is the q_learning agent used in the actor loss

        # create optimizers
        self.create_optimizers()

        self.replay_buffer = ReplayBuffer(self.cfg.algorithm.buffer_size,device=self.device)

    def create_optimizers(self):
        self.optimizer_actor_agent = torch.optim.Adam(self.action_agent.parameters(),lr=self.cfg.algorithm.optimizer.lr)
        self.optimizer_q_agent = [torch.optim.Adam(q_agent.parameters(),lr=self.cfg.algorithm.optimizer.lr)    
                                                    for q_agent in self.q_agents]

    def train_critic_and_actor(self,n_actor_steps,n_total_actor_steps,logger):

        # TODO: Verify if original implementation has 2 independant loops: 
        # first the critic and than the actor. 
        for i in range(n_actor_steps):
            grad_step_id = n_total_actor_steps-n_actor_steps+i
            train_workspace =  self.replay_buffer.get(self.cfg.algorithm.batch_size)
            self.train_critic(train_workspace,grad_step_id,logger)
            if i % self.cfg.algorithm.policy_delay:
                self.train_actor(train_workspace,grad_step_id,logger)

    def train_critic(self,train_workspace,step_id,logger):
        done, reward = train_workspace["env/done", "env/reward"]
        train_workspace = train_workspace.to(self.device)
        # Train the critic : 
        ## Compute q(s,a) into the workspace: 
        q_values =[]
        target_q_values = []
        for i in range(self.n_q_agents): # Compute Q(s,a) for each critics
            self.t_q_agents[i](train_workspace,t=0,detach_action=True,n_steps=self.cfg.algorithm.time_size)
            q_values.append(train_workspace['q_value'].squeeze(-1))
        self.t_target_action_agent(train_workspace,t=0,n_steps=self.cfg.algorithm.time_size) # replace action by \pi_target(s)
        for i in range(self.n_q_agents): # Compute Q(s',\pi_target(a)) for each critics
            with torch.no_grad():
                self.t_target_q_agents[i](train_workspace,t=0,n_steps=self.cfg.algorithm.time_size,
                                        epsilon = self.cfg.algorithm.action_noise) # TD3 : adding noise
                target_q_values.append(train_workspace['q_value'])
        target_q_value = torch.min(*target_q_values).squeeze(-1) # TD3 : the target is the mean of two critics

        for i in range(self.n_q_agents): # TD3, two critics are trained with the same data
            q_value = q_values[i]
            target = reward[1:] + self.cfg.algorithm.discount_factor * target_q_value[1:]*(~done[1:])
            td_error = (target - q_value[:-1])
            critic_loss = torch.mean(td_error**2)
            logger.add_scalar("loss/critic_loss", critic_loss.mean().item(), step_id)
            
            self.optimizer_q_agent[i].zero_grad()
            critic_loss.backward()
            if self.cfg.algorithm.clip_grad !=None:
                n= torch.nn.utils.clip_grad_norm_(self.q_agents[i].parameters(), self.cfg.algorithm.clip_grad)
                logger.add_scalar("monitor/grad_norm_q_critic", n.item(), step_id)
            self.optimizer_q_agent[i].step()
    
            soft_param_update(self.target_q_agents[i],self.q_agents[i],self.cfg.algorithm.update_target_rho)

        