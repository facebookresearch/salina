from salina import Agent
from salina.agents import Agents
from brax.envs import _envs, create_gym_env
from brax.envs.to_torch import JaxToTorchWrapper
from gym.wrappers import TimeLimit
from salina import instantiate_class
import gym
import torch

class MetaRLAgent(Agent):
    def __init__(self,n_exploration_episodes,n_exploitation_episodes,keep_exploration_reward=True,task_id=None):
        super().__init__()
        self.n_exploration_episodes=n_exploration_episodes
        self.n_exploitation_episodes=n_exploitation_episodes
        self.task_id=task_id
        self.keep_exploration_reward=keep_exploration_reward

    def forward(self,t,**extra_args):
        done=self.get(("env/done",t))
        B=done.size()[0]
        if t==0:
            new_idx=torch.zeros(B).long().to(done.device)
            episode_ended=torch.zeros(B).bool().to(done.device)
            is_exploration=new_idx.lt(self.n_exploration_episodes)
            is_exploitation=torch.logical_not(is_exploration)

            self.set(("env/meta/episode_idx",t),new_idx)
            self.set(("env/meta/timestep",t),new_idx)
            self.set(("env/meta/done",t),episode_ended)
            self.set(("env/meta/is_exploration",t),is_exploration)
            self.set(("env/meta/is_exploitation",t),is_exploitation)
        else:
            p_idx=self.get(("env/meta/episode_idx",t-1))
            p_done=self.get(("env/done",t-1))

            new_idx=(p_idx+p_done.float()).long()
            p_episode_ended=self.get(("env/meta/done",t-1)).float()
            new_idx=(p_episode_ended*torch.zeros(B).long().to(done.device)+(1-p_episode_ended)*new_idx).long()
            episode_ended=torch.logical_and(done,new_idx.eq(self.n_exploration_episodes+self.n_exploitation_episodes-1))

            ts=self.get(("env/meta/timestep",t-1))
            ts=ts+1
            ts=(ts*(1-p_episode_ended.float())).long()


            is_exploration=new_idx.lt(self.n_exploration_episodes)
            is_exploitation=torch.logical_not(is_exploration)

            self.set(("env/meta/timestep",t),ts)
            self.set(("env/meta/episode_idx",t),new_idx)
            self.set(("env/meta/done",t),episode_ended)
            self.set(("env/meta/is_exploration",t),is_exploration)
            self.set(("env/meta/is_exploitation",t),is_exploitation)

        if t==0:
            r=self.get(("env/reward",t))
            r=r*is_exploitation.float()
            self.set(("env/meta/cumulated_exploitation_reward",t),r.clone())
        else:
            cr=self.get(("env/meta/cumulated_exploitation_reward",t-1))
            ts=self.get(("env/meta/timestep",t))
            cr=cr*(1.0-ts.eq(0).float())
            r=self.get(("env/reward",t))
            r=r*is_exploitation.float()
            cr=cr+r
            self.set(("env/meta/cumulated_exploitation_reward",t),cr)

        if not self.keep_exploration_reward:
            r=self.get(("env/reward",t))
            r=r*(1.0-is_exploration.float())
            self.set(("env/reward",t),r)

        if t==0:
            r=self.get(("env/reward",t))
            self.set(("env/meta/cumulated_reward",t),r.clone())
        else:
            cr=self.get(("env/meta/cumulated_reward",t-1))
            ts=self.get(("env/meta/timestep",t))
            cr=cr*(1.0-ts.eq(0).float())
            cr=cr+self.get(("env/reward",t))
            self.set(("env/meta/cumulated_reward",t),cr)


        if not self.task_id is None:
            tid=(torch.ones(B)*self.task_id).long().to(done.device)
            self.set(("env/meta/task_id",t),tid)

class MetaEpisodeDone(Agent):
    def __init__(self):
        super().__init__()

    def forward(self, t, **kwargs):
        if t > 0:
            d = self.get(("env/meta/done", t))
            p_d = self.get(("env/meta/done", t-1))
            r = torch.logical_or(d,p_d)
            self.set(("env/meta/done", t), r)

class MetaRLEnvAutoReset(Agents):
    def __init__(self,autoreset_env_agent,n_exploration_episodes,n_exploitation_episodes,keep_exploration_reward=True,task_id=None):
        super().__init__(autoreset_env_agent,MetaRLAgent(n_exploration_episodes,n_exploitation_episodes,keep_exploration_reward,task_id))

class MetaRLEnvNoAutoReset(Agents):
    def __init__(self,autoreset_env_agent,n_exploration_episodes,n_exploitation_episodes,keep_exploration_reward=True,task_id=None):
        super().__init__(autoreset_env_agent,MetaRLAgent(n_exploration_episodes,n_exploitation_episodes,keep_exploration_reward,task_id),MetaEpisodeDone())

def make_env(args):
    if not "env_name" in args:
        return make_class_env(**args)

    if args["env_name"].startswith("brax/"):
        env_name=args["env_name"][5:]
        return make_brax_env(env_name)
    else:
        assert args["env_name"].startswith("gym/")
        env_name=args["env_name"][4:]
        return make_gym_env(env_name,args["max_episode_steps"])

def make_class_env(**args):
    e=instantiate_class(args)
    e = TimeLimit(e, max_episode_steps=args["max_episode_steps"])
    return e

def make_brax_env(env_name):
    e = create_gym_env(env_name)
    return JaxToTorchWrapper(e)

def make_gym_env(env_name,max_episode_steps):
    e = gym.make(env_name)
    e = TimeLimit(e, max_episode_steps=max_episode_steps)
    return e
