import gym
from gym.wrappers import TimeLimit

def make_env(env_name, max_episode_steps):
    return TimeLimit(gym.make(env_name), max_episode_steps=max_episode_steps)