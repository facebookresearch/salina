# Examples

We provide examples of different algorithms in different domains.

## Reinforcement Learning

We provide a set of [Benchmarked RL algorithms](rl/) and also **simplified versions** to allow to well understand the principle of the library. These algorithms can be easily modified to start a new project.

* [A2C](rl/a2c/) Algorithms are used as a **tutorial** to present different functionalities of `salina`
* [REINFORCE](rl/reinforce/) shows a simple implementation of REINFORCE with `salina`.
* [DQN](rl/dqn/) proposes implementations of Q-Learning algorithms
* [TD3](rl/td3/)
* [DDPG](rl/ddpg/)
* [PPO on Brax](rl/ppo_brax/)
* [PPO Continuous actions](rl/ppo_continuous/)
* [PPO Discrete actions](rl/ppo_discrete/)
* (more to come, please do a Pull request if you implement other algorithms)

Note that due to the modularity of `salina`, all the implementations work with any type of policies (recurrent, hierarchical, transformers,....)

## Offline/Imitation Learning

Based on D4RL, we provide implementations of algorithms for off-policy learning

* [BC](offline_rl/bc) Behavioral Cloning

## Computer Vision

See [Computer Vision Algorithms](computer_vision/)

## Language Modelling

Under progress...

## Speed Test

It is a simple script to compute the execution speed of any agent. It allows to optimize experiments by choosing the right ratio between the batch size, the time size and the number of processes
