# Deep Q Learning

We provide different implementations of Deep Q-Learning working on multiple CPUs and one GPU for loss computation. All the implementations work for recurrent policies.

The implementations make use of the `ReplayBuffer` class which allows one to `put` a workspace and to `get` random workspaces. When putting a workspace, the workspace can be split into overlapping windows (i.e you can acquire trajectories of size 50, but store these trajectories as trajectories of size 2 in the replay buffer). Note that, since the `ReplayBuffer` returns a `Workspace`, it is very to replay an `Agent` on previously acquired trajectories just by dooing a forward pass.

## Implementations

* [Double DQN](double_dqn/): A Double DQN implementation with benchmark working with any trtypoe of policy (recurrent, non-recurrent, ...)
* [Simple Double DQN](simplest_double_dqn/): A simple implementation for non-recurrent policy used as a tutoria;
