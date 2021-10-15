# A2C Mono CPU - 2

In this version, we propose to seperate the computation of the action from the computation of the critic.
* First, the workspace is computed by executing the environment and the policy
  * the policy is represented by two agents: the `prob_agent` computing action probabilities, and the `action_agent` computing the action
* In a second time, the `critic_agent` is computed

It illustrates the modularity of the library that allows to implement very complex policies by combining elementary agents -- see [other examples](../complete/)
