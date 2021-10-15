# A2C Complete

Contains the code for the A2C algorithm working for anytype of polic and makes use of the General Advantage Estimator (GAE)

We provide different instances:
* `main.yaml`: A simple MLP
* `main_recurrent.yaml`: A recurrent NN for both the policy and the critic
* `main_recurrent_sep.yaml`: Two seperate recurrent NNs for the policy and the critic
* `main_atari.yaml`: Implementation for Atari Games

Note that the different policies/critics are obtained by combining agents in `agents.py`.

## Benchmark
