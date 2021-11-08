# RewardToGo on full episodes

A re-implementation of Decision Transformers where the policy s guided by the reward-to-go

We propose two implementations:
* One with a simple MLP P(a/s,reward-to-go)
* One with a transformer architecture P(a_t/s_t,reward_to_go,a_{t-1},s_{t-1},.....)

To launch with a simple MLP: `python ludc_salina/salina_examples/offline_rl/decision_transformer/dt.py`
To launch with a Transformer: `python ludc_salina/salina_examples/offline_rl/decision_transformer/dt.py -cn gym_transformers`

Note that the architecture of the network may not be exactly the same than in the original paper, explaining few differences in the final results.
