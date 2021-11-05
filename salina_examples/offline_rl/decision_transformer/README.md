# RewardToGo on full episodes

A re-implementation of Decision Transformers where the policy s guided by the reward-to-go

We propose two implementations:
* One with a simple MLP P(a/s,reward-to-go)
* One with a transformer architecture P(a_t/s_t,reward_to_go,a_{t-1},s_{t-1},.....)
