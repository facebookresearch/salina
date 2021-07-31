# Reinforce Mono CPU

We propose a simple implementation of Reinforce on one single CPU with a MLP policy. It illustrates how one can sample complete episodes in a workspace. Note that, since episodes may have different lengths when collecting multiple episodes in parallel, the loss computation involve the computation of a `mask` making the overall code less simple than for other algorithms like A2C for instance.

To extend REINFORCE to multiple cpus and gpus, please have a look at the [A2C](../a2c) examples.
