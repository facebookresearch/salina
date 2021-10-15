# A2C GPU

We illustrate how computation can be accelerated using one GPU for loss computation

* First, the `prob_agent` has to be copied such that we maintain one version on CPU for trajectory acquisition, and one version on GPU for loss computation
* Second, before computing the loss, we just move the `replay_workspace` to GPU memory to do computation on GPU: `workspace=workspace.to(device)`
* Third, the acquisition agent (on CPU) and the loss agent (on GPU) has to be synchronized at every epoch through `agent.load_state_dict`
 -- an `Agent` is a `nn.Module`

Note that it is also possible to do trajectories acquisition on GPU (typically using another device), which is not illustrated in this example.
