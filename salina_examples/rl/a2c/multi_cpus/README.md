# A2C Multi CPUs

We explain how multiple cpus can be used to sample trajectories

One imporant point when using a `RemoteAgent` is that this agent is returning a `SharedWorkspace` at first call (which is a workspace on which the different processes can share information at high speed). This `SharedWorkspace` does not contain any gradient information. It has to be converted in a classical `Workspace` to be used for gradient computation through `SharedWorkspace.convert_to_workspace()`.

In our case, it means that the `prob_agent` and `critic_agent` have to be forwarded a second time to compute tensors with gradient. Note that this second forward is made on large batches (all the batches collected through multiple processes) and can be very efficient  -- particularly on [GPU](../gpu/)
