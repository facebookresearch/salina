# A2C Asynchronous Evaluation

Starting from the multi-CPUs implementation, we explain how we can implement a non blocking evaluation of the policy made on other CPUs

* It makes use of the `RemoteAgent._asynchronous_call` function that is a non-blocking function (when `num_processes>0`).
* It uses two copies of the policy agents - one for training and one for evaluation
* Note that, in that case, we can evaluate the policy in deterministic mode
