# A2C Examples (and Tutorial)

The examples of A2C implementations illustrate the different capabilities of the library and act as a **tutorial**. To understand all the different aspects, you have to read the examples in the following order:

* [Mono CPU](mono_cpu/) - The simplest implementation
* [Mono CPU 2](mono_cpu_2/) - Splitting the agent in three to illustrate the modularity of `salina`, seperating the critic from the policy.
* [Multi CPUs](multi_cpus/) - Making use of the `RemoteAgent` to execute over multiple CPUs in parallel
* [Asynchronous Evaluation](complete_with_async_eval/) - Making use of the asynchronous functionnality to evaluate policies in parallel without slowing down the learning
* [GPU](gpu/) - Making use of a GPU for the loss computation
* [Complete](complete/) - The reference implementation of A2C used for benchmarking. It contains different policies architectures to illustrate how one can define complex policies.
