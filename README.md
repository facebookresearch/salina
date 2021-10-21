# SaLinA: SaLinA - A Flexible and Simple Library for Learning Sequential Agents (including Reinforcement Learning)

## TL;DR

`salina` is a lightweight library extending PyTorch modules for developping **sequential decision models**.  It can be used for **Reinforcement Learning** (including model-based with differentiable environments, multi-agent RL, etc...), but also in a supervised/unsupervised learning settings (for instance for NLP, Computer Vision, etc...).
* It allows to write very complex sequential models (or policies) in few lines
* It works on multiple CPUs and GPUs

## News

* October 2021:
* * Week 18th of october
* * * Release of the library
* * * Various bug fixes
* * * Add train and eval mode in the RemoteAgent and NRemoteAgent
* * * [Cleaning of the BRAX PPO Implementation](salina_examples/rl/brax_ppo) to make it similar to other implementations
* * * [Cleaning of the Behavioral Cloning implementation](salina_examples/offline_rl/bc)
* * Week 25th of october
* * * [Transformer-based Agents](salina/agents/transformers.py)
* * * [Behavioral Cloning over complete episodes](salina_examples/offline_rl/bc_on_full_episodes)
* * * * [With](salina_examples/offline_rl/bc_on_full_episodes/gym_transformer.yaml) and [without](salina_examples/offline_rl/bc_on_full_episodes/gym.yaml) transformers
* * * [PPO on Brax with Transformers](salina/salina_examples/rl/ppo_brax_transformer)
* * * [GPU-accelerated environments agents](salina/salina_envs)
* * * * [2D maze without reward](salina/salina_envs/2dmazes/2dmaze_nowall.py)


## Quick Start

* Just clone the repo
* `pip install -e .`

### Documentation

* [Principles of the library](md_docs/)
* [Examples and Algorithms](salina_examples/)
* [Tutorial through multiple A2C implementations](salina_examples/rl/a2c)
* [Reinforcement Learning Benchmark](salina_examples/rl/BENCHMARK.md)
* Video Tutorials:
* * [Tutorial 1: Agent and Workspace](https://youtu.be/CSkkoq_k5zU)
* * [Tutorial 2: Multi-CPUs](https://youtu.be/I-trJWUJDMo)
* * [Tutorial 3: Implementing A2C](https://youtu.be/Mec8GEEQYhU)
* * [Tutorial 4: A2C on multi-CPUs](https://youtu.be/euDqlmcC_1Q)
* [Arxiv Paper](https://arxiv.org/abs/2110.07910)


**For development, set up [pre-commit](https://pre-commit.com) hooks:**

* Run `pip install pre-commit`
    * or `conda install -c conda-forge pre-commit`
    * or `brew install pre-commit`
* In the top directory of the repo, run `pre-commit install` to set up the git hook scripts
* Now `pre-commit` will run automatically on `git commit`!
* Currently isort, black and blacken-docs are used, in that order

## Organization of the repo

* [salina](salina/) is the core library
  * [salina.agents](salina/agents/) is the catalog of agents (the same as `torch.nn` but for agents)
* [salina_examples](salina_examples/) provide many examples (in different domains)

## Dependencies

`salina` utilizes `pytorch`, `hydra` for configuring experiments, and `gym` for reinforcement learning algorithms.

## Note on the logger

We provide a simple Logger that logs in both tensorboard format, but also as pickle files that can be re-read to make tables and figures. See [logger](salina/logger.py). This logger can be easily replaced by any other logger.

# Description

**Sequential Decision Making is much more than Reinforcement Learning**

* Sequential Decision Making is about interactions:
 * Interaction with data (e.g attention-models, decision tree, cascade models, active sensing, active learning, recommendation, etc….)
 * Interaction with an environment (e.g games, control)
 * Interaction with humans (e.g recommender systems, dialog systems, health systems, …)
 * Interaction with a model of the world (e.g simulation)
 * Interaction between multiple entities (e.g multi-agent RL)


## What `salina` is

* A sandbox for developping sequential models at scale.

* A small (300 hundred lines) 'core' code that defines everything you will use to implement `agents` involved in sequential decision learning systems.
  * It is easy to understand and use since it keeps the main principles of pytorch, just extending `nn.Module` to `Agent` in order to handle the temporal dimension
* A set of **agents** that can be combined (like pytorch modules) to obtain complex behaviors
* A set of references implementations and examples in different domains **Reinforcement Learning**, **Imitation Learning**, **Computer Vision**, with more to come...

## What `salina` is not

* Yet another reinforcement learning framework: `salina` is focused on **sequential decision making in general**. It can be used for RL (which is our main current use-case), but also for supervised learning, attention models, multi-agent learning, planning, control, cascade models, recommender systems, among other use cases.
* A `library`: salina is just a small layer on top of pytorch that encourages good practices for implementing sequential models. Accordingly, it is very simple to understand and use, while very powerful.

# Citing `salina`

Please use this bibtex if you want to cite this repository in your publications:

Link to the paper: [SaLinA: Sequential Learning of Agents](https://arxiv.org/abs/2110.07910)

```
    @misc{salina,
        author = {Ludovic Denoyer, Alfredo de la Fuente, Song Duong, Jean-Baptiste Gaya, Pierre-Alexandre Kamienny, Daniel H. Thompson},
        title = {SaLinA: Sequential Learning of Agents},
        year = {2021},
        publisher = {Arxiv},
        howpublished = {\url{https://gitHub.com/facebookresearch/salina}},
    }

```

# Papers using SaLinA:
* Learning a subspace of policies for online adaptation in Reinforcement Learning. Jean-Baptiste Gaya, Laure Soulier, Ludovic Denoyer - [Arxiv](https://arxiv.org/abs/2110.05169)


## License

`salina` is released under the MIT license. See [LICENSE](LICENSE) for additional details about it.
See also our [Terms of Use](https://opensource.facebook.com/legal/terms) and [Privacy Policy](https://opensource.facebook.com/legal/privacy).
