# SaLinA: SaLinA - A Flexible and Simple Library for Learning Sequential Agents (including Reinforcement Learning)

**Documentation**:[Read the docs](https://salina.readthedocs.io/en/latest/)

## TL;DR.

`salina` is a lightweight library extending PyTorch modules for developping **sequential decision models**.  It can be used for **Reinforcement Learning** (including model-based with differentiable environments, multi-agent RL, etc...), but also in a supervised/unsupervised learning settings (for instance for NLP, Computer Vision, etc...).
* It allows to write very complex sequential models (or policies) in few lines
* It works on multiple CPUs and GPUs

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

## News
* November 2021:
* * Complete core documentation: [Read the docs](https://salina.readthedocs.io/en/latest/)

* October 2021:
* * Week 8th of November
* * * Include [Decision Transformers](https://arxiv.org/abs/2106.01345)
* * * Include ["A Simple Way to Initialize Recurrent Networks of Rectified Linear Units"](https://arxiv.org/abs/1504.00941)
* * * **FIX: ** To avoid compatibility issues with openAI gym, the `salina/agents/gym.py` file as been renamed as `salina/agents/gyma.py`
* * Week 25th of october
* * * Updated documentation
* * * [Transformer-based Agents](salina/agents/transformers.py)
* * * [Behavioral Cloning over complete episodes](salina_examples/offline_rl/bc_on_full_episodes)
* * * * [With](salina_examples/offline_rl/bc_on_full_episodes/gym_transformer.yaml) and [without](salina_examples/offline_rl/bc_on_full_episodes/gym.yaml) transformers
* * * [PPO on Brax with Transformers](salina_examples/rl/ppo_brax_transformer)
* * Week 18th of october
* * * Release of the library
* * * Various bug fixes
* * * Add train and eval mode in the RemoteAgent and NRemoteAgent
* * * [Cleaning of the BRAX PPO Implementation](salina_examples/rl/ppo_brax) to make it similar to other implementations
* * * [Cleaning of the Behavioral Cloning implementation](salina_examples/offline_rl/bc)

## Quick Start

* Just clone the repo
* `pip install -e .`

### Documentation

* [Read the docs](https://salina.readthedocs.io/en/latest/)
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

## A note on transformers

We include both classical pytorch transformers, and [xformers-based](https://github.com/facebookresearch/xformers) implementations. On the [Behavioral Cloning](https://github.com/facebookresearch/salina/tree/main/salina_examples/offline_rl/bc_on_full_episodes) examples xformers-based models perform faster than classical pytorch implementations since they benefit from the use of sparse attention. Here is a small table describing the obtained results.

| n transitions (in the past) |             5 |             10 |            50 |           100 | All previous transitions (episode size is 1000 transitions) |
|-----------------------------|--------------:|---------------:|--------------:|--------------:|-------------------------------------------------------------|
| xformers                    | 1200K / 2.3Gb | 1000K / 2.3 Gb | 890K / 2.5 Gb | 810K / 2.1 Gb | 390K / 3.8 Gb                                               |
| pytorch                     | 460K / 4.5 Gb | 460K / 4.5 Gb  | 460K / 4.5 Gb | 460K / 4.5 Gb | 460K / 4.5 Gb                                               |

The table contains the number of transitions processed per second (during learning) and the memory used (using GPU)

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



# Papers using SaLinA:
* Learning a subspace of policies for online adaptation in Reinforcement Learning. Jean-Baptiste Gaya, Laure Soulier, Ludovic Denoyer - [Arxiv](https://arxiv.org/abs/2110.05169)
* Direct then Diffuse: Incremental Unsupervised Skill Discovery for State Covering and Goal Reaching. Pierre-Alexandre Kamienny, Jean Tarbouriech, Alessandro Lazaric, Ludovic Denoyer - [Arxiv](https://arxiv.org/abs/2110.14457)

## License

`salina` is released under the MIT license. See [LICENSE](LICENSE) for additional details about it.
See also our [Terms of Use](https://opensource.facebook.com/legal/terms) and [Privacy Policy](https://opensource.facebook.com/legal/privacy).
