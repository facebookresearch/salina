# SaLinA: SaLinA - A Flexible and Simple Library for Learning Sequential Agents (including Reinforcement Learning)

## TL;DR

`salina` is a lightweight library extending PyTorch modules for developping **sequential decision models**.  It can be used for **Reinforcement Learning** (including model-based with differentiable environments, multi-agent RL, ...), but also in a supervised/unsupervised learning settings (for instance for NLP, Computer Vision, etc..).
* It allows to write very complex sequential models (or policies) in few lines
* It works on multiple CPUs and GPUs

## Quick Start

* Just clone the repo

### Documentation

* [Principles of the library](md_docs/)
* [Examples and Algorithms](salina_examples/)
* [Tutorial through multiple A2C implementations](salina_examples/rl/a2c)
* [Reinforcement Learning Benchmark](salina_examples/rl/BENCHMARK.md)
* Video Tutorials:


**For development, set up [pre-commit](https://pre-commit.com) hooks:**

* Run `pip install pre-commit`
    * or `conda install -c conda-forge pre-commit`
    * or `brew install pre-commit`
* In the top directory of the repo, run `pre-commit install` to set up the git hook scripts
* Now `pre-commit` will run automatically on `git commit`!
* Currently isort, black and blacken-docs are used, in that order

## Organization of the repo

* [salina](salina/) is the core library
* * [salina.agents](salina/agents/) is the catalog of agents (the same than `torch.nn` but for agents)
* [salina_examples](salina_examples/) provide many examples (in different domains)

## Dependencies

`salina` is making use of `pytorch`, `hydra` for configuring experiments, and of `gym` for reinforcement learning algorithms.

## Note on the Logger

We provide a simple Logger that logs in both tensorboard format, but also as pickle files that can be re-read to make tables and figures. See [logger](salina/logger.py). This logger can be easily replaced by any other logger.

# Description

**Sequential Decision Making is much more than Reinforcement learning**

* Sequential Decision Making is about interactions:
 * Interaction with data (e.g attention-models, decision tree, cascade models, active sensing, active learning, recommendation, etc….)
 * Interaction with an environment (e.g games, control)
 * Interaction with humans (e.g recommender systems, dialog systems, health systems, …)
 * Interaction with a model of the world (e.g simulation)
 * Interaction between multiple entities (e.g multi-agent RL)


## What `salina` is

* A sandbox for developping sequential models at scale.

* A small (300 hundred lines) 'core' code that defines everything you will use to implement `agents` involved in sequential decision learning systems.
* * It is easy to understand and to use since it keeps the main principles of pytorch, just extending `nn.Module` to `Agent` that handle tthe temporal dimension.
*
 A set of **agents** that can be combined (like pytorch modules) to obtain complex behaviors

* A set of references implementations and examples in different domains **Reinforcement learning**, **Imitation Learning**, **Computer Vision**, **Language Modelling**, ... (nore to come..)

## What `salina` is not

* Yet another reinforcement learning framework: `salina` is focused on **sequential decision making in general**. It can be used for RL (which is our main current use-case), but also for supervised learning, attention models, multi-agent learning, planning, control, cascade models, recommender systems,...
* A `library`: salina is just a small layer on top of pytorch that encourages good practices for implementing sequential models. It thus very simple to understand and to use, but very powerful.

# Current Contributors (alphabetical order)

Alfredo de la Fuente, Ludovic Denoyer, Song Duong, Jean-Baptiste Gaya, Daniel H. Thompson


# Citing `salina`

Please use this bibtex if you want to cite this repository in your publications:

Link to the paper: [SaLinA: Sequential Learning of Agents](salina_examples/rl/BENCHMARK.md)

```
    @misc{salina,
        author = {Ludovic Denoyer, Alfredo de la Fuente, Song Duong, Jean-Baptiste Gaya, Pierre-Alexandre Kamienny, Daniel H. Thompson},
        title = {SaLinA: Sequential Learning of Agents},
        year = {2021},
        publisher = {Arxiv},
        howpublished = {\url{https://gitHub.com/facebookresearch/salina}},
    }

```

## License

`salina` is released under the MIT license. See [LICENSE](LICENSE) for additional details about it.
See also our [Terms of Use](https://opensource.facebook.com/legal/terms) and [Privacy Policy](https://opensource.facebook.com/legal/privacy).
