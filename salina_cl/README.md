# salina_cl

This package aims at providing Continual RL (CRL) algorithms in the SaLinA Library. It is a development branch which may change drastically in the next months

## Get started
Different experiments are already available in [configs](configs/) as yaml files. Once you chose `my_config` experiment file, simply run `python run.py -cn=my_config`.You may want to try other scenarios configs from [configs/scenario](configs/scenario/). In this case, simply add `scenario=my_scenario` as an argument.

## How it works
The `core.py` file contains the building blocks of this framework. Each experiment consists in running a `Model` over a `Scenario`, i.e. a sequence of train and test `Task`. The models are learning procedures that use salina agents to interact with the tasks and learn from them through an algorithm.

## Organization
* [models](models/) contains generic learning procedures (fine-tuning, training from scratch, incremental models,...)
* [scenarios](scenarios/) contains CRL scenarios i.e sequence of train and test tasks
* [algorithms](algorithms/) contains different CRL algorithms (ppo, sac, td3)
* [agents](agents/) contains salina agents (env, policy, critic, ...)