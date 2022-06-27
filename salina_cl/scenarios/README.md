# CRL Scenarios

## Description

A CRL Scenario is a sequence of training and testing tasks. Each task is associated with:
* a SaLinA agent that corresponds to the environment (with auto-reset)
* Additional informations, and particularly the `n_interactions` value that represents the number of environment steps allowed for the task. Note that this is the role of the algorithm  to take into account this maximum number of interactions. 

## Provided Scenarios

You can create new sequences of existing tasks provided in `brax`  You can also design your own scenario and add it in this folder. To use them in an experiment, simply add a yaml file `my_scenario.yaml` in the [configs/scenario](configs/scenario/) folder and use the option `scenario=my_scenario` in the command line. Here is a list of the current scenarios and a link to their yaml file:

**Forgetting scenarios:**  are designed such that a single policy tends to forget the former task when learning a new one.
* [Halfcheetah short](configs/scenario/halfcheetah/forgetting_short.yaml): hugefoot, moon, carry_stuff, rainfall
* [Halfcheetah long](configs/scenario/halfcheetah/forgetting_long.yaml): hugefoot, moon, carry_stuff, rainfall, hugefoot, moon, carry_stuff, rainfall
* [Ant short](configs/scenario/ant/forgetting_short.yaml): normal, hugefoot, rainfall, moon
* [Ant long](configs/scenario/ant/forgetting_long.yaml): normal, hugefoot, rainfall, moon, normal, hugefoot, rainfall, moon

**Transfer scenarios:**  are designed such that a single policy has more difficulties to learn a new task after having learned the former one, rather than learning it from scratch.
* [Halfcheetah short](configs/scenario/halfcheetah/transfer_short.yaml): carry_stuff_hugegravity, moon, defective_module, hugefoot_rainfall
* [Halfcheetah long](configs/scenario/halfcheetah/transfer_long.yaml): carry_stuff_hugegravity, moon, defective_module, hugefoot_rainfall, carry_stuff_hugegravity, moon, defective_module, hugefoot_rainfall
* [Ant short](configs/scenario/ant/transfer_short.yaml): disabled_first_diagonal, disabled_second_diagonal, disabled_forefeet, disabled_backfeet
* [Ant long](configs/scenario/ant/transfer_long.yaml): disabled_first_diagonal, disabled_second_diagonal, disabled_forefeet, disabled_backfeet, disabled_first_diagonal, disabled_second_diagonal, disabled_forefeet, disabled_backfeet

**Distraction scenarios:**  alternate between a normal task and a very different distraction task that disturbs the whole learning process of a single policy.
* [Halfcheetah short](configs/scenario/halfcheetah/distraction_short.yaml): normal, inverted_actions, normal, inverted_actions
* [Halfcheetah long](configs/scenario/halfcheetah/distraction_long.yaml): normal, inverted_actions, normal, inverted_actions, normal, inverted_actions, normal, inverted_actions
* [Ant short](configs/scenario/ant/distraction_short.yaml): normal, inverted_actions, normal, inverted_actions
* [Ant long](configs/scenario/ant/distraction_long.yaml): normal, inverted_actions, normal, inverted_actions, normal, inverted_actions, normal, inverted_actions

**Composability scenarios:**  present two first tasks that will be useful to learn the last one, but a very different distraction task is put at the third place to disturb this forward transfer.
* [Halfcheetah short](configs/scenario/halfcheetah/composability_short.yaml): tinyfoot, moon, carry_stuff_hugegravity, tinyfoot_moon
* [Halfcheetah long](configs/scenario/halfcheetah/composability_long.yaml): tinyfoot, moon, carry_stuff_hugegravity, tinyfoot_moon, tinyfoot, moon, carry_stuff_hugegravity, tinyfoot_moon
* [Ant short](configs/scenario/ant/composability_short.yaml): disabled_hard1, disabled_hard2, disabled_forefeet, disabled_backfeet
* [Ant long](configs/scenario/ant/composability_long.yaml): disabled_hard1, disabled_hard2, disabled_forefeet, disabled_backfeet, disabled_hard1, disabled_hard2, disabled_forefeet, disabled_backfeet