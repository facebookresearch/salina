# Offline RL

We provide implementations of offline/imitation learning algorithms. One of **the big interest** of `salina` is that, since trajectories are stored in a `Workspace`, it is very easy to adapt online RL code (where `Worspace`s are built by executing agents) to offline RL algortihms where `Workspace`s are built from a dataset instead (we provide [functions](__init__.py) to load a dataset to a `Workspace`)

* [Behavioral Cloning](bc/): A implementation of BC

Examples are using sing [d4rl](https://github.com/rail-berkeley/d4rl) and [d4rl_atari](https://github.com/takuseno/d4rl-atari)

To play with offline_rl methods, please install `d4rl` and `d4rl_atari` packages.
