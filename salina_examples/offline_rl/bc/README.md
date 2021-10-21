# BC

We propose  an implementation of the BC algorithm for offline RL using a MSE Loss

* `bc.py` is for continuous actions with MSE loss -- using [d4rl](https://github.com/rail-berkeley/d4rl)
* `bc_discrete.py` is for dsicrete actions with cross-entropy -- using [d4rl_atari](https://github.com/takuseno/d4rl-atari)

Note on openAI gym: The gym API is changing a lot... Don't hesitate to use `pip  install 'gym[all]==0.15.3'` to ensure compatibility with D4RL
