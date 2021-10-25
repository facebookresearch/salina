# BC on full episodes

We provide an implementation of BC where the BC loss is computed on full episodes. It allows for instance to use Transformer-based policies over a complete episode.

## Running

Classical MLP: `python salina/salina_examples/offline_rl/bc_on_full_episodes/bc.py`
Transformer: `python salina/salina_examples/offline_rl/bc_on_full_episodes/bc.py -cn gym_transformer`


!! Not benchmarked !!
