# MNIST Classification

We ilustrate how `salina` can be used for classical computer vision problem. We provide three implementations:
* A [classical MNIST classifier](mnist_dataloader_vanilla.py) in pytorch
* The same model implemented [using salina](mnist_dataloader_salina.py) showing how Data loading can be handled by an `Agent`
* A [cascade model](mnist_spatial_transformer_network.py) (cascade of spatial transformations) as an example of sequential model in `salina` extending classical CV methods.
