#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from salina import TAgent, Workspace
from salina.agents import (
    Agents,
    DataLoaderAgent,
    ShuffledDataLoaderAgent,
    TemporalAgent,
)
from salina.logger import TFLogger

plt.ion()  # interactive mode

"""
NN architecture and STN visualisation taken from
https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
"""


class STNAgent(TAgent):
    def __init__(self, input, output, num_transforms):
        super().__init__()
        self.input = input
        self.output = output
        self.num_transforms = num_transforms

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(9216, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, t, **args):
        assert 0 <= t <= self.num_transforms
        x = self.get((self.input, t))

        if t < self.num_transforms:
            x = self.stn(x)
            self.set((self.input, t + 1), x)
        else:
            logits = self.net(x)
            self.set((self.output, t), logits)


def test(agent, workspace, num_transforms):
    agent.eval()
    test_loss, correct = 0, 0
    n = 0
    with torch.no_grad():
        while True:
            workspace = agent(workspace, t=0)
            mask = workspace.get("x/mask", 0)
            if mask.any():
                n += mask.float().sum().item()
                y = workspace.get("y", 0)
                pred = workspace.get("pred", num_transforms)
                loss = F.cross_entropy(pred, y, reduction="none")
                loss = (loss * mask.float()).sum().item()
                test_loss += loss
                correct += pred.argmax(1).eq(y).float().sum().item()
            else:
                break
    test_loss /= n
    correct /= n
    return test_loss, correct
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description="Spatial Transformer Network (STN) Example"
    )
    parser.add_argument(
        "--num-transforms",
        type=int,
        default=3,
        metavar="N",
        help="number of STN transformations (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10000,
        metavar="N",
        help="number of epochs to train (default: 10000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--no-verbose", action="store_true", default=False, help="Output on console"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./tmp",
        metavar="N",
        help="Directory for logging",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./.data",
        metavar="N",
        help="Directory for logging",
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cpu")
    if use_cuda:
        device = torch.device("cuda:0")
    torch.manual_seed(args.seed)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = torchvision.datasets.MNIST(
        args.data_dir, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        args.data_dir, train=False, transform=transform
    )

    train_dataloader = ShuffledDataLoaderAgent(train_dataset, output=("x", "y"))
    test_dataloader = DataLoaderAgent(test_dataset, output=("x", "y"))
    logger = TFLogger(
        log_dir=args.log_dir,
        hps={k: v for k, v in args.__dict__.items()},
        cache_size=10000,
        verbose=not args.no_verbose,
    )

    agent = STNAgent(input="x", output="pred", num_transforms=args.num_transforms)
    temporal_agent = TemporalAgent(agent)
    train_agent = Agents(train_dataloader, temporal_agent)
    test_agent = Agents(test_dataloader, temporal_agent)
    train_agent.seed(0)
    test_agent.seed(1)
    optimizer = torch.optim.Adam(train_agent.parameters(), lr=args.lr)

    train_agent.to(device)
    test_agent.to(device)

    train_workspace = Workspace(
        batch_size=args.batch_size, time_size=args.num_transforms + 1
    ).to(device)
    test_workspace = Workspace(
        batch_size=args.test_batch_size, time_size=args.num_transforms + 1
    ).to(device)

    iteration = 0
    for epoch in range(args.max_epochs):
        print(f"-- Training, Epoch {epoch+1}")

        agent.train()
        for _ in range(int(len(train_dataset) / args.batch_size)):
            train_workspace = train_agent(train_workspace, t=0)
            y = train_workspace.get("y", 0)
            pred = train_workspace.get("pred", args.num_transforms)
            loss = F.cross_entropy(pred, y, reduction="none")
            loss = loss.mean()

            logger.add_scalar("train/loss", loss.item(), iteration)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1

        loss, accuracy = test(test_agent, test_workspace, args.num_transforms)
        logger.add_scalar("test/loss", loss, epoch)
        logger.add_scalar("test/accuracy", accuracy, epoch)

    print("Done!")

    # Visualize the STN transformation on some input batch
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_dataloader.dataloader))[0].to(device)

        input_tensor = data.cpu()[: args.batch_size, :]
        in_grid = convert_image_np(torchvision.utils.make_grid(input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, args.num_transforms + 1)
        axarr[0].imshow(in_grid)
        axarr[0].set_title("Dataset Images")

        transformed_input_tensor = input_tensor
        for i in range(1, args.num_transforms + 1):
            transformed_input_tensor = agent.stn(transformed_input_tensor).cpu()
            out_grid = convert_image_np(
                torchvision.utils.make_grid(transformed_input_tensor)
            )

            axarr[i].imshow(out_grid)
            axarr[i].set_title(f"Transformed Images {i}")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
