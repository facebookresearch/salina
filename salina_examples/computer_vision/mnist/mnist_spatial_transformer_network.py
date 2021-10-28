#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from salina import Agent, Workspace
from salina.agents import Agents, DataLoaderAgent, ShuffledDatasetAgent
from salina.logger import TFLogger


class STNAgent(Agent):
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

    def forward(self, **kwargs):
        for t in range(self.num_transforms + 1):
            x = self.get((self.input, t))

            if t < self.num_transforms:
                x = self.stn(x)
                self.set((self.input, t + 1), x)
            else:
                logits = self.net(x)
                self.set((self.output, 0), logits)


def test(dataloader_agent, model_agent):
    model_agent.eval()
    test_loss, correct = 0, 0
    n = 0
    dataloader_agent.reset()

    with torch.no_grad():
        while True:
            workspace = Workspace()
            dataloader_agent(workspace)
            if dataloader_agent.finished():
                break
            model_agent(workspace)
            y = workspace.get("y", 0)
            n += y.size()[0]
            pred = workspace.get("py", 0)
            loss = F.cross_entropy(pred, y, reduction="none")
            loss = loss.sum()
            test_loss += loss
            correct += pred.argmax(1).eq(y).float().sum().item()
    test_loss /= n
    correct /= n
    return test_loss, correct


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
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
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 1.0)",
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

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(
        args.data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(args.data_dir, train=False, transform=transform)

    train_agent = ShuffledDatasetAgent(
        train_dataset, batch_size=args.batch_size, output_names=("x", "y")
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )
    test_agent = DataLoaderAgent(test_dataloader, output_names=("x", "y"))
    agent = STNAgent(input="x", output="py", num_transforms=args.num_transforms)
    train_agent = Agents(train_agent, agent)
    train_agent.seed(0)
    test_agent.seed(1)

    logger = TFLogger(
        log_dir=args.log_dir,
        hps={k: v for k, v in args.__dict__.items()},
        every_n_seconds=10,
        verbose=not args.no_verbose,
    )

    optimizer = torch.optim.Adam(train_agent.parameters(), lr=args.lr)

    train_agent.to(device)
    test_agent.to(device)

    train_workspace = Workspace()

    iteration = 0
    avg_loss = None
    for epoch in range(args.max_epochs):
        print(f"-- Training, Epoch {epoch+1}")

        loss, accuracy = test(test_agent, agent)
        logger.add_scalar("test/loss", loss.item(), epoch)
        logger.add_scalar("test/accuracy", accuracy, epoch)

        agent.train()
        for k in range(int(len(train_dataset) / args.batch_size)):
            train_agent(train_workspace)
            y = train_workspace.get("y", 0)
            pred = train_workspace.get("py", 0)
            loss = F.cross_entropy(pred, y, reduction="none")
            loss = loss.mean()

            logger.add_scalar("train/loss", loss.item(), iteration)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1

    print("Done!")


if __name__ == "__main__":
    main()
