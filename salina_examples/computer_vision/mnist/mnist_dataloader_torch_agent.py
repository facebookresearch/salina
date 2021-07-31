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
from torchvision import datasets

from salina import TAgent, Workspace
from salina.agents import Agents, DataLoaderAgent, ShuffledDataLoaderAgent
from salina.logger import TFLogger


class ConvNetAgent(TAgent):
    def __init__(self, input, output):
        super().__init__()
        self.input = input
        self.output = output

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

    def forward(self, t, **args):
        logits = self.net(self.get((self.input, t)))
        self.set((self.output, t), logits)


def test(agent, workspace):
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
                pred = workspace.get("pred", 0)
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


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
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

    train_dataloader = ShuffledDataLoaderAgent(train_dataset, output=("x", "y"))
    test_dataloader = DataLoaderAgent(test_dataset, output=("x", "y"))
    logger = TFLogger(
        log_dir=args.log_dir,
        hps={k: v for k, v in args.__dict__.items()},
        cache_size=10000,
        verbose=not args.no_verbose,
    )

    agent = ConvNetAgent(input="x", output="pred")
    train_agent = Agents(train_dataloader, agent)
    test_agent = Agents(test_dataloader, agent)
    train_agent.seed(0)
    test_agent.seed(1)
    optimizer = torch.optim.Adam(train_agent.parameters(), lr=args.lr)

    train_agent.to(device)
    test_agent.to(device)

    train_workspace = Workspace(batch_size=args.batch_size, time_size=1).to(device)
    test_workspace = Workspace(batch_size=args.test_batch_size, time_size=1).to(device)

    iteration = 0
    avg_loss = None
    for epoch in range(args.max_epochs):
        print(f"-- Training, Epoch {epoch+1}")

        agent.train()
        for k in range(int(len(train_dataset) / args.batch_size)):
            train_workspace = train_agent(train_workspace, t=0)
            y = train_workspace.get("y", 0)
            pred = train_workspace.get("pred", 0)
            loss = F.cross_entropy(pred, y, reduction="none")
            loss = loss.mean()

            logger.add_scalar("train/loss", loss.item(), iteration)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1

        loss, accuracy = test(test_agent, test_workspace)
        logger.add_scalar("test/loss", loss, epoch)
        logger.add_scalar("test/accuracy", accuracy, epoch)

    print("Done!")


if __name__ == "__main__":
    main()
