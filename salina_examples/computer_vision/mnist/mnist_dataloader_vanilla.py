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

from salina.logger import TFLogger


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
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

    def forward(self, x):
        logits = self.net(x)
        return logits


def test(dataloader, model, device):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += F.cross_entropy(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
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
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--no-verbose", action="store_true", default=False, help="Output on console"
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

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size)

    logger = TFLogger(
        log_dir=args.log_dir,
        hps={k: v for k, v in args.__dict__.items()},
        cache_size=10000,
        verbose=not args.no_verbose,
    )

    model = ConvNet()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    iteration = 0
    for epoch in range(args.max_epochs):
        print(f"-- Training, Epoch {epoch+1}")
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = F.cross_entropy(pred, y)
            logger.add_scalar("train/loss", loss.item(), iteration)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1

        loss, accuracy = test(test_dataloader, model, device)
        logger.add_scalar("test/loss", loss, epoch)
        logger.add_scalar("test/accuracy", accuracy, epoch)

    print("Done!")


if __name__ == "__main__":
    main()
