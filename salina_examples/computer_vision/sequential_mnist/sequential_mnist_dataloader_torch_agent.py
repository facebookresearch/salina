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
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

from salina import Agent, Workspace
from salina.agents import Agents, DataLoaderAgent, ShuffledDatasetAgent
from salina.logger import TFLogger

class SequentialMNIST(Dataset):
    def __init__(self, data_dir, train=True, permute=False):
        
        self.data_dir = data_dir
        self.train = train
        self.permute = permute

        if self.permute:
            self.permute_ind = torch.from_numpy(np.random.permutation(784)).long()
            
        self.trans = transforms.Compose([transforms.ToTensor(),])
        self.training_dataset = datasets.MNIST(self.data_dir, train=True, 
                                               download=True, transform=self.trans)

        self.train_data = self.training_dataset.train_data
        self.train_labels = self.training_dataset.train_labels
        permutation = torch.randperm(len(self.training_dataset))
        self.X_train = self.train_data[permutation]
        self.y_train = self.train_labels[permutation]

        self.testing_dataset = datasets.MNIST(self.data_dir, train=False, 
                                              download=True, transform=self.trans)
        
        self.train_size = len(self.training_dataset)
        self.test_size = len(self.testing_dataset)

    def __len__(self):
        if self.train:
            return self.train_size
        else:
            return self.test_size

    def __getitem__(self, i):
        if self.train:
            inp = self.X_train[i]
            out = self.y_train[i]
        else:
            inp, out = self.testing_dataset[i]
        
        inp = inp.view(-1, 1)
        if self.permute:
            inp = inp[self.permute_ind]
        return inp, out

class IRNN(nn.Module):
    '''
    Model from arXiv:1504.00941v2
    A Simple Way to Initialize Recurrent Networks of
    Rectified Linear Units
    '''
    def __init__(self, input_size, hidden_size, output_size):
        super(IRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size,
                          nonlinearity='relu', batch_first=True, bias=True)
        self.output_weights = nn.Linear(hidden_size, output_size)

        # Identity Initialization 
        self.rnn.state_dict()['weight_hh_l0'].copy_(torch.eye(hidden_size))
        # Set the bias term to zero
        self.rnn.bias_ih_l0.data.fill_(0)
        self.rnn.bias_hh_l0.data.fill_(0)
        # Gaussian distribution with mean of zero and standard deviation of 0.001
        self.rnn.state_dict()['weight_ih_l0'].copy_(
            torch.randn(hidden_size, input_size)/ 1000.0)

    def forward(self, inp):
        _, hnn = self.rnn(inp.transpose(1,2).float())
        out = self.output_weights(hnn[0])
        return out    
    
class IRNNAgent(Agent):
    def __init__(self, input="x", output="py"):
        super().__init__()
        self.input = input
        self.output = output
        self.net = IRNN(784, 100, 10)
        
    def forward(self, t=0, **kwargs):
        logits = self.net(self.get((self.input, t)))
        self.set((self.output, t), logits)

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
    parser = argparse.ArgumentParser(description="PyTorch Sequential MNIST Classification Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for testing (default: 16)",
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
        default=5e-3,
        metavar="LR",
        help="learning rate (default: 5e-3)",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=1.0,
        metavar="CP",
        help="clipping threshold (default: 1.0)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--no-verbose", action="store_true", default=False, help="Output on console"
    )
    parser.add_argument(
        "--permute", action="store_true", default=False, help="Permutate pixels"
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

    train_dataset = SequentialMNIST(data_dir=args.data_dir, train=True, permute=args.permute)
    test_dataset = SequentialMNIST(data_dir=args.data_dir, train=False, permute=args.permute)

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
    agent = IRNNAgent()
    train_agent = Agents(train_agent, agent)
    train_agent.seed(0)
    test_agent.seed(1)

    logger = TFLogger(
        log_dir=args.log_dir,
        hps={k: v for k, v in args.__dict__.items()},
        every_n_seconds=10,
        verbose=not args.no_verbose,
    )
    
    optimizer = torch.optim.SGD(train_agent.parameters(), lr=args.lr)

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
            torch.nn.utils.clip_grad_norm_(train_agent.parameters(), args.clip)
            optimizer.step()
            iteration += 1

    print("Done!")


if __name__ == "__main__":
    main()
