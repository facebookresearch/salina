import d4rl
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from salina import Workspace
from salina.rl.replay_buffer import ReplayBuffer


def d4rl_dataset_to_replaybuffer(dataset, time_size, proportion):
    with torch.no_grad():
        if isinstance(dataset["observations"], list):
            obs = np.array(dataset["observations"])
        else:
            obs = dataset["observations"]
        T = obs.shape[0]
        T = int(T * proportion)
        print("\tUsing ", T, " transitions...")
        print("\tConverting observations to torch.Tensor...")
        _obs = torch.tensor(obs[:T]).float()
        _done = torch.tensor(dataset["terminals"][:T]).bool()
        _action = torch.tensor(dataset["actions"][:T])
        if np.issubdtype(dataset["actions"][0], np.integer):
            _action = torch.tensor(dataset["actions"][:T]).long()
        else:
            _action = torch.tensor(dataset["actions"][:T]).float()
        _reward = torch.tensor(dataset["rewards"][:T]).float()
        _reward[_done] = 0.0
        _reward = torch.cat([torch.tensor([0.0]), _reward], dim=0)[:-1]

        buffer_size = T - time_size - 1
        print("\tCreating replay buffer")
        replay_buffer = ReplayBuffer(buffer_size)
        for t in tqdm(range(0, T - time_size)):
            workspace = Workspace()
            workspace.set_full("env/env_obs", _obs[t : t + time_size].unsqueeze(1))
            workspace.set_full("env/done", _done[t : t + time_size].unsqueeze(1))
            workspace.set_full("action", _action[t : t + time_size].unsqueeze(1))
            workspace.set_full("env/reward", _reward[t : t + time_size].unsqueeze(1))
            replay_buffer.put(workspace)
    return replay_buffer


def d4rl_dataset_to_workspaces(dataset, time_size, proportion):
    with torch.no_grad():
        if isinstance(dataset["observations"], list):
            obs = np.array(dataset["observations"])
        else:
            obs = dataset["observations"]
        T = obs.shape[0]
        T = int(T * proportion)
        print("Using ", T, " transitions...")
        print("Converting observations to torch.Tensor...")
        _obs = torch.tensor(obs[:T]).float()
        _done = torch.tensor(dataset["terminals"][:T]).bool()
        _action = torch.tensor(dataset["actions"][:T])
        if _action.dtype == torch.float64:
            _action = _action.float()
        _reward = torch.tensor(dataset["rewards"][:T]).float()
        _reward[_done] = 0.0
        _reward = torch.cat([torch.tensor([0.0]), _reward], dim=0)[:-1]

        for t in range(0, T - time_size):
            workspace = Workspace()
            workspace.set_full("env/env_obs", _obs[t : t + time_size].unsqueeze(1))
            workspace.set_full("env/done", _done[t : t + time_size].unsqueeze(1))
            workspace.set_full("action", _action[t : t + time_size].unsqueeze(1))
            workspace.set_full("env/reward", _reward[t : t + time_size].unsqueeze(1))
            yield workspace
