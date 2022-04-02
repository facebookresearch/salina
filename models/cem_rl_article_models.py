from copy import deepcopy
import torch.nn as nn
import torch 
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

## All these agents have been taken from 
# https://github.com/apourchot/CEM-RL


if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor
USE_CUDA =torch.cuda.is_available()


def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()



class RLNN(nn.Module):

    def __init__(self, state_dim, action_dim, max_action,**kwargs):
        super(RLNN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

    def set_params(self, params):
        """
        Set the params of the network to the given parameters
        """
        cpt = 0
        for param in self.parameters():
            tmp = np.product(param.size())

            if torch.cuda.is_available():
                param.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()).cuda())
            else:
                param.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()))
            cpt += tmp

    def get_params(self):
        """
        Returns parameters of the actor
        """
        return deepcopy(np.hstack([to_numpy(v).flatten() for v in
                                   self.parameters()]))

    def get_grads(self):
        """
        Returns the current gradient
        """
        return deepcopy(np.hstack([to_numpy(v.grad).flatten() for v in self.parameters()]))

    def get_size(self):
        """
        Returns the number of parameters of the network
        """
        return self.get_params().shape[0]

    def load_model(self, filename, net_name):
        """
        Loads the model
        """
        if filename is None:
            return

        self.load_state_dict(
            torch.load('{}/{}.pkl'.format(filename, net_name),
                       map_location=lambda storage, loc: storage)
        )

    def save_model(self, output, net_name):
        """
        Saves the model
        """
        torch.save(
            self.state_dict(),
            '{}/{}.pkl'.format(output, net_name)
        )


class Actor(RLNN):

    def __init__(self, state_dim, action_dim, max_action, layer_norm=False, init=True,**kwargs):
        super(Actor, self).__init__(state_dim, action_dim, max_action)
        
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        if layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)
        self.layer_norm = layer_norm

    def forward(self, x):

        if not self.layer_norm:
            x = torch.tanh(self.l1(x))
            x = torch.tanh(self.l2(x))
            x = self.max_action * torch.tanh(self.l3(x))

        else:
            x = torch.tanh(self.n1(self.l1(x)))
            x = torch.tanh(self.n2(self.l2(x)))
            x = self.max_action * torch.tanh(self.l3(x))

        return x


class Critic(RLNN):
    def __init__(self, state_dim, action_dim, layer_norm=False,**kwargs):
        super(Critic, self).__init__(state_dim, action_dim, 1)

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        if layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)
        self.layer_norm = layer_norm

    def forward(self, x, u):

        if not self.layer_norm:
            x = torch.leaky_relu(self.l1(torch.cat([x, u], 1)))
            x = torch.leaky_relu(self.l2(x))
            x = self.l3(x)

        else:
            x = torch.leaky_relu(self.n1(self.l1(torch.cat([x, u], 1))))
            x = torch.leaky_relu(self.n2(self.l2(x)))
            x = self.l3(x)

        return x


class CriticTD3(RLNN):
    def __init__(self, state_dim, action_dim, layer_norm=False,**kwargs):
        super(CriticTD3, self).__init__(state_dim, action_dim, 1)

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        if layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

        if layer_norm:
            self.n4 = nn.LayerNorm(400)
            self.n5 = nn.LayerNorm(300)
        self.layer_norm = layer_norm

    def forward(self, x, u):

        if not self.layer_norm:
            x1 = torch.leaky_relu(self.l1(torch.cat([x, u], 1)))
            x1 = torch.leaky_relu(self.l2(x1))
            x1 = self.l3(x1)

        else:
            x1 = torch.leaky_relu(self.n1(self.l1(torch.cat([x, u], 1))))
            x1 = torch.leaky_relu(self.n2(self.l2(x1)))
            x1 = self.l3(x1)

        if not self.layer_norm:
            x2 = torch.leaky_relu(self.l4(torch.cat([x, u], 1)))
            x2 = torch.leaky_relu(self.l5(x2))
            x2 = self.l6(x2)

        else:
            x2 = torch.leaky_relu(self.n4(self.l4(torch.cat([x, u], 1))))
            x2 = torch.leaky_relu(self.n5(self.l5(x2)))
            x2 = self.l6(x2)

        return x1, x2

class Actor(RLNN):

    def __init__(self, state_dim, action_dim, max_action, args,**kwargs):
        super(Actor, self).__init__(state_dim, action_dim, max_action)

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        if args.layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)
        self.layer_norm = args.layer_norm

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=args.actor_lr)
        # self.tau = args.tau
        # self.discount = args.discount
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

    def forward(self, x):

        if not self.layer_norm:
            x = torch.tanh(self.l1(x))
            x = torch.tanh(self.l2(x))
            x = self.max_action * torch.tanh(self.l3(x))

        else:
            x = torch.tanh(self.n1(self.l1(x)))
            x = torch.tanh(self.n2(self.l2(x)))
            x = self.max_action * torch.tanh(self.l3(x))

        return x

    def update(self, memory, batch_size, critic, actor_t):

        # Sample replay buffer
        states, _, _, _, _ = memory.sample(batch_size)

        # Compute actor loss
        if self.args.use_td3:
            actor_loss = -critic(states, self(states))[0].mean()
        else:
            actor_loss = -critic(states, self(states)).mean()

        # Optimize the actor
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), actor_t.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)


class Critic(RLNN):
    def __init__(self, state_dim, action_dim, max_action, args,**kwargs):
        super(Critic, self).__init__(state_dim, action_dim, 1)

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        if args.layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)

        self.layer_norm = args.layer_norm
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=args.critic_lr)
        # self.tau = args.tau
        # self.discount = args.discount
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

    def forward(self, x, u):

        if not self.layer_norm:
            x = F.leaky_relu(self.l1(torch.cat([x, u], 1)))
            x = F.leaky_relu(self.l2(x))
            x = self.l3(x)

        else:
            x = F.leaky_relu(self.n1(self.l1(torch.cat([x, u], 1))))
            x = F.leaky_relu(self.n2(self.l2(x)))
            x = self.l3(x)

        return x

    def update(self, memory, batch_size, actor_t, critic_t):

        # Sample replay buffer
        states, n_states, actions, rewards, dones = memory.sample(batch_size)

        # Q target = reward + discount * Q(next_state, pi(next_state))
        with torch.no_grad():
            target_Q = critic_t(n_states, actor_t(n_states))
            target_Q = rewards + (1 - dones) * self.discount * target_Q

        # Get current Q estimate
        current_Q = self(states, actions)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        # Optimize the critic
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), critic_t.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)


class CriticTD3(RLNN):
    def __init__(self, state_dim, action_dim, max_action, args,**kwargs):
        super(CriticTD3, self).__init__(state_dim, action_dim, 1)

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        if args.layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

        if args.layer_norm:
            self.n4 = nn.LayerNorm(400)
            self.n5 = nn.LayerNorm(300)

        self.layer_norm = args.layer_norm
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=args.critic_lr)
        # self.tau = args.tau
        # self.discount = args.discount
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip

    def forward(self, x, u):

        if not self.layer_norm:
            x1 = F.leaky_relu(self.l1(torch.cat([x, u], 1)))
            x1 = F.leaky_relu(self.l2(x1))
            x1 = self.l3(x1)

        else:
            x1 = F.leaky_relu(self.n1(self.l1(torch.cat([x, u], 1))))
            x1 = F.leaky_relu(self.n2(self.l2(x1)))
            x1 = self.l3(x1)

        if not self.layer_norm:
            x2 = F.leaky_relu(self.l4(torch.cat([x, u], 1)))
            x2 = F.leaky_relu(self.l5(x2))
            x2 = self.l6(x2)

        else:
            x2 = F.leaky_relu(self.n4(self.l4(torch.cat([x, u], 1))))
            x2 = F.leaky_relu(self.n5(self.l5(x2)))
            x2 = self.l6(x2)

        return x1, x2

    def update(self, memory, batch_size, actor_t, critic_t):

        # Sample replay buffer
        states, n_states, actions, rewards, dones = memory.sample(batch_size)

        # Select action according to policy and add clipped noise
        noise = np.clip(np.random.normal(0, self.policy_noise, size=(
            batch_size, self.action_dim)), -self.noise_clip, self.noise_clip)
        n_actions = actor_t(n_states) + FloatTensor(noise)
        n_actions = n_actions.clamp(-self.max_action, self.max_action)

        # Q target = reward + discount * min_i(Qi(next_state, pi(next_state)))
        with torch.no_grad():
            target_Q1, target_Q2 = critic_t(n_states, n_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self(states, actions)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + \
            nn.MSELoss()(current_Q2, target_Q)

        # Optimize the critic
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), critic_t.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)



    