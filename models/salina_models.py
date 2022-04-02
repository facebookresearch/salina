from salina import Agent
import torch
import torch.nn as nn 


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    m=nn.Sequential(*layers)
    return m

class DeterministicAgent(Agent):
    def __init__(self,state_dim,action_dim,hidden_layers,**kwargs):
        super().__init__()
        self.model = mlp([state_dim] + list(hidden_layers) + [action_dim],
            activation=nn.ReLU,output_activation=nn.Tanh)
        
    def forward(self,t,epsilon = 0, **kwargs):
        obs = self.get(('env/env_obs',t))
        action = self.model(obs)
        
        noise = torch.randn(*action.size(), device=action.device) * epsilon
        action = action + noise
        action = torch.clip(action, min=-1.0, max=1.0)
        self.set(('action',t),action)


class Q_Agent(Agent):
    def __init__(self,state_dim,action_dim,hidden_layers,**kwargs):
        super().__init__()
        self.model = mlp([state_dim+action_dim] + list(hidden_layers) + [1],
            activation=nn.ReLU)
    def forward(self,t,detach_actions=False, **kwargs):
        obs = self.get(('env/env_obs',t))
        action = self.get(('action',t))
        if detach_actions :
            action = action.detach()
        input = torch.cat((obs,action),dim=1)
        q_value = self.model(input)
        self.set(('q_value',t),q_value)



class ProbAgent(Agent):
    def __init__(self,state_dim,n_action,hidden_layers,**kwargs):
        super().__init__(name="prob_agent")
        self.model = mlp([state_dim] + list(hidden_layers) + [n_action],
                        activation=nn.ReLU)
        

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs", t))
        scores = self.model(observation)
        probs = torch.softmax(scores, dim=-1)
        if torch.any(torch.isnan(probs)):
            print('Here')
        self.set(("action_probs", t), probs)


class ActionAgent(Agent):
    def __init__(self,**kwargs):
        super().__init__()

    def forward(self, t, stochastic, **kwargs):
        probs = self.get(("action_probs", t))
        if stochastic:
            action = torch.distributions.Categorical(probs).sample()
        else:
            action = probs.argmax(1)

        self.set(("action", t), action)


class V_Agent(Agent):
    def __init__(self,state_dim,hidden_layers,**kwargs):
        super().__init__()
        self.model = mlp([state_dim] + list(hidden_layers) + [1],
                        activation=nn.ReLU)

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs", t))
        critic = self.model(observation).squeeze(-1)
        self.set(("critic", t), critic)