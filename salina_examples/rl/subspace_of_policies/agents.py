#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from salina import Agent, TAgent
from brax.envs.to_torch import JaxToTorchWrapper
from salina_examples.rl.subspace_of_policies.envs.brax import create_brax_env
from salina_examples.rl.subspace_of_policies.subspace import Linear, Sequential
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.categorical import Categorical
from torch.distributions.uniform import Uniform
from salina.agents.brax import BraxAgent

class CustomBraxAgent(BraxAgent):
    def _initialize_envs(self, n_envs):
        assert self._seed is not None, "[GymAgent] seeds must be specified"

        self.gym_env = create_brax_env(
            self.brax_env_name, batch_size=n_envs, seed=self._seed, **self.args
        )
        self.gym_env = JaxToTorchWrapper(self.gym_env)

#class Normalizer(TAgent):
#    def __init__(self, env):
#        super().__init__()
#        env = JaxToTorchWrapper(create_brax_env(env.env_name))
#        self.n_features = env.observation_space.shape[0]
#        self.n = None
#        self.mean = nn.Parameter(torch.zeros(self.n_features), requires_grad = False)
#        self.mean_diff = torch.zeros(self.n_features)
#        self.var = nn.Parameter(torch.ones(self.n_features), requires_grad = False)
#
#    def forward(self, t, update_normalizer=True, **kwargs):
#        input = self.get(("env/env_obs", t))
#        if update_normalizer:
#            self.update(input)
#        input = self.normalize(input)
#        self.set(("env/transformed_obs", t), input)
#
#    def update(self, x):
#        if self.n is None:
#            device = x.device
#            self.n = torch.zeros(self.n_features).to(device)
#            self.mean = self.mean.to(device)
#            self.mean_diff = self.mean_diff.to(device)
#            self.var = self.var.to(device)
#        self.n += 1.0
#        last_mean = self.mean.clone()
#        self.mean += (x - self.mean).mean(dim=0) / self.n
#        self.mean_diff += (x - last_mean).mean(dim=0) * (x - self.mean).mean(dim=0)
#        self.var = nn.Parameter(torch.clamp(self.mean_diff / self.n, min=1e-2), requires_grad = False).to(x.device)
#
#    def normalize(self, inputs):
#        obs_std = torch.sqrt(self.var)
#        return (inputs - self.mean) / obs_std
#
#    def seed(self, seed):
#        torch.manual_seed(seed)
#
class Normalizer(TAgent):
    """
    Pre-trained normalizer over Brax envs. Helps to compare models fairly.
    """
    def __init__(self, env_name):
        super().__init__()
        if "Halfcheetah" in env_name: #halfcheetah
            self.running_mean = nn.Parameter(torch.Tensor([ 6.1431e-01,  4.5919e-01,  0.0000e+00, -6.2606e-03,  0.0000e+00,
                                                            1.1327e-01, -6.0021e-02, -1.5187e-01, -2.2399e-01, -4.0081e-01,
                                                            -2.8977e-01,  6.5863e+00,  0.0000e+00, -7.2588e-03,  0.0000e+00,
                                                            1.6598e-01,  0.0000e+00,  5.8859e-02, -6.8729e-02,  8.9783e-02,
                                                            1.1471e-01, -1.9337e-01,  1.9044e-01]),requires_grad = False)
            self.std = nn.Parameter(torch.sqrt(torch.Tensor([2.7747e-02, 7.0441e-01, 5.6052e-45, 5.3661e-02, 5.6052e-45, 4.2445e-01,
                                                            3.5026e-01, 1.3651e-01, 4.2359e-01, 5.5605e-01, 9.4230e-02, 5.3188e+00,
                                                            5.6052e-45, 1.9010e+00, 5.6052e-45, 1.0593e+01, 5.6052e-45, 1.5619e+02,
                                                            2.1769e+02, 7.1641e+02, 2.4682e+02, 1.0647e+03, 9.3556e+02])),requires_grad = False)
        elif "Ant" in env_name: #ant
            self.running_mean = nn.Parameter(torch.Tensor([ 0.6112,  0.9109, -0.0210, -0.0481,  0.2029, -0.0707,  0.6687,  0.0399,
                                                            -0.6143,  0.0655, -0.6917, -0.1086,  0.6811,  4.4375, -0.2056, -0.0135,
                                                            0.0437,  0.0760,  0.0340, -0.0578, -0.1628,  0.0781,  0.2136,  0.0246,
                                                            0.1336, -0.0270, -0.0235]),requires_grad = False)
            self.std = nn.Parameter(torch.sqrt(torch.Tensor([1.4017e-02, 4.3541e-02, 2.2925e-02, 2.3364e-02, 3.6594e-02, 1.4881e-01,
                                                            4.0282e-02, 1.3600e-01, 2.6437e-02, 1.4056e-01, 4.4391e-02, 1.4609e-01,
                                                            5.3686e-02, 2.6051e+00, 1.0322e+00, 6.5472e-01, 2.7710e+00, 1.4680e+00,
                                                            6.2482e+00, 2.8523e+01, 1.3569e+01, 2.3991e+01, 1.1001e+01, 2.8217e+01,
                                                            1.6400e+01, 2.5816e+01, 1.4181e+01])),requires_grad = False)
        

    def forward(self, t, **kwargs):
        if not t is None:
            x = self.get(("env/env_obs", t))
            x = (x - self.running_mean) / self.std
            self.set(("env/transformed_obs", t), x)

class AlphaAgent(TAgent):
    def __init__(self, device, n_dim = 2, geometry = "simplex", dist = "flat"):
        super().__init__()
        self.n_dim = n_dim
        self.geometry = geometry
        self.device = device
        assert geometry in ["simplex","bezier"], "geometry must be 'simplex' or 'bezier'"
        if geometry == "bezier":
            assert n_dim == 3, "number of dimensions must be equal to 3 for Bezier subspaces"
        assert dist in ["flat","categorical"], "distribution must be 'flat' or 'categorical'" 
        if dist == "flat":
            self.dist = Dirichlet(torch.ones(n_dim)) if geometry == "simplex" else Uniform(0,1)
        else:
            self.dist = Categorical(torch.ones(n_dim))

    def forward(self, t, replay = False, **args):
        B = self.workspace.batch_size()
        alphas =  self.dist.sample(torch.Size([B])).to(self.device)

        if isinstance(self.dist,Categorical):
            alphas = F.one_hot(alphas,num_classes = self.n_dim).float()
        elif self.geometry == "bezier":
            alphas = torch.stack([(1 - alphas) ** 2, 2 * alphas * (1 - alphas), alphas ** 2],dim = 1)

        if (t > 0) and (not replay):
            done = self.get(("env/done", t)).float().unsqueeze(-1)
            alphas_old = self.get(("alphas", t-1))
            alphas =  alphas * done + alphas_old * (1 - done)
    
        self.set(("alphas", t), alphas)

class LoPAgent(TAgent):
    def __init__(self, **args):
        super().__init__()
        env = JaxToTorchWrapper(create_brax_env(args["env"].env_name))
        input_size = env.observation_space.shape[0]
        num_outputs = env.action_space.shape[0]
        hs = args["hidden_size"]
        self.n_models = args["n_models"]
        n_layers = args["n_layers"]
        hidden_layers = [Linear(self.n_models,hs,hs) if i%2==0 else nn.ReLU() for i in range(2*(n_layers - 1))] if n_layers >1 else [nn.Identity()]
        self.model = Sequential(
            Linear(self.n_models, input_size, hs),
            nn.ReLU(),
            *hidden_layers,
            Linear(self.n_models, hs, num_outputs),
        )

    def cosine_similarity(self,i,j):
        assert (i < self.n_models) and (j < self.n_models), "index higher than n_models"
        cos_sim = torch.Tensor([0.]).to(list(self.parameters())[0].device)
        n = 0
        for module in self.model:
            if isinstance(module,Linear):
                w1 = module.anchors[i].weight
                w2 = module.anchors[j].weight
                p1 = ((w1 * w2).sum() / max(((w1 ** 2).sum().sqrt() * (w2 ** 2).sum().sqrt()),1e-8)) ** 2
                b1 = module.anchors[i].bias
                b2 = module.anchors[j].bias
                p2 = ((b1 * b2).sum() / max(((b1 ** 2).sum().sqrt() * (b2 ** 2).sum().sqrt()),1e-8)) ** 2
                cos_sim += p1 + p2
                n += 2
        return cos_sim / n

    def L2_norm(self,i,j):
        assert (i < self.n_models) and (j < self.n_models), "index higher than n_models"
        L2_norm = torch.Tensor([0.]).to(list(self.parameters())[0].device)
        n = 0
        for w in self.parameters():
            L2_norm += torch.linalg.norm(w[i] - w[j])
            n += 1
        return L2_norm / n

    def forward(self, t, replay, action_std, **args):
        if replay:
            input = self.get("env/transformed_obs")
            alphas = self.get("alphas")
            mean = self.model(input,alphas)
            std = torch.ones_like(mean) * action_std + 0.000001
            dist = Normal(mean, std)
            action = self.get("action_before_tanh")
            logp_pi = dist.log_prob(action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=-1)
            self.set("action_logprobs", logp_pi)

        else:
            input = self.get(("env/transformed_obs", t))
            alphas = self.get(("alphas",t))
            with torch.no_grad():
                mean = self.model(input,alphas)
            std = torch.ones_like(mean) * action_std + 0.000001
            dist = Normal(mean, std)
            action = dist.sample() if action_std > 0 else dist.mean
            self.set(("action_before_tanh", t), action)
            logp_pi = dist.log_prob(action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=-1)
            self.set(("action_logprobs", t), logp_pi)
            action = torch.tanh(action)
            self.set(("action", t), action)

    def seed(self,seed):
        pass

class CriticAgent(Agent):
    def __init__(self, **args):
        super().__init__()
        env = JaxToTorchWrapper(create_brax_env(args["env"].env_name))
        input_size = env.observation_space.shape[0]
        alpha_size = args["alpha_size"]
        hs = args["hidden_size"]
        n_layers = args["n_layers"]
        hidden_layers = [nn.Linear(hs,hs) if i%2==0 else nn.ReLU() for i in range(2*(n_layers - 1))] if n_layers >1 else [nn.Identity()]
        self.model_critic = nn.Sequential(
            nn.Linear(input_size + alpha_size, hs),
            nn.ReLU(),
            *hidden_layers,
            nn.Linear(hs, 1),
        )

    def forward(self, **args):
        input = self.get("env/transformed_obs")
        alphas = self.get("alphas")
        x = torch.cat([input,alphas], dim=-1)
        critic = self.model_critic(x).squeeze(-1)
        self.set("critic", critic)
