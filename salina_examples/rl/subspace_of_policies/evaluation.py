#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import pandas as pd
import hydra
import os
import torch
from salina import Workspace
from salina.agents import Agents, TemporalAgent
from salina import Workspace, instantiate_class
from salina_examples.rl.subspace_of_policies.agents import Normalizer, CustomBraxAgent
from salina.agents import Agents, TemporalAgent
from salina_examples.rl.subspace_of_policies.envs.halfcheetah import env_cfgs
from torch.distributions.dirichlet import Dirichlet

def generate_k_shot_points(n,dim,geometry,dist):
    if dist == "categorical":
        vals = torch.eye(dim)
    elif dim in [2,3]:
        if geometry == "bezier":
            vals =   [[round((1-x/(n-1))**2,2),round(2*(x/(n-1))*(1-x/(n-1)),2),round((x/(n-1))**2,2)] for x in range(n)]
        elif dim == 2:
            vals =   [[round(x/(n-1),2),round(1-x/(n-1),2)] for x in range(n)]
        elif dim == 3:
            vals =  Dirichlet(torch.ones(dim)).sample(torch.Size([n]))
            #vals =  [[round(x/(n-1),2),round(y/(n-1),2),round(1-(x+y)/(n-1),2)] for x,y in product(range(n),range(n)) if (x+y)/(n-1)<=1.]
        vals = torch.Tensor(vals)
    else:
        vals =  Dirichlet(torch.ones(dim)).sample(torch.Size([n]))
    return vals

def _generate_mask(done):
    T, B = done.size()
    done = done.detach().clone()
    done[-1] = True
    index_done = done.float().argmax(0)
    assert index_done.size()[0] == B
    arange = torch.arange(T, device=done.device).unsqueeze(-1).repeat(1, B)
    index_done = index_done.unsqueeze(0).repeat(T, 1)
    mask = arange.le(index_done)
    return mask

def run_eval(cfg):
    dfs = []
    model_normalizer = torch.load(cfg.path+"/normalizer")
    model_policy = torch.load(cfg.path+"/policy")
    for key in list(model_policy.keys()):
        if "agent." in key:
            model_policy[key.replace('agent.','')] = model_policy.pop(key)
    cfg.model.policy.n_models = model_policy[list(model_policy)[0]].shape[0]
    cfg.model.policy.hidden_size = model_policy[list(model_policy)[0]].shape[1]
    cfg.model.policy.n_layers = int((len(model_policy) - 2) // 2)
    if cfg.model.distribution == "categorical":
        cfg.k_shot = cfg.model.policy.n_models

    policy_agent = instantiate_class(cfg.model.policy).to(cfg.device)
    normalizer_agent = Normalizer(cfg.env).to(cfg.device)
    normalizer_agent.load_state_dict(model_normalizer)
    policy_agent.load_state_dict(model_policy)

    alpha = generate_k_shot_points(cfg.k_shot,cfg.model.policy.n_models,cfg.model.geometry,cfg.model.distribution)

    for env_name, env_spec in env_cfgs.items():
        cfg.env.env_spec.torso = env_spec["torso"]
        cfg.env.env_spec.thig = env_spec["thig"]
        cfg.env.env_spec.shin = env_spec["shin"]
        cfg.env.env_spec.foot = env_spec["foot"]
        cfg.env.env_spec.gravity = env_spec["gravity"]
        cfg.env.env_spec.friction = env_spec["friction"]
        env_agent = CustomBraxAgent(cfg.k_shot,**cfg.env)
        agent = TemporalAgent(Agents(env_agent, normalizer_agent, policy_agent))
        agent.seed(cfg.seed)
        workspace = Workspace()
        alphas = alpha.unsqueeze(0).repeat(cfg.env.episode_length,1,1).to(cfg.device)
        workspace.set_full("alphas",alphas)
        agent(workspace, t = 0, n_steps = cfg.env.episode_length, replay=False, update_normalizer = False, action_std=0.0)
        reward, done, alphas = workspace["env/reward", "env/done","alphas"]
        mask = _generate_mask(done)
        reward = (reward * mask).sum(0)
        reward = reward.reshape(cfg.k_shot)

        print("\n--- For",env_name,":")
        for k,r in enumerate(reward):
            print("k =",k+1,"\t:",round(r.item(),0))
        df = {"alpha":alphas[0,:,0].cpu(),"rewards":reward.reshape(-1).cpu()}
        df["env_name"] = env_name
        dfs.append(pd.DataFrame(df))
        
    dfs = pd.concat(dfs)
    dfs.to_pickle(cfg.path+"/eval.pkl")

@hydra.main(config_path=".", config_name="evaluation.yaml")
def main(cfg):
    import torch.multiprocessing as mp
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        v = torch.ones(1, device="cuda:0")
    run_eval(cfg)

if __name__ == "__main__":
    main()
