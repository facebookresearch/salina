import torch
from torch import autograd
import torch.autograd.functional as TAF
import time
import copy


class ewc:
    def __init__(self,params):
        self.cfg = params
    
    def run(self,action_agent, q_agent, infos, logger, task_id):
        
        logger = logger.get_logger(type(self).__name__+str("/"))
        logger.message("Starting EWC procedure: computing Fisher matrix")
        _training_start_time=time.time()
        ## We want to compute the fisher matrix with  fisher_nb samples of  fisher_batch element each times
        
        
        action_agent.train()
        action_agent = action_agent.to(self.cfg.device)
        policy = action_agent[-1].model[min(len(action_agent[-1].model)-1,task_id)]
        output_dim = action_agent[-1].output_dimension
        policy.zero_grad()
        reg_weights = [copy.deepcopy(param.grad) for param in policy.parameters()]
        batch_obs = infos['replay_buffer'].get(self.cfg.n_samples).to(self.cfg.device)["env/env_obs"][0]

        #We do it sample by sample
        for obs in batch_obs:
            #gathering mus grad
            grads_mu = []
            for i in range(output_dim // 2):
                mu_i = policy(obs)[i]
                mu_i.backward()
                grads_mu.append([copy.deepcopy(param.grad) for param in policy.parameters()])
                policy.zero_grad()
            
            #gathering std grad      
            grads_std = []
            stds = []
            for i in range(output_dim // 2,output_dim):
                std_i = policy(obs)[i]
                std_i = torch.clip(std_i, min=-20., max=2.)
                std_i = std_i.exp()
                std_i.backward()
                grads_std.append([copy.deepcopy(param.grad) for param in policy.parameters()])
                stds.append(std_i)
                policy.zero_grad()

            #calculating fisher matrix
            fisher = [copy.deepcopy(param.grad) for param in policy.parameters()]
            for grad_mu, grad_std, std in zip(grads_mu,grads_std, stds): #for each output scalar
                for i in range(len(fisher)): #for each policy parameter
                    fisher[i] += (grad_mu[i] ** 2 + 2 * grad_std[i] ** 2) / (std ** 2 + 1e-6) #closed form, see page 21 in https://arxiv.org/pdf/2105.10919.pdf
            
            #averaging over batch dimension
            for i in range(len(reg_weights)):
                fisher[i] = torch.clamp(fisher[i],min=1e-5) #clipping from below, see https://github.com/awarelab/continual_world/blob/main/continualworld/methods/ewc.py#L66
                reg_weights[i] += fisher[i] / self.cfg.n_samples

        #register new regluarisation weights for next task
        action_agent[-1].register_and_consolidate(reg_weights)
        
        r={"n_epochs":self.cfg.n_samples,"training_time":time.time() - _training_start_time,"n_interactions":0}
        return r, action_agent.to('cpu'), q_agent.to('cpu')
      