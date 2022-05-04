import torch
from torch import autograd
import time

class ewc:
    def __init__(self,params):
        self.cfg = params
    
    def run(self,action_agent, q_agent, infos, logger):
        
        logger = logger.get_logger(type(self).__name__+str("/"))
        logger.message("Starting EWC procedure: computing Fisher matrix")
        _training_start_time=time.time()
        ## We want to compute the fisher matrix with  fisher_nb samples of  fisher_batch element each times (20 batch of 64 samples)
        replay_buffer = infos['replay_buffer']
        log_likelihoods = []
        action_agent.train()
        q_agent.train()
        action_agent = action_agent.to(self.cfg.device)
        q_agent = q_agent.to(self.cfg.device)
        for _ in range(self.cfg.iterations):
            replay_workspace = replay_buffer.get(self.cfg.batch_size).to(self.cfg.device)
            action_agent(replay_workspace, policy_update = True)
            q_agent(replay_workspace, policy_update = True)
            q1 = replay_workspace["q1"]
            q2 = replay_workspace["q2"]
            qloss = torch.min(q1,q2).mean()
            loss = - qloss.mean()
            log_likelihoods.append(loss)    ### Fisher formulas is \nabla_{\theta} log p(a|s)  (weighted by reward)
                                            ###  https://github.com/AGI-Labs/continual_rl/blob/develop/continual_rl/policies/ewc/ewc_monobeast.py#L249-L256
                                            ### and here  https://github.com/AGI-Labs/continual_rl/blob/bcf17d879e8a983340be233ff8f740c424d0f303/continual_rl/policies/impala/torchbeast/monobeast.py#L365-L379
                                            ###   in the last link:  vtrace_returns is equivalent to our "gae"
        log_likelihoods = torch.stack(log_likelihoods).unbind()
        grads = zip(*[autograd.grad(l,action_agent[0].parameters(),retain_graph=(i < len(log_likelihoods))) for i, l in enumerate(log_likelihoods, 1)])  
        grads = [torch.stack(grad) for grad in grads]
        fisher_diagonals = [  (grad ** 2).mean(0) for grad in grads]
        action_agent[0].register_and_consolidate(fisher_diagonals)
        
        r={"n_epochs":self.cfg.iterations,"training_time":time.time() - _training_start_time,"n_interactions":0}
        return r, action_agent.to('cpu'), q_agent.to('cpu')
      