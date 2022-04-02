from unittest import BaseTestSuite
from omegaconf import OmegaConf
import torch
from salina import instantiate_class,get_class
import random 
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
from salina import Agent
from typing import Type
import random
import hydra 
import copy

def rand_in_range(min_val,max_val):
            return torch.rand(1).item() * (max_val - min_val) + min_val

def apply_params(acq_agent,param):
    actor = acq_agent.get_by_name('action_agent')

class PBT_Agent():
    def __init__(self,cfg) -> None:
        self.learnable_hyper_params = cfg.algorithm.learnable_hyper_parameters
        self.rl_agent = instantiate_class(cfg.algorithm.rl_agent)
        # self.rl_agent = get_class(cfg.algorithm.rl_agent)(cfg.algorithm.rl_agent)

    def update_acquisition_agent(self, acquisition_agent):
        self.rl_agent.update_acquisition_agent(acquisition_agent)

    def get_acq_actor(self):
        return self.rl_agent.get_acquisition_actor()

    def get_acq_args(self):
        return self.rl_agent.get_acquisition_args()

    def get_eval_args(self):
        return self.rl_agent.get_evaluation_args()
    
    def train(self,workspace,actor_step,total_actor_steps,logger=None):
        self.rl_agent.train(workspace,actor_step,total_actor_steps,logger)

    def resample(self,resample_prob):
        ''' Resample each parameters according to cfg range specification'''
        hyper_params = self.rl_agent.get_hyper_params()
        for param in  self.learnable_hyper_params:
            if torch.rand(1).item() < resample_prob:
                # Here, all hyper parameters are supposed to be continuous values 
                # If more diverse types are created, 
                # the cfg could also indicate a mutation function that need to be called. 
                min_val, max_val = param.range
                value = rand_in_range(min_val, max_val)
                OmegaConf.update(hyper_params,param.name,value)
        self.rl_agent.apply_hyper_params(hyper_params)

    def perturb(self,perturb_prob):
        hyper_params = self.rl_agent.get_hyper_params()
        for param in  self.learnable_hyper_params:
            if torch.rand(1).item() < perturb_prob:
                old_val = OmegaConf.select(hyper_params,param.name)
                pertubation = 0.8 if random.randint(0,1) % 2 == 0 else 1.2
                value = old_val * pertubation
                OmegaConf.update(hyper_params,param.name,value)
        self.rl_agent.apply_hyper_params(hyper_params)

class PBT_ES:
    def __init__(self,mutation_prob,selection_fct,mutation_fct,**kwargs) -> None:
        self.selection = get_class(selection_fct)
        # self.mutation_fct = get_class(mutation_fct)
        self.mutation_fct = hydra.utils.get_method(mutation_fct.classname)
        self.mutation_prob = mutation_prob

    def init(self,population):
        for pop in population: 
            self.mutation_fct(pop,self.mutation_prob)
            # pop.resample(self.mutation_prob)

    def tell(self,population: list[PBT_Agent],fitness) -> None:
        self.population,new_ids = self.selection(population,fitness)
        for i in new_ids:
            self.mutation_fct(self.population[i],self.mutation_prob)

    def ask(self) -> list[PBT_Agent]:
        return self.population

# def truncation_selection(agents : list,fitness):
#     max_pop_id = len(agents)-1
#     order = torch.argsort(-fitness)

#     worst_ids = list(range(int(max_pop_id * 0.8), max_pop_id))
#     for i in worst_ids:
#         target = random.randint(0,int(max_pop_id * 0.2))
#         agents[order[i]] = copy.deepcopy(agents[order[target]])
#     return agents, order[worst_ids]
def truncation_selection(agents : list,fitness :torch.FloatTensor):
    pop_size = len(agents)
    _,best_ids = torch.topk(fitness,int(pop_size*0.2))
    _,worst_ids = torch.topk(-fitness,int(pop_size*0.2))

    for worst in worst_ids:
        target_id = best_ids[random.randint(0,len(best_ids)-1)]
        agents[worst] = copy.deepcopy(agents[target_id])

    return agents, worst_ids


def t_test_selection(agents: list,fitness):
    raise NotImplementedError()
