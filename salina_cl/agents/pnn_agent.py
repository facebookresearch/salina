#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.normal import Normal
from salina_cl.core import CRLAgent, CRLAgents
from salina_cl.agents.single_agents import  Normalizer

def PNNAgent(input_dimension,output_dimension, n_layers, hidden_size):
    return CRLAgents(Normalizer(input_dimension),PNAction(input_dimension,output_dimension, n_layers, hidden_size))
   
class PNAction(CRLAgent):
    """
    just a NN.
    """
    def __init__(self, input_dimension,output_dimension, n_layers, hidden_size, input_name = "env/normalized_env_obs", output_name = "env/transformed_env_obs"):
        super().__init__()
        self.iname = input_name
        self.oname = output_name
        self.task_id = 0
            
        self.input_size = input_dimension[0]
        self.num_outputs = output_dimension[0]
        self.hs = hidden_size
        self.n_layers = n_layers
      
       
        self.increment=0
        self.activation=nn.ReLU()
        self.columns=nn.ModuleList()
        self.laterals=nn.ModuleList()
        
        self.create_columns()
        
        
  
        
    def create_columns(self):
        print('Creating a new columns and its lateral')
        ## we create a column and its lateral connection
        
        backbone_list = ([nn.Linear(self.hs, self.hs) for i in range(self.n_layers-1)] if self.n_layers > 1 else [nn.Identity()])
        model=nn.ModuleList([nn.Linear(self.input_size,self.hs)]) ## input
        for i in range(len(backbone_list)):
            model.append(backbone_list[i])
        model.append(nn.Linear(self.hs, self.num_outputs)) ## ouput
        # import ipdb;ipdb.set_trace()
        self.columns.append(model)
        
        
        
        ## The tricky part is the lateral layers: for the 2nd task you have only one lateral from task 1 to task 2 
        ## BUT for task 3 you have laterals connection from task 1 and 2 to task 3's columns (this is managed by self.increment that tells how many laterals connection we need)
        laterals_weights_carrier=nn.ModuleList()
        print('valeur de self increment {}'.format(self.increment))
        for _ in range(self.increment):
            lateral_model=nn.ModuleList([nn.Linear(self.input_size,self.hs)]) ## input
            for i in range(len(backbone_list)):
                lateral_model.append(backbone_list[i])
            lateral_model.append(nn.Linear(self.hs, self.num_outputs)) ## ouput
            laterals_weights_carrier.append(lateral_model)
        # import ipdb;ipdb.set_trace()
        
      
        self.laterals.append(laterals_weights_carrier)
      
        
        
    def forward(self, t=None, action_std=0.0, **kwargs):
        model_id = min(self.task_id,len(self.columns) - 1)
     
        
        if t is None:
            x = self.get(self.iname)
        
        else:
            x = self.get((self.iname, t))

        
        ### we precompute parents output (1st and 2nd columns if we're dealing with task 3 for instance)
        parent_output=[]
        
        for parent_index in range(model_id):
            intermediate_parent_output=[]
            for depth in range(len(self.columns[parent_index])):
                if depth==0:
                    output=self.columns[parent_index][depth](x)
                else:
                    output=self.columns[parent_index][depth](output)
                if depth<len(self.columns[parent_index])-2:
                    output=self.activation(output)
                intermediate_parent_output.append(output)
            parent_output.append(intermediate_parent_output)
     
        for depth in range(len(self.columns[0])):
                if model_id==0:
                    x=self.columns[model_id][depth](x)
                    if depth<len(self.columns[0])-1:
                            x=self.activation(x)
                else: ## model_id >0
                    
                    column_out=self.columns[model_id][depth](x)
                    
                    if depth>0:
                        if len(parent_output)>0:
                            aux_var=torch.zeros_like(column_out)
                            ## We take the parents output (intermediate) and feed it to the laterals connection (self.laterals)
                            for parent_id in range(model_id):
                                    intermediate=parent_output[parent_id][depth-1] ## output of parents
                                    aux_var+=self.laterals[model_id][parent_id][depth](intermediate)
                            ## theoretically they have a scalar "a" to reduce dimensions
                            ## but in practice in all the githubs I found they don t use it
                            ## https://github.com/chengtan9907/ProgressiveNeuralNetworks-Pytorch/blob/master/progressive.py#L31-L36
                            ## https://github.com/sumanvid97/progressive_nets_for_multitask_rl/blob/master/pnn.py#L52-L68
                           
                         
                            output=column_out+aux_var
                        
                    else:
                        output=column_out
                   
                    if depth<len(self.columns[model_id])-1:
                            x=self.activation(output)
                    else:
                        x=output
       
        mean=x
        if t is None:
            var = torch.ones_like(mean) * action_std + 0.000001
            dist = Normal(mean, var)
            action = self.get("action_before_tanh")
            logp_pi = dist.log_prob(action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=-1)
            self.set("action_logprobs", logp_pi)
          
        else:
            var = torch.ones_like(mean) * action_std + 0.000001
            dist = Normal(mean, var)
            action = dist.sample() if action_std > 0 else dist.mean
            self.set(("action_before_tanh", t), action)
            logp_pi = dist.log_prob(action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=-1)
            self.set(("action_logprobs", t), logp_pi)
            action = torch.tanh(action)
            self.set(("action", t), action)
            
    def set_task(self,task_id = None):
        if task_id is None:
            self.increment+=1
            
    
            if len(self.columns)>0: ## I have doubt about this: we are freezing the previous columns weights 
               
                for param in self.columns[-1].parameters():
                    param.requires_grad = False
               
            self.create_columns()
            print('len of column is now {}'.format(len(self.columns)))
                  
                   
        else:
            
            self.task_id=task_id