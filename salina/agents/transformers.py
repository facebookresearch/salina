#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


from salina import Agent
from salina.agents import Agents
import torch.nn as nn
from salina import Workspace, get_arguments, get_class, instantiate_class
import torch

def _layer_norm(module,x):
    if len(x.size())==2:
        return module(x)
    else:
        s=x.size()
        x=x.reshape(s[0]*s[1],s[2])
        x=module(x)
        return x.reshape(*s)

class Id(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x

class TransformerBlockAgent(Agent):
    def __init__(self,embedding_size,n_heads,n_steps=None,input_name="attn_in/x",output_name="attn_out/x",use_layer_norm=False):
        super().__init__()
        self.n_steps=n_steps
        self.multiheadattention=nn.MultiheadAttention(embedding_size,n_heads)
        self.input_name=input_name
        self.output_name=output_name
        if use_layer_norm:
            self.ln1=nn.LayerNorm(embedding_size)
            self.ln2=nn.LayerNorm(embedding_size)
        else:
            self.ln1=Id()
            self.ln2=Id()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.GELU(),
            nn.Linear(4 * embedding_size, embedding_size),
            #nn.Dropout(config.resid_pdrop),
        )


    def forward(self,t=None,**kwargs):
        if not t is None:
            if self.n_steps is None:
                tokens=self.get(self.input_name)[:t+1]
            else:
                from_time=max(0,t+1-self.n_steps)
                to_time=t+1
                tokens=self.get_time_truncated(self.input_name,from_time,to_time)
            ln_tokens=_layer_norm(self.ln1,tokens)
            previous_tokens=ln_tokens[:]
            keys=previous_tokens
            values=previous_tokens
            queries=ln_tokens[-1].unsqueeze(0)
            attn_output, attn_output_weights = self.multiheadattention(queries, keys, values)
            attn_output=attn_output.squeeze(0)
            x=tokens[-1]+attn_output
            nx=_layer_norm(self.ln2,x)
            x=x+self.mlp(nx)
            self.set((self.output_name,t),x)
        else:
            tokens=self.get(self.input_name)
            tokens=_layer_norm(self.ln1,tokens)
            keys=tokens
            values=tokens
            queries=tokens
            T=queries.size()[0]
            if (self.n_steps is None):
                attn_mask=torch.triu(torch.ones(T,T), diagonal=1).bool().to(keys.device)
            else:
                attn_mask=torch.triu(torch.ones(T,T), diagonal=1).to(keys.device)
                attn_mask2=torch.triu(torch.ones(T,T) , diagonal=1-self.n_steps).to(keys.device)
                attn_mask=attn_mask+1-attn_mask2
                attn_mask=attn_mask.bool()

            attn_output, attn_output_weights = self.multiheadattention(queries, keys, values,attn_mask=attn_mask)
            x=tokens+attn_output
            nx=_layer_norm(self.ln2,x)
            x=x+self.mlp(nx)
            self.set(self.output_name,x)

class TransformerMultiBlockAgent(Agents):
    def __init__(self,n_layers,embedding_size,n_heads,n_steps=None,prefix="attn_",use_layer_norm=False):
        agents=[]
        for k in range(n_layers):
            in_prefix=prefix+str(k+1)
            out_prefix=prefix+str(k+2)
            if k==n_layers-1:
                out_prefix=prefix+"out"
            if k==0:
                in_prefix=prefix+"in"
            agents.append(TransformerBlockAgent(embedding_size,n_heads,n_steps,in_prefix+"/x",out_prefix+"/x",use_layer_norm=use_layer_norm))
        super().__init__(*agents)


if __name__=="__main__":
    print("Check that transformers and batch transformers are computing the same output")
    a=torch.randn(5,3,4)
    workspace=Workspace()
    workspace.set_full("x",a)
    agent=TransformerBlockAgent(embedding_size=4,n_heads=1,n_steps=2,input_name="x",output_name="y",use_layer_norm=False)
    for t in range(5):
        agent(workspace,t=t)
    y1=workspace.get_full("y")
    print(y1)

    workspace=Workspace()
    workspace.set_full("x",a)
    agent(workspace)
    y2=workspace.get_full("y")
    print(y2)
    assert ((y1-y2)**2).lt(0.0000001).all(),"Problem..."
