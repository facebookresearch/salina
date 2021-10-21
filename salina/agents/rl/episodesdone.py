from salina import Agent
import torch

class EpisodesDone(Agent):
    # Compute a variable that tells if all episodes are done when using an auto-reset wrapper. It allows to convert an autoreset env to a classical env
    def __init__(self,in_var="env/done",out_var="env/_done"):
        super().__init__()
        self.in_var=in_var
        self.out_var=out_var

    def forward(self,t,**args):
        d=self.get((self.in_var,t))
        if t==0:
            self.state=self.zeros_like(d).bool()
        self.state=torch.logical_or(self.state,d)
        self.set((self.out_var,t),self.state)
