from salina import Agent



# TODO: look at : 
# https://stackoverflow.com/questions/26515595/how-does-one-ignore-unexpected-keyword-arguments-passed-to-a-function
# (response 2)
# for a more generic response.
# add decorator pattern to use alois pourchot'agents as salina agents
class Salina_Actor_Decorator(Agent):

    def __init__(self, nn):
        super().__init__()
        self.nn = nn
    
    def forward(self,t, **kwargs):
        obs = self.get(('env/env_obs',t))
        action = self.nn.forward(obs)
        self.set(('action',t),action)

class Salina_Qcritic_Decorator(Agent):

    def __init__(self, nn):
        super().__init__()
        self.nn = nn
    
    def forward(self,t, **kwargs):
        obs = self.get(('env/env_obs',t))
        action = self.get(('action',t))
        q_value = self.nn.forward(obs,action)
        self.set(('q_value',t),q_value)
