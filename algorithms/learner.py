
class learner():
    def get_acquisition_actor(self):
        raise NotImplementedError

    def update_acquisition_agent(self,acquisition_agent):
        raise NotImplementedError
    
    def get_acquisition_args(self):
        raise NotImplemented

    def train(self,workspace):
        raise NotImplementedError