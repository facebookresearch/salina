import copy
import gym
from gym.wrappers import TimeLimit
from omegaconf import DictConfig
from salina import Workspace,Agent, instantiate_class
from salina.agents.asynchronous import AsynchronousAgent
import torch
from typing import Callable, Union
from gym.spaces import Box,Discrete

def get_env_dimensions(env) -> tuple:
    env = instantiate_class(env)
    obs_dim = env.observation_space.shape[0]
    if isinstance(env.action_space,Discrete) : 
        action_dim = env.action_space.n
        del env 
        return obs_dim,action_dim

    elif isinstance(env.action_space,Box) : 
        action_dim = env.action_space.shape[0]
        max_action = env.action_space.high[0]
        del env 
        return obs_dim,action_dim,max_action
    else:
        raise Exception(f'{type(env.action_space)} unknown')
    
def make_gym_env(max_episode_steps, env_name):
    return TimeLimit(gym.make(env_name), max_episode_steps=max_episode_steps)


def soft_param_update(network_to_update, network, rho):
    for n_to_update, p_net in zip(network_to_update.parameters(),
                                  network.parameters()):
        n_to_update.data.copy_(rho * p_net.data + (1 - rho) * n_to_update.data)


# dict configs utils :
def key_path_in_dict(nested_dict: dict, key_path: str):
    ''' Check if a sequences of keys exists in a nested dict '''
    try:
        keys = key_path.split('.')
        rv = nested_dict
        for key in keys:
            rv = rv[key]
        return True
    except KeyError:
        return False


def set_value_with_key_path(nested_dict: DictConfig, key_path: str, value):
    keys = key_path.split('.')
    for key in keys[:-1]:
        nested_dict = nested_dict[key]
    nested_dict[keys[-1]] = value


### Salina additions ####

# need to check if this function works well using cuda
def vector_to_parameters(vec: torch.Tensor, parameters) -> None:
    r"""Convert one vector to the parameters

    Args:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data.copy_(vec[pointer:pointer + num_param].view_as(param).data)

        # Increment the pointer
        pointer += num_param

# !!!! nRemoteParamAgent,  Not ready (WIP) !!!
class nRemoteParamAgent(Agent):
    '''
        Class that allows to evaluate N (different) individuals with m processes
        The user have to provide: 
            1/ the aquisition agent list or template
            2/ list of parameters for each of the individual of the pop
            3/ the function that apply the parameters to the acquisition agent
        This implementation is based on the  Asynchronous agents
        (i think another implementation could use the Nremote agent 
        maybe by slicing the shared workspace to separate the experiences 
        collected by each individual)
    '''
    def __init__(self, acq_agent: Agent, n_process: int,
                 name: str = '') -> None:
        '''
        Implements a list of agent which are executed aynchronously in another process.
        Each agent can be parametrized by specific parameters and will returns it's own 
        workspace. 
        acq_agent : an instance of the agent that will be runned over each processes
        n_process : 
        apply_params : a function f(acq_agent, param) => acq_agent
                       Allow to update each of the agent with a specific set of parameters.
        '''
        super().__init__(name)
        self.async_agents = []
        self.n_process = n_process
        for i in range(n_process):
            async_agent = AsynchronousAgent(copy.deepcopy(acq_agent))
            self.async_agents.append(async_agent)

    def __call__(self, params: list, apply_params: Callable, **kwargs):
        self.workspaces = []
        nb_agent_to_launch = len(params)
        pool = []
        to_launch_id = 0

        def launch_agent(agent, to_launch_id):
            apply_params(agent, params[to_launch_id])
            agent(**kwargs)
            to_launch_id += 1
            pool.append(agent)

        for i in range(min(nb_agent_to_launch, self.n_process)):
            launch_agent(self.async_agents[i], i)

        while(True):
            for agent in pool:
                if not agent.is_running():
                    workspace = agent.get_workspace()
                    if workspace:
                        self.workspaces.append(workspace)
                        if len(self.workspaces) == nb_agent_to_launch:
                            return
                    if to_launch_id < nb_agent_to_launch:
                        last_launched_id = len(self.workspaces) - 1
                        apply_params(agent, params[last_launched_id])
                        agent(**kwargs)
                        last_launched_id += 1
                    else:
                        pool.remove(agent)

    def get_workspaces(self) -> list[Workspace]:
        try:
            return self.workspaces
        except AttributeError:
            raise Exception("The nRemoteParamAgent has not been called yet, workspaces have not been created")

    def close(self) -> None:
        for a in self.async_agents:
            a.close()

# !!!! nRemoteDistinctAgents Not functionnal !!!
# To my knownledge you can't
# change the content of an async agent
class nRemoteDistinctAgents(Agent):
    '''
        Class that allows to evaluate N (different) individuals with m processes
        Basic usage : 
        remote = nRemoteDistinctAgents(n_process)
        remote(acq_agent_list,)
        The user have to provide: 
            1/ a list of acqusition_agent that will be copied to remotes
        This implementation is based on the  Asynchronous agents
        (i think another implementation could use the Nremote agent
        maybe by slicing the shared workspace to separate the experiences
        collected by each individual)
    '''
    def __init__(self, n_process: int, name: str = '') -> None:
        '''
        Implements a list of agent which are executed aynchronously in another process.
        Each agent can be parametrized by specific parameters and will returns it's own 
        workspace. 
        acq_agent : an instance of the agent that will be runned over each processes
        n_process : 
        apply_params : a function f(acq_agent, param) => acq_agent
                       Allow to update each of the agent with a specific set of parameters.
        '''
        super().__init__(name)
        self.async_agents = []
        self.n_process = n_process
        for i in range(n_process):
            async_agent = AsynchronousAgent(None)
            self.async_agents.append(async_agent)

    def __call__(self, acq_agents: list[Agent],
                 agents_args: Union[list, dict, None],
                 **kwargs):

        def get_agent_args(agent_id):
            if agents_args is None:
                args = {}
            elif isinstance(agents_args, dict):
                args = agents_args
            elif isinstance(agents_args, list):
                args = agents_args[agent_id]
            else:
                raise Exception('Unsupported')
            return args

        self.workspaces = []
        nb_agent_to_launch = len(acq_agents)
        to_launch_id = 0

        pool = []
        for _ in range(min(self.n_process, nb_agent_to_launch)):
            args = get_agent_args(to_launch_id)
            self.async_agents[to_launch_id].agent = acq_agents[to_launch_id]
            self.async_agents[to_launch_id](**args, **kwargs)
            pool.append(self.async_agents[to_launch_id])
            to_launch_id += 1

        while(len(self.workspaces) < nb_agent_to_launch):
            j = 0
            for async_agent in pool:
                if not async_agent.is_running():
                    workspace = async_agent.get_workspace()
                    self.workspaces.append(workspace)
                    # print(f'process {j} finished total {len(self.workspaces)}/{nb_agent_to_launch}')
                    if len(self.workspaces) == nb_agent_to_launch:
                        return
                    if to_launch_id < nb_agent_to_launch:
                        async_agent.agent = acq_agents[to_launch_id]
                        args = get_agent_args(to_launch_id)
                        async_agent(**args, **kwargs)
                        # print(f'process {j} launched for agent {to_launch_id}')
                        to_launch_id += 1
                    else: 
                        pool.remove(async_agent)
                j += 1

    def get_workspaces(self) -> list[Workspace]:
        try:
            return self.workspaces
        except AttributeError:
            raise Exception("The nRemoteParamAgent has not been called yet, workspaces have not been created")

    def close(self) -> None:
        for a in self.async_agents:
            a.close()
