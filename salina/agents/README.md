# salina.agents

We propose a list of agents to reuse (see Documentation in the code)

## utils

* Agents: Execute multiple agents sequentially
* TemporalAgent: Execute one agent over multiple timesteps
* CopyAgent: An agent to create copies of variables
* PrintAgent: An agent that print variables

## remote

* RemoteAgent: A single agent in a single process
* NRemoteAgent: A single agent parallelized over multiple processes

## gym

* GymAgent: An agent based on an openAI gym environment
* AutoResetGymAgent: The same, but with an autoreset when reaching terminal states

## brax
* AutoResetBraxAgent: An agent based on a BRAX environment with autoreset
* NoAutoResetBraxAgent: An agent based on a BRAX environment without autoreset

## dataloader
* ShuffledDatasetAgent: An agent to read random batches in a torch.utils.data.Dataset
* DataLoaderAgent: An agent to do one pass over a complete dataset (based on a DataLoader)

## asynchronous

* AsynchronousAgent: it is used to execute any agent asynchronously, the agent creating its own workspace at each execution

1. `agent=AsynchronousAgent(my_agent)`
2. `agent(**execution_arguments` (not workspace provided)
3. `if not agent.is_running(): workspace=agent.get_workspace()`
