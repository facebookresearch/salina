# PPO benchmark on brax environments (in progress)

We benchmarked PPO on several complex continuous control tasks coming from Brax suit (https://github.com/google/brax). We provide in this section results and hyperparameters to reproduce the experiments in GPU (both acquisition and loss computation).

## Results

![alt text](results/halfcheetah_results.png)


## Running experiments
 * Make sure your version of Brax is up to date (https://github.com/google/brax) 
 * If you want to change the env tested, choose the correct yaml file name in `ppo.py` (line 143)
 * Run `OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false python ppo.py`