# Double DQN (Fastest)

We propose an experimental version of Double DQN where loss computation is made **while acquiring trajectories** instead of alternating between acquisition and loss minimization. It allows to make acquisition and loss minimization in parallel and can increase the learning speed of the original Double DQN algorithm.
