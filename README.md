# This repository contains implementations of several reinforcement learning (RL) algorithms using PyTorch and Gym:

-PPO (Proximal Policy Optimization) - Discrete

-PPO (Proximal Policy Optimization) - Continuous

-TD3 (Twin Delayed Deep Deterministic Policy Gradient) - Continuous

-DQN (Deep Q-Learning) - Notebook-based prototype

### Requirements
```bash
pip install gym numpy matplotlib torch gymnasium
```



### Running the Code
## PPO Algorithm
### Discrete
```bash
python main.py
```
Example:
-Train PPO on CartPole-v1 using seeds [0, 23, 1337]
-Save evaluation scores as .npy files
-Plot:
  Mean ± std of evaluation reward over training
  Mean ± std of training reward

  
