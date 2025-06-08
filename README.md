This repository contains implementations of several reinforcement learning (RL) algorithms using PyTorch and Gym:

-PPO (Proximal Policy Optimization) - Discrete

-PPO (Proximal Policy Optimization) - Continuous

-TD3 (Twin Delayed Deep Deterministic Policy Gradient) - Continuous

-DQN (Deep Q-Learning) - Notebook-based prototype

## Requirements
```bash
pip install gym numpy matplotlib torch gymnasium
```



## Running the Code
## PPO Algorithm
### Discrete
```bash
python main.py
```
Example:

-Train PPO on CartPole-v1 using seeds [0, 23, 1337] (edit seeds to try more)

-You can also edit the environment and pick the parameters commented at the bottom of the main.py file. Go to ppo_algo.py and modify the variables in the PPOAgent Class.

-Save evaluation scores as .npy files

-Plot:

  Mean ± std of evaluation reward over training
  
  Mean ± std of training reward

### Continuous
```bash
python main_continuous.py
```

## TD3 Algortihm
```bash
python td3.py
```

## DQN Algortihm
Open DQN.ipynb in Jupyter Notebook

## SAC Algorithm
Open testing.ipynb in Jupyter Notebook



  
