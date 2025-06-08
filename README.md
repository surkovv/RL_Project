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

```bash
# modify the variables here
class PPOAgent:
    def __init__(self, env_name="CartPole-v1", hidden_size=64, learning_rate=3e-4, gamma=0.99,
                 lam=0.95, clip_eps=0.2, value_coef=0.5, entropy_coef=0.01, train_iters=500,
                 steps_per_iter=4096, mini_batch_size=512, ppo_epochs=10, eval_interval=5):
```

-Save evaluation scores as .npy files

-Plot:

  Mean ± std of evaluation reward over training
  
  Mean ± std of training reward

### Continuous
```bash
# Same thing can be done for this algorithm as in the discrete case
python main_continuous.py
```


## DQN Algortihm
Open DQN.ipynb in Jupyter Notebook

```bash
# Paramters are changed in the 3rd cell
class DQNExperiments:
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4
    NUM_EPISODES = 600
    steps_done = 0
```

## SAC Algorithm
Open testing.ipynb in Jupyter Notebook

```bash
# Paramters are changed in the first cell
env_name = "Pendulum-v1"
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
agent = SAC(env, 
            env_name,
            state_dim,
             action_dim,
             batch_size=64,
             episodes=200,
             num_steps=300,
             scale=0.1, 
             max_actions=max_action, 
             alpha=0.3, 
             use_autotune=False)

returns, eval_rewards = agent.train_with_seed(seed = 48)
```

## TD3 Algortihm
```bash
python td3.py
```



  
