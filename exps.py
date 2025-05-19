import json
from datetime import datetime
from td3 import TD3

def do_experiment(env_name, params_dict):
    training_rewards, eval_rewards = [], []
    for seed in range(3):
        agent = TD3(env_name=env_name, **params_dict)
        t, e = agent.train_with_seed(seed)
        training_rewards.append(t)
        eval_rewards.append(e)
    
    with open(f"exps/{datetime.now()}.json", 'w') as f:
        json.dump({
            "env_name": env_name,
            "param_dict": params_dict,
            "training_rewards": training_rewards,
            "eval_rewards": eval_rewards
        }, f)
    print("DONE")

default_pendulum_dict = {"episodes": 200, "policy_noise": 0.1, "exploration_noise_start": 0.5, "hidden_dim": 256, "num_layers": 3}

learning_rates = [3e-4, 1e-3, 3e-3]
start_epsilons = [0.1, 0.5, 2.5]
policy_epsilons = [0.02, 0.1, 0.5]
hidden_dims = [8, 32, 128]
num_layers = [1, 2, 3, 4]

from copy import deepcopy

for lr in learning_rates:
    d = deepcopy(default_pendulum_dict)
    d["learning_rate"] = lr
    do_experiment("Pendulum-v1", d)

for eps in learning_rates:
    d = deepcopy(default_pendulum_dict)
    d["exploration_noise_start"] = eps
    do_experiment("Pendulum-v1", d)

for eps in policy_epsilons:
    d = deepcopy(default_pendulum_dict)
    d["policy_noise"] = eps
    do_experiment("Pendulum-v1", d)

for h in hidden_dims:
    d = deepcopy(default_pendulum_dict)
    d["hidden_dim"] = h
    do_experiment("Pendulum-v1", d)

for n in hidden_dims:
    d = deepcopy(default_pendulum_dict)
    d["num_layers"] = n
    do_experiment("Pendulum-v1", d)