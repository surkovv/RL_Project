import numpy as np
import matplotlib.pyplot as plt
from ppo_algo_continuous import PPOContinuousAgent
import time

if __name__ == '__main__':
    seeds = [42, 0, 345]
    all_eval_histories = []
    all_train_histories = []
    start_time = time.time()
    name = 'Pendulum-v1'
    best_means = []
    best_stds = []

    for seed in seeds:
        agent = PPOContinuousAgent(random_seed=seed)
        eval_scores, train_scores, best_mean, best_std = agent.train()
        all_eval_histories.append(eval_scores)
        all_train_histories.append(train_scores)
        best_means.append(best_mean)
        best_stds.append(best_std)

        np.save(f'ppo_eval_seed{seed}_{name}.npy', np.array(eval_scores))
    print("Best mean reward: ", best_means)
    print("Best std reward: ", best_stds)
    total_time = time.time() - start_time
    print("Total time taken: ", total_time)

    min_eval_len = min(len(s) for s in all_eval_histories)
    eval_array = np.array([list(s.values())[:min_eval_len] for s in all_eval_histories])
    step_array = np.array([list(s.keys())[:min_eval_len] for s in all_eval_histories])
    mean_eval = np.mean(eval_array, axis=0)
    std_eval = np.std(eval_array, axis=0)
    eval_x = np.mean(step_array, axis=0) 

    plt.figure(figsize=(10, 6))
    plt.plot(eval_x, mean_eval, label='Mean Eval Reward', color='blue')
    plt.fill_between(eval_x, mean_eval - std_eval, mean_eval + std_eval, alpha=0.3, label='±1 Std Dev', color='blue')
    plt.xlabel('Training Iteration')
    plt.ylabel('Evaluation Reward')
    plt.title(f"PPO Evaluation Mean ± Std across Seeds on {agent.env_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    min_train_len = min(len(s) for s in all_train_histories)
    all_train_histories = [s[:min_train_len] for s in all_train_histories]
    train_array = np.array(all_train_histories)
    mean_train = np.mean(train_array, axis=0)
    std_train = np.std(train_array, axis=0)
    train_x = np.arange(min_train_len)

    plt.figure(figsize=(10, 6))
    plt.plot(train_x, mean_train, label='Mean Training Reward', color='blue')
    plt.fill_between(train_x, mean_train - std_train, mean_train + std_train, alpha=0.3, label='±1 Std Dev', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Training Reward')
    plt.title(f"PPO Training Reward Mean ± Std across Seeds on {agent.env_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()