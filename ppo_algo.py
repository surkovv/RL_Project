import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from gym.wrappers import NormalizeObservation

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value


class PPOAgent:
    def __init__(self, env_name="Acrobot-v1", hidden_size=64, learning_rate=1e-3, gamma=0.99,
                 lam=0.95, clip_eps=0.2, value_coef=0.5, entropy_coef=0.01, train_iters=500,
                 steps_per_iter=1024, mini_batch_size=128, ppo_epochs=10, eval_interval=5):

        self.env_name = env_name
        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.train_iters = train_iters
        self.steps_per_iter = steps_per_iter
        self.mini_batch_size = mini_batch_size
        self.ppo_epochs = ppo_epochs
        self.eval_interval = eval_interval

    def train_with_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

        env = NormalizeObservation(gym.make(self.env_name))
        env.seed(seed)
        obs_space = env.observation_space.shape[0]
        action_space = env.action_space.n

        model = ActorCritic(obs_space, action_space, self.hidden_size)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        score_history = []
        best_reward_mean = -np.inf
        best_reward_std = 0.0
        eval_score_history = {}
        global_step = 0

        for i in range(self.train_iters):
            states, actions, rewards, dones = [], [], [], []
            log_probs, values, ep_rewards = [], [], []
            ep_reward = 0.0
            state = np.asarray(env.reset(), dtype=np.float32)

            for t in range(self.steps_per_iter):
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                with torch.no_grad():
                    logits, value = model(state_tensor)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                next_state, reward, done, _ = env.step(action.item())
                next_state = np.asarray(next_state, dtype=np.float32)

                states.append(state)
                actions.append(action.item())
                rewards.append(reward)
                dones.append(done)
                log_probs.append(log_prob.item())
                values.append(value.item())

                ep_reward += reward
                global_step += 1

                if done:
                    score_history.append(ep_reward)
                    ep_rewards.append(ep_reward)
                    ep_reward = 0.0
                    next_state = np.asarray(env.reset(), dtype=np.float32)

                state = next_state

            next_value = 0.0
            if not dones[-1]:
                _, next_val = model(torch.from_numpy(state).float().unsqueeze(0))
                next_value = next_val.item()

            advantages = [0.0] * len(rewards)
            last_adv = 0.0
            for t in reversed(range(len(rewards))):
                if dones[t]:
                    delta = rewards[t] - values[t]
                    last_adv = 0.0
                else:
                    next_val = values[t+1] if t < len(rewards)-1 else next_value
                    delta = rewards[t] + self.gamma * next_val - values[t]
                last_adv = delta + self.gamma * self.lam * last_adv
                advantages[t] = last_adv

            states_tensor = torch.tensor(states, dtype=torch.float32)
            actions_tensor = torch.tensor(actions, dtype=torch.long)
            old_log_probs_tensor = torch.tensor(log_probs, dtype=torch.float32)
            values_tensor = torch.tensor(values, dtype=torch.float32)
            advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
            returns_tensor = advantages_tensor + values_tensor
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

            for _ in range(self.ppo_epochs):
                indices = np.arange(self.steps_per_iter)
                np.random.shuffle(indices)
                for start in range(0, self.steps_per_iter, self.mini_batch_size):
                    end = start + self.mini_batch_size
                    batch_idx = indices[start:end]

                    batch_states = states_tensor[batch_idx]
                    batch_actions = actions_tensor[batch_idx]
                    batch_old_logs = old_log_probs_tensor[batch_idx]
                    batch_adv = advantages_tensor[batch_idx]
                    batch_ret = returns_tensor[batch_idx]

                    logits, values_pred = model(batch_states)
                    dist = Categorical(logits=logits)
                    new_log_probs = dist.log_prob(batch_actions)

                    ratio = torch.exp(new_log_probs - batch_old_logs)
                    clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    policy_loss = -torch.mean(torch.min(ratio * batch_adv, clipped_ratio * batch_adv))
                    value_loss = torch.mean((values_pred.squeeze(-1) - batch_ret)**2)
                    entropy_loss = -torch.mean(dist.entropy())

                    loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            if ep_rewards:
                print(f"Seed {seed} | Iter {i+1}, Avg Reward: {np.mean(ep_rewards):.2f}")

            if (i + 1) % self.eval_interval == 0:
                eval_rewards = []
                for _ in range(5):
                    s = np.asarray(env.reset(), dtype=np.float32)
                    done = False
                    total_reward = 0.0
                    while not done:
                        s_tensor = torch.from_numpy(s).float().unsqueeze(0)
                        with torch.no_grad():
                            logits, _ = model(s_tensor)
                            a = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                        s, r, done, _ = env.step(int(a.item()))
                        s = np.asarray(s, dtype=np.float32)
                        total_reward += r
                    eval_rewards.append(total_reward)
                avg_eval = float(np.mean(eval_rewards))
                eval_score_history[global_step] = avg_eval 
                print(f"Seed {seed} | Steps {global_step}, Eval Avg Reward: {avg_eval:.2f}")

                if avg_eval >= best_reward_mean:
                # Re-evaluate
                    reevaluated = []
                    for _ in range(20):
                        s = env.reset()
                        s = np.asarray(s, dtype=np.float32)
                        done = False
                        total_reward = 0.0
                        while not done:
                            s_tensor = torch.from_numpy(s).float().unsqueeze(0)
                            with torch.no_grad():
                                logits, _ = model(s_tensor)
                                a = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                            s, r, done, _ = env.step(int(a.item()))
                            s = np.asarray(s, dtype=np.float32)
                            total_reward += r
                        reevaluated.append(total_reward)

                    mean = np.mean(reevaluated)
                    std = np.std(reevaluated)

                    if mean >= best_reward_mean:
                        best_reward_mean = mean
                        best_reward_std = std
        env.close()
        return eval_score_history, score_history, best_reward_mean, best_reward_std

