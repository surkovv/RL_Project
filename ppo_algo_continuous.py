import gym
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from torch.distributions import Normal
from gym.wrappers import NormalizeObservation


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(ActorCritic, self).__init__()
        self.fc1_actor = nn.Linear(state_dim, hidden_size)
        self.fc2_actor = nn.Linear(hidden_size, hidden_size)

        self.fc1_critic = nn.Linear(state_dim, hidden_size)
        self.fc2_critic = nn.Linear(hidden_size, hidden_size)

        self.mean_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

        self.activation = nn.ReLU()
        self.log_std = nn.Parameter(0.01 * torch.ones(action_dim))

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)

        x_actor = self.activation(self.fc1_actor(x))
        x_actor = self.activation(self.fc2_actor(x_actor))

        x_critic = self.activation(self.fc1_critic(x))
        x_critic = self.activation(self.fc2_critic(x_critic))

        mean = self.mean_head(x_actor)
        log_std = torch.clamp(self.log_std, -1, 1)
        std = torch.exp(log_std).expand_as(mean)
        value = self.value_head(x_critic)

        return mean, std, value


class PPOContinuousAgent:
    def __init__(self, env_name="MountainCarContinuous-v0", hidden_size=64, learning_rate=1e-4, gamma=0.99,
                 lam=0.95, clip_eps=0.2, value_coef=0.5, entropy_coef=0.5, train_iters=500,
                 steps_per_iter=4096, mini_batch_size=256, ppo_epochs=5, eval_interval=10, save_interval=50,
                 random_seed=42):

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
        self.save_interval = save_interval
        self.seed = random_seed
        self.env = NormalizeObservation(gym.make(env_name))
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            self.env.seed(random_seed)

        self.obs_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.shape[0] if isinstance(self.env.action_space, gym.spaces.Box) else self.env.action_space.n

        self.model = ActorCritic(self.obs_space, self.action_space, hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        score_history = []
        eval_score_history = {}
        global_step = 0
        episode_count = 0
        best_reward_mean = -float("inf")
        best_reward_std = 0.0
        start_time = time.time()

        for i in range(self.train_iters):
            if i > 20:
                self.entropy_coef = 0.0
                self.lr = 1e-5
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr

            states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
            ep_rewards = []
            ep_reward = 0.0

            state = np.asarray(self.env.reset(), dtype=np.float32)

            for t in range(self.steps_per_iter):
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                with torch.no_grad():
                    mean, std, value = self.model(state_tensor)
                    dist = Normal(mean, std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(dim=-1)
                    action = torch.clamp(action, self.env.action_space.low[0], self.env.action_space.high[0])

                action_val = action.cpu().numpy().flatten()
                next_state, reward, done, _ = self.env.step(action_val)
                next_state = np.asarray(next_state, dtype=np.float32)

                states.append(state)
                actions.append(action_val)
                rewards.append(float(reward))
                dones.append(done)
                log_probs.append(float(log_prob.item()))
                values.append(float(value.item()))

                ep_reward += reward
                global_step += 1

                if done:
                    score_history.append(ep_reward)
                    ep_rewards.append(ep_reward)
                    ep_reward = 0.0
                    next_state = np.asarray(self.env.reset(), dtype=np.float32)
                    episode_count += 1

                state = next_state

            if dones[-1]:
                next_value = 0.0
            else:
                _, _, next_val = self.model(torch.from_numpy(state).float().unsqueeze(0))
                next_value = float(next_val.item())

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

            states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
            actions_tensor = torch.tensor(actions, dtype=torch.float32)
            old_log_probs_tensor = torch.tensor(log_probs, dtype=torch.float32)
            values_tensor = torch.tensor(values, dtype=torch.float32)
            advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
            returns_tensor = advantages_tensor + values_tensor
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

            for epoch in range(self.ppo_epochs):
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

                    mean_pred, std_pred, values_pred = self.model(batch_states)
                    dist = Normal(mean_pred, std_pred)
                    new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)

                    ratio = torch.exp(new_log_probs - batch_old_logs)
                    clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    policy_loss = -torch.mean(torch.min(ratio * batch_adv, clipped_ratio * batch_adv))
                    value_loss = torch.mean((values_pred.squeeze(-1) - batch_ret) ** 2)
                    entropy_loss = -torch.mean(dist.entropy())

                    loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            if ep_rewards:
                avg_batch_reward = float(np.mean(ep_rewards))
            else:
                avg_batch_reward = float(ep_reward)

            log_msg = f"Iter {i+1}/{self.train_iters}, Episodes: {len(ep_rewards)}, Avg Reward: {avg_batch_reward:.2f}"

            if self.eval_interval and (i+1) % self.eval_interval == 0:
                eval_rewards = []
                for _ in range(5):
                    s = np.asarray(self.env.reset(), dtype=np.float32)
                    done = False
                    total_reward = 0.0
                    while not done:
                        s_tensor = torch.from_numpy(s).float().unsqueeze(0)
                        with torch.no_grad():
                            mean, std, _ = self.model(s_tensor)
                            a = mean
                        a = torch.clamp(a, self.env.action_space.low[0], self.env.action_space.high[0])
                        s, r, done, _ = self.env.step(a.cpu().numpy().flatten())
                        s = np.asarray(s, dtype=np.float32)
                        total_reward += r
                    eval_rewards.append(total_reward)
                eval_reward = float(np.mean(eval_rewards))
                eval_score_history[global_step] = eval_reward
                log_msg += f", Eval Avg Reward: {eval_reward:.2f}"
                if eval_reward > best_reward_mean:
                    reevaluated = []
                    for _ in range(20):
                        s = self.env.reset()
                        s = np.asarray(s, dtype=np.float32)
                        done = False
                        total_reward = 0.0
                        while not done:
                            s_tensor = torch.from_numpy(s).float().unsqueeze(0)
                            with torch.no_grad():
                                mean, _, _ = self.model(s_tensor)
                                a = mean  
                            s, r, done, _ = self.env.step(a.cpu().numpy().flatten())
                            s = np.asarray(s, dtype=np.float32)
                            total_reward += r
                        reevaluated.append(total_reward)

                    mean = np.mean(reevaluated)
                    std = np.std(reevaluated)

                    if mean >= best_reward_mean:
                        best_reward_mean = mean
                        best_reward_std = std

            print(log_msg)

            if self.save_interval and (i+1) % self.save_interval == 0:
                save_path = f"ppo_{self.env_name}_checkpoint.pth"
                torch.save(self.model.state_dict(), save_path)
                print(f"Checkpoint saved: {save_path}")

        final_model_path = f"ppo_{self.env_name}_final.pth"
        torch.save(self.model.state_dict(), final_model_path)
        print(f"Training finished in {time.time() - start_time:.2f} seconds. Final model saved to {final_model_path}")

        self.env.close()
        return eval_score_history, score_history, best_reward_mean, best_reward_std
