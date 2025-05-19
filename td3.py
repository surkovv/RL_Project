import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt

from utils import ReplayBuffer

device = "cuda"

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim, num_layers):
        super().__init__()
        layer_list = [
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
        ]
        for _ in range(num_layers - 2):
            layer_list.append(nn.Linear(hidden_dim, hidden_dim))
            layer_list.append(nn.ReLU())
        layer_list.append(nn.Linear(hidden_dim, action_dim))
        layer_list.append(nn.Tanh())
        self.net = nn.Sequential(
            *layer_list
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)
    

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers):
        super().__init__()
        layer_list = [
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
        ]
        for _ in range(num_layers - 2):
            layer_list.append(nn.Linear(hidden_dim, hidden_dim))
            layer_list.append(nn.ReLU())
        layer_list.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(
            *layer_list
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], 1))
    

class TD3:
    def __init__(self, env_name="Pendulum-v1", episodes=100, 
                 batch_size=100, learning_rate=1e-3, 
                 hidden_dim=128, num_layers=2,
                 gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_delay=2, eval_interval=5):
        self.env = gym.make(env_name)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        max_action = float(self.env.action_space.high[0])
        self.episodes = episodes
        self.batch_size = batch_size
        self.num_steps = self.env.spec.max_episode_steps
        self.gamma = gamma
        self.tau = tau
        self.policy_noise=policy_noise
        self.noise_clip = noise_clip
        self.policy_delay=policy_delay
        self.eval_interval = eval_interval

        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim, num_layers).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action, hidden_dim, num_layers).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = Critic(state_dim, action_dim, hidden_dim, num_layers).to(device)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim, num_layers).to(device)
        self.critic1_target = Critic(state_dim, action_dim, hidden_dim, num_layers).to(device)
        self.critic2_target = Critic(state_dim, action_dim, hidden_dim, num_layers).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)

        self.max_action = max_action
        self.total_it = 0

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            return self.actor(state).cpu().data.numpy().flatten()

    def train(self, buffer):
        self.total_it += 1
        state, action, reward, next_state, done = buffer.sample(self.batch_size)
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).unsqueeze(1).to(device)

        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

        target_Q1 = self.critic1_target(next_state, next_action)
        target_Q2 = self.critic2_target(next_state, next_action)
        target_Q = reward + (1 - done) * self.gamma * torch.min(target_Q1, target_Q2)

        current_Q1 = self.critic1(state, action)
        current_Q2 = self.critic2(state, action)

        loss1 = nn.MSELoss()(current_Q1, target_Q.detach())
        loss2 = nn.MSELoss()(current_Q2, target_Q.detach())

        self.critic1_optimizer.zero_grad()
        loss1.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        loss2.backward()
        self.critic2_optimizer.step()

        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def train_with_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.env.reset(seed=seed)

        buffer = ReplayBuffer()
        exploration_noise_start = 2

        returns = []
        eval_returns = []

        action_dim = self.env.action_space.shape[0]
        max_action = float(self.env.action_space.high[0])

        for ep in range(self.episodes):
            state, _ = self.env.reset()
            total_reward = 0
            for n_step in range(self.num_steps):
                action = self.select_action(state)
                exploration_noise = exploration_noise_start * (self.episodes - ep) / self.episodes
                action = (action + np.random.normal(0, exploration_noise, size=action_dim)).clip(-max_action, max_action)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                # print(reward)
                done = terminated or truncated
                buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if len(buffer) > self.batch_size:
                    self.train(buffer)
                
                if done:
                    break

            print(f"Episode {ep}, Return: {total_reward:.2f}")
            returns.append(total_reward)

            if ep % self.eval_interval == self.eval_interval - 1:
                state, _ = self.env.reset()
                total_reward = 0
                for n_step in range(self.num_steps):
                    action = self.select_action(state)
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    state = next_state
                    total_reward += reward
                    if done:
                        break

                print(f"Episode {ep}, Eval return: {total_reward:.2f}")
                eval_returns.append(total_reward)

        # Plot
        plt.plot(returns)
        plt.title("TD3 Training on Pendulum-v1")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid()
        plt.show()

        return returns, eval_returns