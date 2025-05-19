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
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)
    

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], 1))
    

class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=1e-3)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=1e-3)

        self.max_action = max_action
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, buffer, batch_size=100, gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_delay=2):
        self.total_it += 1
        state, action, reward, next_state, done = buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).unsqueeze(1).to(device)

        noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
        next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

        target_Q1 = self.critic1_target(next_state, next_action)
        target_Q2 = self.critic2_target(next_state, next_action)
        target_Q = reward + (1 - done) * gamma * torch.min(target_Q1, target_Q2)

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

        if self.total_it % policy_delay == 0:
            actor_loss = -self.critic1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
            for p, tp in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
            for p, tp in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)


def train_agent_td3(agent, env, episodes, batch_size, num_steps):
    buffer = ReplayBuffer()
    exploration_noise_start = 0.3

    returns = []

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        for n_step in range(num_steps):
            action = agent.select_action(state)
            exploration_noise = exploration_noise_start * (num_steps - n_step) / num_steps
            action = (action + np.random.normal(0, exploration_noise, size=action_dim)).clip(-max_action, max_action)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(buffer) > 1000:
                agent.train(buffer, batch_size=batch_size)

        returns.append(total_reward)
        print(f"Episode {ep}, Return: {total_reward:.2f}")

    # Plot
    plt.plot(returns)
    plt.title("TD3 Training on Pendulum-v1")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid()
    plt.show()