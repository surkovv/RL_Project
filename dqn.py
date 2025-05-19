import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import ReplayBuffer
import random
import matplotlib.pyplot as plt

device = "cuda"

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.model(x)
    

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)
        self.criteria = nn.MSELoss()
        self.action_dim = action_dim
        self.gamma = 0.99
        self.tau = 0.005
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            return self.q_network(state).argmax().item()

    def train(self, buffer, batch_size):
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        q_values = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.criteria(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update()

    def soft_update(self):
        for param, target_param in zip(self.q_network.parameters(), self.target_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_agent_dqn(agent, env, episodes, batch_size, num_steps):
    buffer = ReplayBuffer()
    rewards_history = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(num_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(buffer) > batch_size:
                agent.train(buffer, batch_size)

            if done:
                break

        agent.update_epsilon()
        rewards_history.append(total_reward)
        print(f"Episode {ep}, Reward: {total_reward:.1f}, Epsilon: {agent.epsilon:.2f}")

    # Plotting
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN on CartPole-v1")
    plt.grid()
    plt.show()