import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import ReplayBuffer
import random
import matplotlib.pyplot as plt
import gymnasium as gym

device = "cpu"

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, num_layers):
        super(QNetwork, self).__init__()
        layer_list = [
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
        ]
        for _ in range(num_layers - 2):
            layer_list.append(nn.Linear(hidden_dim, hidden_dim))
            layer_list.append(nn.ReLU())
        layer_list.append(nn.Linear(hidden_dim, action_dim))
        self.model = nn.Sequential(
            *layer_list
        )

    def forward(self, x):
        return self.model(x)
    

class DQNAgent:
    def __init__(self, env_name="CartPole-v1", 
            hidden_dim=128, episodes=600, 
            batch_size=64, num_steps=200, num_layers=2, 
            learning_rate=1e-3, eval_interval=5, epsilon=1.0,
            tau = 0.05, gamma = 0.99,
            epsilon_min=0.05, epsilon_decay=0.99):
        self.env = gym.make(env_name)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        self.episodes = episodes
        self.batch_size = batch_size
        self.num_steps = num_steps

        self.q_network = QNetwork(state_dim, action_dim, hidden_dim, num_layers).to(device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim, num_layers).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.best_policy = self.q_network

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criteria = nn.MSELoss()
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.eval_interval = eval_interval

    def select_action(self, state, deterministic=False):
        if not deterministic and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                return self.q_network(state).argmax().item()

    def train(self, buffer, batch_size, do_soft_update):
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
        
        if do_soft_update:
            self.soft_update()

    def soft_update(self):
        for param, target_param in zip(self.q_network.parameters(), self.target_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train_with_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.env.reset(seed=seed)

        buffer = ReplayBuffer()
        rewards_history = []
        eval_rewards_history = []

        for ep in range(self.episodes):
            state, _ = self.env.reset()
            total_reward = 0

            for step in range(self.num_steps):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                buffer.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if len(buffer) > self.batch_size:
                    self.train(buffer, self.batch_size, step)

                if done:
                    break

            self.update_epsilon()
            rewards_history.append(total_reward)
            # print(f"Episode {ep}, Reward: {total_reward:.1f}, Epsilon: {self.epsilon:.2f}")

            if ep % self.eval_interval == self.eval_interval - 1:
                eval_rewards = self.evaluate_policy(num_episodes=5)
                eval_rewards_history.append(np.mean(eval_rewards))
                # print(f"Episode {ep}, Eval return: {total_reward:.2f}")

        # # Plotting
        # plt.plot(rewards_history)
        # plt.xlabel("Episode")
        # plt.ylabel("Reward")
        # plt.title("DQN on CartPole-v1")
        # plt.grid()
        # plt.show()
        return rewards_history, eval_rewards_history

    def evaluate_policy(self, num_episodes=20):
        rewards = []
        self.env.reset()

        for ep in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action = self.select_action(state, deterministic=True)
                next_state, reward, terminated, truncated, _ = self.env.step(
                    action)
                total_reward += reward
                done = terminated or truncated
                state = next_state

            print(f"Episode {ep}, Eval return: {total_reward:.2f}")

            rewards.append(total_reward)

        return rewards