import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from Agents.QNet import QNet
from Agents.ReplayMemory import ReplayMemory, Transition

class DQNAgent:
    def __init__(self, env, gamma=0.99, lr=1e-3, batch_size=64, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01, hidden_dim=128, tau=1.0,
                 num_episodes=500, eval_interval=5, target_update_freq=10,
                 constant_epsilon=False):

        self.env = env
        self.n_observations = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.constant_epsilon = constant_epsilon
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.eval_interval = eval_interval
        self.num_episodes = num_episodes
        self.target_update_freq = target_update_freq

        self.policy_net = QNet(self.n_observations, self.n_actions, self.hidden_dim)
        self.target_net = QNet(self.n_observations, self.n_actions, self.hidden_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = ReplayMemory()

    def act(self, state, eps_greedy=True):
        sample = np.random.random()
        if eps_greedy and sample < self.epsilon:
            action = self.env.action_space.sample()

        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = self.policy_net(state).argmax().item()
        return action

    def training_step(self):
        """
        Perform a single training step for the DQN agent using experience replay.
        """

        if len(self.memory) < self.batch_size:
            # Not enough data in memory to sample a full batch
            return

        # Sample a batch of transitions
        transitions = self.memory.sample(self.batch_size)
        transitions = Transition(*zip(*transitions))
        # Convert batch data to PyTorch tensors
        state = torch.FloatTensor(np.array(transitions.state))  # shape: [batch_size, state_dim]
        action = torch.LongTensor(transitions.action).unsqueeze(1)  # shape: [batch_size, 1]
        reward = torch.FloatTensor(transitions.reward).unsqueeze(1)  # shape: [batch_size, 1]
        next_state = torch.FloatTensor(transitions.next_state)  # shape: [batch_size, state_dim]
        done = torch.FloatTensor(transitions.done).unsqueeze(1)  # shape: [batch_size, 1]

        # Compute Q(s_t, a) — the model computes Q-values for all actions;
        # we gather only the Q-values of the taken actions
        q_values = self.policy_net(state).gather(1, action)
        with torch.no_grad():
            max_next_q = self.target_net(next_state).max(1)[0].unsqueeze(1)
            target_q = reward + self.gamma * max_next_q * (1 - done)

        loss = self.criterion(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def soft_update(self):
        for param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def train_with_seed(self, seed=42):

        rewards_history = []
        eval_rewards_history = []

        for episode in range(self.num_episodes):
            state, _ = self.env.reset()
            self.env.action_space.seed(seed)
            self.env.observation_space.seed(seed)

            done = False
            total_reward = 0

            while not done:
                action = self.act(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.memory.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                self.training_step()

            if not self.constant_epsilon:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            rewards_history.append(total_reward)

            if self.tau == 1:
                if episode % self.target_update_freq == 0:
                    self.update_target()
            else:
                self.soft_update()

            # print(f"Episode {episode}, Total reward: {total_reward}, Epsilon: {self.epsilon:.3f}")

            if episode % self.eval_interval == self.eval_interval - 1:
                state, _ = self.env.reset()
                total_reward = 0
                while not done:
                    action = self.act(state, eps_greedy=False)
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated

                    self.memory.push(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward


                eval_rewards_history.append(total_reward)


        return rewards_history, eval_rewards_history
