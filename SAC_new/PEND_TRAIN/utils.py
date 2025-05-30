from collections import deque
import random
import numpy as np 

class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)
    
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()

    def reset(self):
        self.x = np.zeros(self.action_dim)

    def sample(self):
        dx = self.theta * (self.mu - self.x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.action_dim)
        self.x += dx
        return self.x
