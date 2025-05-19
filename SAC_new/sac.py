import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
import torch.nn.functional as F
import time
from utils import ReplayBuffer

device = "cpu"

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_actions = None):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU())
        self.mean = nn.Linear(256,action_dim)
        self.variance = nn.Linear(256,action_dim)
        self.max_action = max_actions
        self.reparam_noise = 1e-6
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def forward(self, state):

        latent = self.features(state)
        mean = self.mean(latent)
        variance = self.variance(latent)
        variance = torch.clamp(variance, min=1e-6, max=1)

        return mean, variance 
    
    def sample_normal(self, state, reparametrize = True, greedy = False):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if greedy:
            actions = mu # greedy action we only take mu

        if reparametrize:
            # in this case this is the same as doing: a = mu + sigma*eps where epsilon is gaussian with zero mean and unity variance.
            # Better as it allows gradient to flow through the sampling, better for learning.
            actions = probabilities.rsample() # this one adds noise to the action
        else:
            actions = probabilities.sample()
        
        action = torch.tanh(actions)*torch.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        squash_correction = torch.log(torch.clamp(1 - action.pow(2) + self.reparam_noise, min=1e-6))
        log_probs -= squash_correction
        log_probs = log_probs.sum(1, keepdim = True)

        return action, log_probs
    
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

class Value(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        return self.net(state)


class SAC:
    def __init__(self, env, 
                 env_name,
                 state_dim, 
                 action_dim,
                 episodes = 100, 
                 num_steps = 512,
                 scale = 2,
                 tau = 0.005,
                 max_actions = None,
                 alpha = 0.3,
                 batch_size = 128,
                 eval_interval = 5):
        self.actor = Actor(state_dim=state_dim, action_dim=action_dim, max_actions = max_actions)
        self.critic1 = Critic(state_dim=state_dim, action_dim=action_dim)
        self.critic2 = Critic(state_dim=state_dim, action_dim=action_dim)
        self.value = Value(state_dim=state_dim)
        self.target_value = Value(state_dim=state_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=1e-3)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=1e-3)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=1e-3)

        self.total_it = 0 
        self.scale = scale
        self.alpha = alpha

        # Training Hyperparameters:

        self.batch_size = batch_size
        self.eval_interval = eval_interval
        self.env = env
        self.env_name = env_name
        self.episodes = episodes
        self.num_steps = num_steps

        self.update_network_parameters(tau=tau)

    def update_network_parameters(self, tau=0.005):
        """
        This function takes care of the target value network parameter update.
        In the paper it is said that the target value parameters are an exponential moving
        average of the value network parameters.
        In this function this is what we do, we initialize tau which is the MA parameter
        then we load both V and target_V parameters as state_dicts and update target_V params as shown in the 
        original paper. 
        """
        
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + (1-tau)*target_value_state_dict[name].clone()
        self.target_value.load_state_dict(value_state_dict)
    
    def select_action(self, state, greedy = False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        actions, _ = self.actor.sample_normal(state, reparametrize=True, greedy = greedy)
        return actions.cpu().data.numpy().flatten()
    
    def train(self, buffer, batch_size = 128, gamma = 0.99):
        self.total_it+=1

        state, action, reward, next_state, done = buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device).view(-1)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.tensor(done, dtype=bool).to(device).view(-1)

        # --- Value network update ---
        value = self.value(state).view(-1)
        target_value = self.target_value(next_state).view(-1)
        target_value = torch.where(done, torch.zeros_like(target_value), target_value)

        actions, log_probs = self.actor.sample_normal(state, reparametrize=False)
        log_probs = log_probs.view(-1)

        Q1_new = self.critic1(state, actions).view(-1)
        Q2_new = self.critic2(state, actions).view(-1)
        critic_min = torch.min(Q1_new, Q2_new).view(-1)

        value_target = critic_min - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target.detach())  # detach target

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step() 

        # --- Actor network update ---
        actions, log_probs = self.actor.sample_normal(state, reparametrize=True)
        log_probs = log_probs.view(-1)
        Q1_new = self.critic1(state, actions).view(-1)
        Q2_new = self.critic2(state, actions).view(-1)
        critic_min = torch.min(Q1_new, Q2_new)

        # actor_loss = F.mse_loss(log_probs, critic_min.view(-1))
        actor_loss = (self.alpha*log_probs - critic_min.view(-1)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Critic networks update ---
        q_hat = self.scale * reward + gamma * self.target_value(next_state).view(-1).detach()
        q_hat = torch.where(done, reward, q_hat)  # no future reward if done

        q1_pred = self.critic1(state, action).view(-1)
        q2_pred = self.critic2(state, action).view(-1)

        critic1_loss = 0.5 * F.mse_loss(q1_pred, q_hat)
        critic2_loss = 0.5 * F.mse_loss(q2_pred, q_hat)
        critic_loss = critic1_loss + critic2_loss

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        self.update_network_parameters()
    
    def train_with_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.env.reset(seed=seed)

        buffer = ReplayBuffer()
        exploration_noise_start = 2

        returns = {}
        eval_returns = {}
        episode_times = {}

        action_dim = self.env.action_space.shape[0]
        max_action = float(self.env.action_space.high[0])


        step_counter = 0
        for ep in range(self.episodes):
            state, _ = self.env.reset()
            start_time = time.time()
            total_reward = 0
            for n_step in range(self.num_steps):
                action = self.select_action(state)
                exploration_noise = exploration_noise_start * (self.episodes - ep) / self.episodes
                # action = (action + np.random.normal(0, exploration_noise, size=action_dim)).clip(-max_action, max_action)
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

            print(f"Episode {ep}, Return: {total_reward:.2f}, average time: {np.mean(np.array(list(episode_times.values()))):.2f}")
            step_counter += 1 
            end_time = time.time() - start_time
            episode_times[step_counter] = end_time
            returns[step_counter] = total_reward

            n_eval = 5

            if ep % self.eval_interval == self.eval_interval - 1:
                total_reward = 0
                for _ in range(n_eval):
                    state, _ = self.env.reset()
                    for n_step in range(self.num_steps):
                        action = self.select_action(state, greedy = True)
                        next_state, reward, terminated, truncated, _ = self.env.step(action)
                        done = terminated or truncated
                        state = next_state
                        total_reward += reward
                        if done:
                            break
                total_reward = total_reward / n_eval
                print(f"Episode {ep}, Eval return: {total_reward:.2f}")
                eval_returns[step_counter] = total_reward

        # Plot
        plt.plot(returns.keys(), returns.values())
        plt.plot(eval_returns.keys(), eval_returns.values())
        plt.title("SAC Training on "+self.env_name)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid()
        plt.show()

        return returns, eval_returns

# def train_agent_sac(agent, env,env_name,episodes, batch_size, num_steps):
#     buffer = ReplayBuffer()
#     returns = []

#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.shape[0]
#     max_action = float(env.action_space.high[0])

#     exploration_noise_start = agent.exploration_noise

#     for ep in range(episodes):
#         state, _ = env.reset()
#         total_reward = 0
#         for n_step in range(num_steps):
#             exploration_noise = exploration_noise_start * (1 - ep / episodes)
#             action = agent.select_action(state)
#             # noise = np.random.normal(0, exploration_noise, size=action_dim)
#             # action = (action + noise).clip(-max_action, max_action)
#             next_state, reward, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated
#             buffer.push(state, action, reward, next_state, done)
#             state = next_state
#             total_reward += reward

#             if len(buffer) > 1000:
#                 agent.train(buffer, batch_size=batch_size)

#         returns.append(total_reward)
#         print(f"Episode {ep}, Return: {total_reward:.2f}")

#     # Plot
#     plt.plot(returns)
#     plt.title("SAC Training on "+env_name)
#     plt.xlabel("Episode")
#     plt.ylabel("Total Reward")
#     plt.grid()
#     plt.show()


