import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from torch.distributions import Categorical


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.t_conv1(x))
        x = torch.sigmoid(self.t_conv2(x))
        return x

    def extraction(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x)
        return x


class Actor_network(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Actor_network, self).__init__()

        self.shared_weights = nn.Sequential(
            nn.Linear(num_inputs, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU()
        )

        self.agent1 = nn.Sequential(
            nn.Linear(200, num_outputs),
            nn.Softmax()
        )

        self.agent2 = nn.Sequential(
            nn.Linear(200, num_outputs),
            nn.Softmax()
        )

    def forward(self, x):
        shared = self.shared_weights(x)
        agent1_p = self.agent1(shared)
        agent2_p = self.agent2(shared)
        dist_cat_1 = Categorical(agent1_p)
        dist_cat_2 = Categorical(agent2_p)
        return dist_cat_1, dist_cat_2

    def save(self):
        torch.save(self.network, 'neural-network-actor.pth')


class Critic_network(nn.Module):
    def __init__(self, num_inputs):
        super(Critic_network, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

    def forward(self, x):
        return self.critic(x)

    def save(self):
        torch.save(self.critic, 'neural-network-critic.pth')


class ExperienceReplayBuffer(object):
    def __init__(self, maximum_length=1000):
        self.buffer = deque(maxlen=maximum_length)

    def append(self, experience):
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)

    def all_samples(self):
        batch = [self.buffer[i] for i in range(len(self.buffer))]

        states, actions1, actions2, rewards = zip(*batch)

        rewards = torch.from_numpy(np.vstack([r for r in rewards])).float()
        states = torch.from_numpy(np.vstack([s for s in states])).float()
        actions1 = torch.from_numpy(np.vstack([a for a in actions1])).float()
        actions2 = torch.from_numpy(np.vstack([a for a in actions2])).float()

        return states, actions1, actions2, rewards


class PPO:
    def __init__(self):
        self.discount_factor = 0.99  # Value of gamma
        self.epsilon = 0.2
        self.num_steps = 10
        self.step_count = 0
        self.steps_per_game = []
        self.ppo_epochs = 4
        self.minibatch_size = 20
        self.action_dim = 5
        self.state_dim = 256
        self.buffer_size = 10000
        self.lr_actor = 3e-4
        self.lr_critic = 1e-4
        self.c1 = 0.5  # Critic
        self.c2 = 0.001  # Exploration
        self.buffer = ExperienceReplayBuffer(maximum_length=self.buffer_size)
        self.initialize_networks()

    def initialize_networks(self):
        self.actor_network = Actor_network(self.state_dim, self.action_dim)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.lr_actor)

        self.critic_network = Critic_network(self.state_dim)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.lr_critic)

        self.feature_extractor = ConvAutoencoder() # TODO - Load weights

        self.load_weights()

    def load_weights(self):
        try:
            checkpoint = torch.load('neural-network.pth')
            self.actor_network.load_state_dict(checkpoint['network_actor_state_dict'])
            self.critic_network.load_state_dict(checkpoint['network_critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['optimizer_actor_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['optimizer_critic_state_dict'])
            print("Loaded previous model")
        except:
            print("Error loading model")

    def save_weights(self):
        try:
            torch.save({
                'network_actor_state_dict': self.actor_network.state_dict(),
                'network_critic_state_dict': self.critic_network.state_dict(),
                'optimizer_actor_state_dict': self.actor_optimizer.state_dict(),
                'optimizer_critic_state_dict': self.critic_optimizer.state_dict()
            }, 'neural-network.pth')
            print("Model saved")
        except:
            print("Error saving the model")

    def process_state(self, gameState):
        state_representation = torch.rand(1, 3, 32, 32)
        state = self.feature_extractor.extraction(state_representation)
        state = torch.reshape(state, (1, -1))
        return state

    def compute_action(self, state, l1, l2):
        dist1, dist2 = self.actor_network.forward(state)
        action1 = dist1.sample().numpy()[0]
        action2 = dist2.sample().numpy()[0]
        if action1 not in l1 and action2 not in l2:
            return [action1, action2], 200, None
        elif action1 not in l1:
            return [action1, action2], 100, 0
        elif action2 not in l2:
            return [action1, action2], 100, 1
        return [action1, action2], None, None

    def store_experience(self, state, action1, action2, reward, done):
        self.step_count += 1
        exp = (state, action1, action2, reward)
        self.buffer.append(exp)
        if done:
            self.steps_per_game.append(self.step_count)
            self.step_count = 0
        if len(self.steps_per_game) == self.num_steps:
            self.train()

    def compute_log_probabilities(self, states, actions1, actions2):
        dist1, dist2 = self.actor_network(states)
        return dist1.log_prob(actions1)*dist2.log_prob(actions2)

    def compute_target_value(self, rewards):
        y = []
        start_idx = 0
        for t in self.steps_per_game:
            temp_y = [np.sum([self.discount_factor ** (n - e) * rewards[n] for n in range(e+start_idx, t+start_idx)]) for e in range(start_idx, t+start_idx)]
            start_idx += t
            y += temp_y
        y = torch.tensor([y], requires_grad=False, dtype=torch.float32)
        y = torch.reshape(y, (-1, 1))
        return y

    def train(self):
        states, actions1, actions2, rewards = self.buffer.all_samples()
        value_functions = self.critic_network(states)
        old_log_probs = self.compute_log_probabilities(states, actions1, actions2)
        y = self.compute_target_value(rewards)
        y = y.detach()
        old_log_probs = old_log_probs.detach()
        value_functions = value_functions.detach()
        advantage_estimation = y - value_functions
        self.ppo_update_split(states, actions1, actions2, old_log_probs, y, advantage_estimation)
        self.buffer = ExperienceReplayBuffer(maximum_length=self.buffer_size)

    def ppo_iter(self, states, actions1, actions2, log_probs, ys, advantage):
        batch_size = states.size(0)
        for _ in range(batch_size // self.minibatch_size):
            rand_ids = np.random.randint(0, batch_size, self.minibatch_size)
            yield states[rand_ids, :], actions1[rand_ids, :], actions2[rand_ids, :], log_probs[rand_ids, :], ys[rand_ids, :], advantage[
                                                                                                      rand_ids, :]

    def ppo_update_split(self, states, actions1, actions2, log_probs, ys, advantages):
        for _ in range(self.ppo_epochs):
            for state_, action_, old_log_prob_, y_, advantage_ in self.ppo_iter(states, actions1, actions2, log_probs, ys,
                                                                                advantages):
                value_ = self.critic_network(state_)
                dist1_, dist2_ = self.actor_network(state_)
                entropy_ = dist1_.entropy().mean()*dist2_.entropy().mean()
                new_log_prob_ = dist1_.log_prob(action_)*dist2_.log_prob(action_)

                ratio = (new_log_prob_ - old_log_prob_).exp()
                surr1 = ratio * advantage_
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantage_

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (y_ - value_).pow(2).mean()
                actor_loss -= self.c2 * entropy_

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

    def random_actions(self, l1, l2):
        return [random.choice(l1), random.choice(l2)], 100
