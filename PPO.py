import random
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.distributions import Categorical
import matplotlib.pyplot as plt


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
            nn.Softmax(dim=1)
        )

        self.agent2 = nn.Sequential(
            nn.Linear(200, num_outputs),
            nn.Softmax(dim=1)
        )

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1),
            # nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        state_representation, score, time_left = x
        extra_info = torch.tensor([score, time_left])
        state_representation = torch.tensor(state_representation, dtype=torch.float32)
        state_representation = torch.reshape(state_representation, (1, 6, 16, 32))

        x = self.feature_extraction(state_representation)
        x = torch.flatten(x)
        x = torch.cat((x, extra_info), 0)  # Include score and time to feature extraction
        x = torch.reshape(x, (1, -1))

        shared = self.shared_weights(x)
        agent1_p = self.agent1(shared)
        agent2_p = self.agent2(shared)
        dist_cat_1 = Categorical(agent1_p)
        dist_cat_2 = Categorical(agent2_p)
        return dist_cat_1, dist_cat_2

class Critic_network(nn.Module):
    def __init__(self, num_inputs):
        super(Critic_network, self).__init__()

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1),
            # nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.MaxPool2d(2, 2)
        )

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

    def forward(self, x):
        state_representation, score, time_left = x
        extra_info = torch.tensor([score, time_left])
        state_representation = torch.tensor(state_representation, dtype=torch.float32)
        state_representation = torch.reshape(state_representation, (1, 6, 16, 32))

        x = self.feature_extraction(state_representation)
        x = torch.flatten(x)
        x = torch.cat((x, extra_info), 0)  # Include score and time to feature extraction
        x = torch.reshape(x, (1, -1))

        return self.critic(x)


class ExperienceReplayBuffer(object):
    def __init__(self, maximum_length=1000):
        self.buffer = deque(maxlen=maximum_length)

    def append(self, experience):
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)

    def all_samples(self):
        batch = [self.buffer[i] for i in range(len(self.buffer))]

        states, actions1, actions2, dones, rewards = zip(*batch)

        rewards = torch.from_numpy(np.vstack([r for r in rewards])).float()
        actions1 = torch.from_numpy(np.vstack([a for a in actions1])).float()
        actions2 = torch.from_numpy(np.vstack([a for a in actions2])).float()
        dones = torch.from_numpy(np.vstack([d for d in dones]).astype(np.uint8)).float()

        return states, actions1, actions2, dones, rewards


class PPO:
    def __init__(self, training_agent=False):
        self.discount_factor = 0.99  # Value of gamma
        self.epsilon = 0.2
        self.num_steps = 2
        self.step_count = 0
        self.steps_per_game = []
        self.ppo_epochs = 5
        self.minibatch_size = 32
        self.action_dim = 5
        self.state_dim = 1026
        self.buffer_size = 10000
        self.lr_actor = 3e-4
        self.lr_critic = 1e-4
        # self.c1 = 0.5  # Critic
        self.c2 = 0.01  # Exploration
        self.buffer = ExperienceReplayBuffer(maximum_length=self.buffer_size)
        self.training_agent = training_agent
        self.reward_count = 0
        self.rewards_games = []

        self.initialize_networks()

        # self.to_save = []

    def initialize_networks(self):
        self.actor_network = Actor_network(self.state_dim, self.action_dim)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.lr_actor)

        self.critic_network = Critic_network(self.state_dim)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.lr_critic)

        # self.feature_extractor = ConvAutoencoder()

        self.load_weights()

    def load_weights(self):
        try:
            file = 'neural-network.pth'
            if self.training_agent:
                agent_options = os.listdir('past_agents')
                file = random.choice(agent_options)
            checkpoint = torch.load(file)
            self.actor_network.load_state_dict(checkpoint['network_actor_state_dict'])
            self.critic_network.load_state_dict(checkpoint['network_critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['optimizer_actor_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['optimizer_critic_state_dict'])
            print("Loaded previous model")
        except Exception as e:
            print(e)
            exit()
            print("Error loading model")

    def save_weights(self):
        try:
            torch.save({
                'network_actor_state_dict': self.actor_network.state_dict(),
                'network_critic_state_dict': self.critic_network.state_dict(),
                'optimizer_actor_state_dict': self.actor_optimizer.state_dict(),
                'optimizer_critic_state_dict': self.critic_optimizer.state_dict()
            }, 'neural-network.pth')

            torch.save({
                'network_actor_state_dict': self.actor_network.state_dict(),
                'network_critic_state_dict': self.critic_network.state_dict(),
                'optimizer_actor_state_dict': self.actor_optimizer.state_dict(),
                'optimizer_critic_state_dict': self.critic_optimizer.state_dict()
            }, 'past_agents/neural-network_' + str(time.time()) + '.pth')

            print("Model saved")
        except:
            print("Error saving the model")

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

    def store_experience(self, exp):
        self.step_count += 1
        self.reward_count += exp[4]
        self.buffer.append(exp)
        if exp[3]:
            self.steps_per_game.append(self.step_count)
            self.step_count = 0
            self.rewards_games.append(self.reward_count)
            if len(self.steps_per_game)%10==0:
                print("Game - %d, Reward - %.2f "%(len(self.steps_per_game), self.reward_count))
            self.reward_count = 0
            if len(self.steps_per_game) == self.num_steps:
                self.train()
            if len(self.steps_per_game)%250 == 0:
                self.save_weights()
            if len(self.steps_per_game)==500:
                plt.plot(self.rewards_games)
                plt.show()

    def compute_log_probabilities(self, states, actions1, actions2):
        log_probs = []
        for i in range(len(states)):
            dist1, dist2 = self.actor_network(states[i])
            combined = dist1.log_prob(actions1[i])*dist2.log_prob(actions2[i])
            log_probs.append(combined)
        return torch.tensor(log_probs)

    def compute_target_value(self, rewards):
        y = []
        start_idx = 0
        for t in self.steps_per_game:
            temp_y = [
                np.sum([self.discount_factor ** (n - e) * rewards[n] for n in range(e + start_idx, t + start_idx)]) for
                e in range(start_idx, t + start_idx)]
            start_idx += t
            y += temp_y
        y = torch.tensor([y], requires_grad=False, dtype=torch.float32)
        y = torch.reshape(y, (-1, 1))
        return y

    def train(self):
        states, actions1, actions2, _, rewards = self.buffer.all_samples()

        value_functions = torch.tensor([self.critic_network(state) for state in states])
        value_functions = torch.reshape(value_functions, (-1, 1))

        old_log_probs = self.compute_log_probabilities(states, actions1, actions2)
        old_log_probs = torch.reshape(old_log_probs, (-1, 1))

        y = self.compute_target_value(rewards)

        y = y.detach()
        old_log_probs = old_log_probs.detach()
        value_functions = value_functions.detach()

        advantage_estimation = y - value_functions
        self.ppo_update_split(states, actions1, actions2, old_log_probs, y, advantage_estimation)
        self.buffer = ExperienceReplayBuffer(maximum_length=self.buffer_size)

    def ppo_iter(self, states, actions1, actions2, log_probs, ys, advantage):
        batch_size = len(states)
        states = np.array(states, dtype=object)
        for _ in range(batch_size // self.minibatch_size):
            rand_ids = np.random.randint(0, batch_size, self.minibatch_size)
            yield states[rand_ids, :], actions1[rand_ids, :], actions2[rand_ids, :], \
                  log_probs[rand_ids, :], ys[rand_ids, :], advantage[rand_ids, :]

    def ppo_update_split(self, states, actions1, actions2, log_probs, ys, advantages):
        for _ in range(self.ppo_epochs):
            for state_, action1_, action2_, old_log_prob_, y_, advantage_ in \
                    self.ppo_iter(states, actions1, actions2, log_probs, ys, advantages):

                value_ = torch.tensor([self.critic_network(s) for s in state_], requires_grad=True)

                entropy_ = []
                new_log_prob_ = []
                for i in range(len(state_)):
                    d1, d2 = self.actor_network(state_[i])
                    e = d1.entropy() * d2.entropy()
                    combined = d1.log_prob(action1_[i]) * d2.log_prob(action2_[i])
                    entropy_.append(e)
                    new_log_prob_.append(combined)

                entropy_ = torch.tensor(entropy_, requires_grad=True).mean()
                new_log_prob_ = torch.tensor(new_log_prob_, requires_grad=True)
                new_log_prob_ = torch.reshape(new_log_prob_, (-1, 1))

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
