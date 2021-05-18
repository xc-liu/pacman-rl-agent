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



def convert_state_to_tensor(state):
    state_representation = torch.tensor(state, dtype=torch.float32)
    state_representation = torch.reshape(state_representation, (1, 6, 18, 34))
    return state_representation

class Actor_network(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Actor_network, self).__init__()

        # feature extraction: input 6 * 16 * 32 + 2, output: 256 + 2
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1),
            # nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.MaxPool2d(2, 2)
        )

        self.feature_extraction2 = nn.Sequential(
            nn.Conv2d(6, 18, 3, stride=1),
            nn.Conv2d(18, 18, 3, stride=1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(18, 18, 3, stride=1),
            nn.Conv2d(18, 18, 3, stride=1),
            nn.AvgPool2d(2, 2),
            nn.Flatten()
        )

        self.shared_weights = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )

        self.agent1 = nn.Sequential(
            nn.Linear(64, num_outputs),
            nn.Softmax(dim=0)
        )

        self.agent2 = nn.Sequential(
            nn.Linear(64, num_outputs),
            nn.Softmax(dim=0)
        )

    def forward(self, states, scores, times, agents):
        x = self.feature_extraction2(states)
        x = torch.cat((x, scores, times), dim=1)
        shared = self.shared_weights(x)

        outputs = torch.tensor([])
        for i in range(len(agents)):
            if agents[i] == 0:
                agent_out= self.agent1(shared[i])
            else:
                agent_out = self.agent2(shared[i])
            outputs = torch.cat((outputs, agent_out), 0)
        return Categorical(outputs)


class Critic_network(nn.Module):
    def __init__(self, num_inputs):
        super(Critic_network, self).__init__()

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1),
            # nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.MaxPool2d(2, 2)
        )

        self.feature_extraction2 = nn.Sequential(
            nn.Conv2d(6, 18, 3, stride=1),
            nn.Conv2d(18, 18, 3, stride=1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(18, 18, 3, stride=1),
            nn.Conv2d(18, 18, 3, stride=1),
            nn.AvgPool2d(2, 2),
            nn.Flatten()
        )


        self.critic = nn.Sequential(
            nn.Linear(num_inputs, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self, states, scores, times):
        x = self.feature_extraction2(states)
        x = torch.cat((x, scores, times), dim=1)
        return self.critic(x)


class ExperienceReplayBuffer(object):
    def __init__(self, maximum_length=1000):
        self.buffer = deque(maxlen=maximum_length)

    def append(self, experience):
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)

    def change_last_reward(self, reward):
        self.buffer[-1] = (*self.buffer[-1][:-2], True, reward)

    def all_samples(self):
        batch = [self.buffer[i] for i in range(len(self.buffer))]

        states, scores, times, actions, agents, dones, rewards = zip(*batch)

        rewards = torch.from_numpy(np.vstack([r for r in rewards])).float()
        actions = torch.from_numpy(np.vstack([a for a in actions])).float()
        scores = torch.from_numpy(np.vstack([s for s in scores])).float()
        times = torch.from_numpy(np.vstack([t for t in times])).float()
        agents = torch.from_numpy(np.vstack([a for a in agents])).int()
        dones = torch.from_numpy(np.vstack([d for d in dones]).astype(np.uint8)).float()

        return states, scores, times, actions, agents, dones, rewards


class PPO:
    def __init__(self, training_agent=False):
        self.discount_factor = 0.99
        self.GAE_gamma = 0.95
        self.epsilon = 0.2
        # self.num_steps = 4 # 8
        self.exp_to_learn = 2400
        self.step_count = 0
        self.steps_per_game = []
        self.ppo_epochs = 10
        self.minibatch_size = 64
        self.action_dim = 5
        self.state_dim = 92
        self.buffer_size = 4000
        self.lr_actor = 3e-4
        self.lr_critic = 3e-4
        self.c2 = 3e-4  # Exploration
        self.input_clipping = 10

        self.buffer = ExperienceReplayBuffer(maximum_length=self.buffer_size)
        self.training_agent = training_agent
        self.reward_count = 0
        self.rewards_games = []

        self.target_value_mean = 0.0
        self.target_value_squared_mean = 0.0
        self.target_value_std = 0.0
        self.training_samples = 0

        self.observation_mean = np.zeros(shape=(6, 18, 34))
        self.observation_squared_mean = np.zeros(shape=(6, 18, 34))
        self.time_mean = self.score_mean = self.time_squared_mean = self.score_squared_mean = 0

        self.next_state = None

        self.initialize_networks()


    def initialize_networks(self):

        self.actor_network = Actor_network(self.state_dim, self.action_dim)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.lr_actor)

        self.critic_network = Critic_network(self.state_dim)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.lr_critic)

        print()
        self.load_weights()
        print()

    def load_weights(self):
        try:
            file = 'neural-network_2.pth'
            if self.training_agent:
                agent_options = os.listdir('past_agents_2')
                # file = random.choice(agent_options)
                file = agent_options[-1]
                file = 'past_agents_2/' + file
            checkpoint = torch.load(file)
            self.actor_network.load_state_dict(checkpoint['network_actor_state_dict'])
            self.critic_network.load_state_dict(checkpoint['network_critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['optimizer_actor_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['optimizer_critic_state_dict'])
            self.target_value_mean, self.target_value_squared_mean, self.target_value_std,\
            self.observation_mean, self.observation_squared_mean, self.time_mean, self.score_mean,\
            self.time_squared_mean, self.score_squared_mean, self.training_samples = checkpoint['previous_info']
            print("Loaded previous model")
        except:
            print("Error loading model")

    def save_weights(self):
        try:
            previous_info = [self.target_value_mean, self.target_value_squared_mean, self.target_value_std,
                             self.observation_mean, self.observation_squared_mean, self.time_mean,
                             self.score_mean, self.time_squared_mean, self.score_squared_mean, self.training_samples]
            torch.save({
                'network_actor_state_dict': self.actor_network.state_dict(),
                'network_critic_state_dict': self.critic_network.state_dict(),
                'optimizer_actor_state_dict': self.actor_optimizer.state_dict(),
                'optimizer_critic_state_dict': self.critic_optimizer.state_dict(),
                'previous_info': previous_info
            }, 'neural-network_2.pth')

            # torch.save({
            #     'network_actor_state_dict': self.actor_network.state_dict(),
            #     'network_critic_state_dict': self.critic_network.state_dict(),
            #     'optimizer_actor_state_dict': self.actor_optimizer.state_dict(),
            #     'optimizer_critic_state_dict': self.critic_optimizer.state_dict(),
            #     'previous_info': previous_info
            # }, 'past_agents_2/neural-network_' + str(time.time()) + '.pth')

            # print("Model saved")
        except:
            print("Error saving the model")


    def compute_action(self, state, l, agent):
        state_rep, score, time_ = state
        state_rep, score, time_ = self.normalize_state(state_rep, score, time_)
        state_rep = torch.reshape(state_rep, (1, 6, 18, 34))
        score = torch.reshape(score, (-1, 1))
        time_ = torch.reshape(time_, (-1, 1))
        dist = self.actor_network.forward(state_rep, score, time_, [agent])
        action = int(dist.sample().numpy())
        if action not in l:
            return action, 100
        return action, None

    def last_experience_reward(self, reward):
        self.buffer.change_last_reward(reward)

    def store_experience(self, exp):
        self.training_samples += 1
        self.step_count += 1
        self.reward_count += exp[-2]
        self.buffer.append(exp[:-1])
        self.next_state = exp[-1]
        if exp[-3]:
            self.steps_per_game.append(self.step_count)
            # print("Current game ", len(self.steps_per_game))
            self.step_count = 0
            self.rewards_games.append(self.reward_count)
            self.reward_count = 0
            if len(self.buffer) >= self.exp_to_learn:
                self.mean_rewards = np.mean(self.rewards_games[-20:])
                print("Game - %d, Reward - %.2f " % (len(self.steps_per_game), self.mean_rewards), end='\r')
                self.train()
                self.save_weights()
            # if len(self.steps_per_game)%50 == 0:
            #     self.save_weights()
            # if len(self.steps_per_game)==50:
            #     plt.plot(self.rewards_games)
            #     plt.show()


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
        y = self.normalize_target_value(y)
        return y

    def normalize_state(self, state, score, time_):
        observation_std = (self.observation_squared_mean - self.observation_mean ** 2) ** 0.5
        time_std = (self.time_squared_mean - self.time_mean ** 2) ** 0.5
        score_std = (self.score_squared_mean - self.score_mean ** 2) ** 0.5

        digital_state = (state - self.observation_mean) / np.clip(observation_std, a_min=1e-6, a_max=None)
        score = (score - self.score_mean) / max(score_std, 1e-6)
        time_ = (time_ - self.time_mean) / max(time_std, 1e-6)

        digital_state = np.clip(digital_state, a_min=-self.input_clipping, a_max=self.input_clipping)
        score = float(np.clip(score, a_min=-self.input_clipping, a_max=self.input_clipping))
        time_ = float(np.clip(time_, a_min=-self.input_clipping, a_max=self.input_clipping))

        digital_state = convert_state_to_tensor(digital_state)
        time_ = torch.tensor(time_, dtype=torch.float32)
        score = torch.tensor(score, dtype=torch.float32)

        return digital_state, score, time_

    def compute_gae(self, values, rewards, dones):
        self.next_state = self.normalize_state(self.next_state[0], self.next_state[1], self.next_state[2])

        state_rep = torch.reshape(self.next_state[0], (1, 6, 18, 34))
        score = torch.reshape(self.next_state[1], (-1, 1))
        time_ = torch.reshape(self.next_state[2], (-1, 1))
        next_value = self.critic_network(state_rep, score, time_)

        next_value = self.de_normalize_target_value(next_value)
        masks = 1 - np.array(dones)
        values = torch.cat((values, next_value), 0).detach().numpy()
        rewards = rewards.numpy()
        gae = 0
        ys = np.zeros(len(rewards))
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.discount_factor * values[step + 1] * masks[step] - values[step]
            gae = delta + self.discount_factor * self.GAE_gamma * masks[step] * gae
            ys[step] = gae + values[step]
        ys = torch.tensor(ys)
        ys = torch.reshape(ys, (-1, 1))
        return ys

    def normalize_target_value(self, y):
        percentage = (len(y)/self.training_samples)
        self.target_value_mean = self.target_value_mean*(1-percentage) + y.mean() * percentage
        self.target_value_squared_mean = self.target_value_squared_mean*(1-percentage) + torch.square(y).mean() * percentage
        self.target_value_std = torch.clamp(torch.sqrt(self.target_value_squared_mean - torch.square(self.target_value_mean)), min=1e-6)
        y = (y-self.target_value_mean)/self.target_value_std
        return y

    def normalize_value_functions(self, value_functions):
        return (value_functions - self.target_value_mean) / self.target_value_std

    def de_normalize_target_value(self, y):
        y = y*self.target_value_std+self.target_value_mean
        return y

    def normalize_all_observation(self, digital_state, score, time_):
        percentage = len(time_) / self.training_samples
        self.observation_mean = self.observation_mean * (1 - percentage) + np.average(digital_state, axis=0) * percentage
        self.observation_squared_mean = self.observation_squared_mean * (1 - percentage) + np.average(np.square(digital_state), axis=0) * percentage
        self.time_mean = self.time_mean * (1 - percentage) + np.average(time_) * percentage
        self.time_squared_mean = self.time_squared_mean * (1 - percentage) + np.average(np.square(time_)) * percentage
        self.score_mean = self.score_mean * (1 - percentage) + np.average(score) * percentage
        self.score_squared_mean = self.score_squared_mean * (1 - percentage) + np.average(np.square(score)) * percentage

        observation_std = (self.observation_squared_mean - self.observation_mean ** 2) ** 0.5
        time_std = (self.time_squared_mean - self.time_mean ** 2) ** 0.5
        score_std = (self.score_squared_mean - self.score_mean ** 2) ** 0.5

        digital_state = (np.array(digital_state) - self.observation_mean) / np.clip(observation_std, a_min=1e-6, a_max=None)
        score = (np.array(score) - self.score_mean) / max(score_std, 1e-6)
        time_ = (np.array(time_) - self.time_mean) / max(time_std, 1e-6)

        digital_state = np.clip(digital_state, a_min=-self.input_clipping, a_max=self.input_clipping)
        score = np.clip(score, a_min=-self.input_clipping, a_max=self.input_clipping)
        time_ = np.clip(time_, a_min=-self.input_clipping, a_max=self.input_clipping)

        time_ = time_.astype('float32')
        score = score.astype('float32')

        time_ = torch.tensor(time_, dtype=torch.float32)
        score = torch.tensor(score, dtype=torch.float32)


        states = torch.tensor([])
        for i in range(len(digital_state)):
            states = torch.cat((states, convert_state_to_tensor(digital_state[i])), 0)
        return states, score, time_


    def train(self):
        states, scores, times, actions, agents, dones, rewards = self.buffer.all_samples()

        states, scores, times = self.normalize_all_observation(states, scores, times)

        value_functions = self.critic_network(states, scores, times)
        value_functions = torch.reshape(value_functions, (-1, 1))

        old_log_probs = self.actor_network(states, scores, times, agents).log_prob(actions)
        old_log_probs = torch.reshape(old_log_probs, (-1, 1))

        value_functions = self.de_normalize_target_value(value_functions)
        y = self.compute_gae(value_functions, rewards, dones)
        y = self.normalize_target_value(y)
        y = y.detach()

        old_log_probs = old_log_probs.detach()
        value_functions = value_functions.detach()
        value_functions = self.normalize_value_functions(value_functions)

        advantage_estimation = y - value_functions
        self.ppo_update_split(states, scores, times, actions, agents, old_log_probs, y, advantage_estimation)
        self.buffer = ExperienceReplayBuffer(maximum_length=self.buffer_size)

    def ppo_iter(self, states, scores, times, actions, agents, log_probs, ys, advantage):
        batch_size = len(states)
        for _ in range(batch_size // self.minibatch_size):
            rand_ids = np.random.randint(0, batch_size, self.minibatch_size)
            yield states[rand_ids, :], scores[rand_ids, :], times[rand_ids, :], actions[rand_ids, :], agents[rand_ids, :], \
                  log_probs[rand_ids, :], ys[rand_ids, :], advantage[rand_ids, :]

    def ppo_update_split(self, states, scores, times, actions, agents, log_probs, ys, advantages):
        # actor_loss = 0
        # critic_loss = 0
        for _ in range(self.ppo_epochs):
            for state_, score_, time_, action_, agent_, old_log_prob_, y_, advantage_ in self.ppo_iter(states, scores, times, actions, agents,
                                                                                        log_probs,
                                                                                        ys,
                                                                                        advantages):

                value_ = self.critic_network(state_, score_, time_)
                value_ = torch.reshape(value_, (-1, 1))

                dist_ = self.actor_network(state_, score_, time_, agent_)
                entropy_ = dist_.entropy().mean()
                new_log_prob_ = dist_.log_prob(action_)
                new_log_prob_ = torch.reshape(new_log_prob_, (-1, 1))

                ratio = (new_log_prob_ - old_log_prob_).exp()
                surr1 = ratio * advantage_
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantage_

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (y_ - value_).pow(2).mean()
                actor_loss -= self.c2 * entropy_


                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_network.parameters(), max_norm=1.)
                self.critic_optimizer.step()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_network.parameters(), max_norm=1.)
                self.actor_optimizer.step()
        # print("Critic loss ", critic_loss)
        # print("Actor loss ", actor_loss)
