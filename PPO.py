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
from score_keeper import return_number_games, save_timesteps

from captum.attr import IntegratedGradients

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def visualize_importances(feature_names, importances, title="Average Feature Importances", plot=True,
                          axis_title="Features"):
    x_pos = (np.arange(len(feature_names)))
    if plot:
        plt.figure(figsize=(12, 6))
        plt.bar(x_pos, importances, align='center')
        plt.xticks(x_pos, feature_names, wrap=True)
        plt.xlabel(axis_title)
        plt.title(title)
        plt.show()


def convert_state_to_tensor(state):
    state_representation = torch.tensor(state, dtype=torch.float32)
    state_representation = torch.reshape(state_representation, (1, -1)).cuda()
    return state_representation


def convert_states_to_tensors(states):
    tensor_states = torch.tensor([]).cuda()
    for i in range(len(states)):
        tensor_states = torch.cat((tensor_states, convert_state_to_tensor(states[i])), 0)
    return tensor_states


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0.01)

class Actor_network(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Actor_network, self).__init__()

        self.agent = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, num_outputs),
            nn.Softmax(dim=1)
        )

        self.agent.apply(init_weights)

    def forward(self, x):
        agent = self.agent(x)
        dist = Categorical(agent)
        # return dist.probs
        return dist


class Critic_network(nn.Module):
    def __init__(self, num_inputs):
        super(Critic_network, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.critic.apply(init_weights)

    def forward(self, x):
        result = self.critic(x)
        return result


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

        states, actions, dones, rewards = zip(*batch)

        rewards = torch.from_numpy(np.vstack([r for r in rewards])).float().cuda()
        actions = torch.from_numpy(np.vstack([a for a in actions])).float().cuda()
        dones = torch.from_numpy(np.vstack([d for d in dones]).astype(np.uint8)).float().cuda()

        return states, actions, dones, rewards


class PPO:
    def __init__(self, training_agent=False):
        self.discount_factor = 0.99
        self.GAE_gamma = 0.95
        self.epsilon = 0.2
        self.exp_to_learn = 10000
        self.step_count = 0
        self.steps_per_game = []
        self.ppo_epochs = 10
        self.minibatch_size = 256
        self.action_dim = 5
        # self.state_dim = 1476
        self.state_dim = 386
        self.buffer_size = 15000
        self.lr_actor = 1e-4
        self.lr_critic = 3e-4
        self.c2 = 3e-4  # Exploration

        self.buffer = ExperienceReplayBuffer(maximum_length=self.buffer_size)
        self.training_agent = training_agent
        self.reward_count = 0
        self.rewards_games = []
        self.mean_rewards_games = []

        self.target_value_mean = 0.0
        self.target_value_squared_mean = 0.0
        self.target_value_std = 0.0
        self.training_samples = 0

        self.number_games = 0

        self.saving_frequency = return_number_games()
        self.initial_frequency = int(self.saving_frequency)
        self.do_plotting = False if self.saving_frequency > 1200 else True
        while self.saving_frequency > 400:
            if self.saving_frequency % 2 == 0:
                self.saving_frequency /= 2
            else:
                self.saving_frequency /= 3

        if self.initial_frequency > 400 and self.saving_frequency < 200:
            self.saving_frequency = 200

        self.saving_frequency = int(self.saving_frequency) - 1
        self.next_state = None

        self.initialize_networks()

    def initialize_networks(self):

        self.actor_network = Actor_network(self.state_dim, self.action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.lr_actor)

        self.critic_network = Critic_network(self.state_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.lr_critic)

        print()
        self.load_weights()
        print()

    def load_weights(self):
        try:
            file = 'neural-network_2.pth'
            if self.training_agent:
                agent_options = os.listdir('past_agents_2')
                file = random.choice(agent_options)
                file = 'self-play-agents/' + file
            checkpoint = torch.load(file)
            self.actor_network.load_state_dict(checkpoint['network_actor_state_dict'])
            self.critic_network.load_state_dict(checkpoint['network_critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['optimizer_actor_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['optimizer_critic_state_dict'])
            self.target_value_mean, self.target_value_squared_mean, self.target_value_std, \
            self.training_samples = checkpoint['previous_info']
            # self.target_value_mean, self.target_value_squared_mean, self.target_value_std, \
            # self.training_samples, self.number_games = checkpoint['previous_info']
            print("Loaded previous model ", int(self.training_samples / self.exp_to_learn))

        except:
            print("Error loading model")

    def save_weights(self):
        try:
            # previous_info = [self.target_value_mean, self.target_value_squared_mean, self.target_value_std,
            #                  self.training_samples, self.number_games]
            previous_info = [self.target_value_mean, self.target_value_squared_mean, self.target_value_std,
                             self.training_samples]
            torch.save({
                'network_actor_state_dict': self.actor_network.state_dict(),
                'network_critic_state_dict': self.critic_network.state_dict(),
                'optimizer_actor_state_dict': self.actor_optimizer.state_dict(),
                'optimizer_critic_state_dict': self.critic_optimizer.state_dict(),
                'previous_info': previous_info
            }, 'neural-network_2.pth')

            if random.uniform(0, 1) > 0.7:
                torch.save({
                    'network_actor_state_dict': self.actor_network.state_dict(),
                    'network_critic_state_dict': self.critic_network.state_dict(),
                    'optimizer_actor_state_dict': self.actor_optimizer.state_dict(),
                    'optimizer_critic_state_dict': self.critic_optimizer.state_dict(),
                    'previous_info': previous_info
                }, 'self-play-agents/neural-network_' + str(self.training_samples / self.exp_to_learn) + '.pth')

            print("Model saved")
        except:
            print("Error saving the model")


    def visualize_network(self, dist, original):
        state = convert_state_to_tensor(original)
        dist = dist.detach().cpu().numpy()[0]
        posibilities = [i for i in range(0, 5)]
        action = np.random.choice(posibilities, p=dist)
        if len(self.buffer) % 200 == 0 and len(self.buffer)>1:
            ig = IntegratedGradients(self.actor_network)

            test_input_tensor = state.requires_grad_()
            attr, delta = ig.attribute(test_input_tensor, target=1, return_convergence_delta=True)
            attr = attr.detach().cpu().numpy()[0]

            names = [str(original[i]) for i in range(len(attr))]
            names1 = ["p1x", "p1y", "p2x", "p2y", "p3x", "p3y", "pac1", "pac2", "pac3", "pac4", "food1", "food2", "food3", "food4", "sc1", "sc2", "sc3", "sc4", "score", "time"]

            visualize_importances(names, attr)
            visualize_importances(names1, attr[-20:])

        return action


    def compute_action(self, state, l):
        state_rep = convert_state_to_tensor(state)
        dist = self.actor_network.forward(state_rep)
        # action = self.visualize_network(dist, state)
        action = int(dist.sample().cpu().numpy())
        if action not in l:
            return action, 1
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
            # self.number_games += 1
            # save_timesteps(self.number_games)
            self.steps_per_game.append(self.step_count)
            self.step_count = 0
            self.rewards_games.append(self.reward_count)
            self.reward_count = 0
            if len(self.buffer) >= self.exp_to_learn:
                self.mean_rewards = np.mean(self.rewards_games[-100:])
                if len(self.steps_per_game) >= 100 or self.initial_frequency < 100: self.mean_rewards_games.append(
                    self.mean_rewards)
                print("Game - %d, Reward - %.2f " % (len(self.steps_per_game), self.mean_rewards), end='\r')
                # print("Game - %d, Reward - %.2f " % (len(self.steps_per_game), self.mean_rewards))
                self.train()
            if len(self.steps_per_game) % (self.saving_frequency - 1) == 0:
                self.save_weights()
            if len(self.steps_per_game) % (self.initial_frequency - 1) == 0 and self.do_plotting:
                self.save_weights()
                plt.plot(self.mean_rewards_games)
                plt.title("Mean reward is %.2f" % np.mean(self.mean_rewards_games))
                plt.savefig('plots/plot_%d'%int(self.training_samples / self.exp_to_learn))
                # plt.show()

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
        y = torch.reshape(y, (-1, 1)).cuda()
        # y = self.normalize_target_value(y)
        return y

    def compute_gae(self, values, rewards, dones):
        self.next_state = convert_state_to_tensor(self.next_state)
        next_value = self.critic_network(self.next_state)

        # next_value = self.de_normalize_target_value(next_value)
        masks = 1 - np.array(dones.cpu())
        values = torch.cat((values, next_value), 0).detach().cpu().numpy()
        rewards = rewards.cpu().numpy()
        gae = 0
        ys = np.zeros(len(rewards))
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.discount_factor * values[step + 1] * masks[step] - values[step]
            gae = delta + self.discount_factor * self.GAE_gamma * masks[step] * gae
            ys[step] = gae + values[step]
        ys = torch.tensor(ys)
        ys = torch.reshape(ys, (-1, 1)).cuda()
        return ys

    def normalize_target_value(self, y):
        percentage = (len(y) / self.training_samples)
        self.target_value_mean = self.target_value_mean * (1 - percentage) + y.mean() * percentage
        self.target_value_squared_mean = self.target_value_squared_mean * (1 - percentage) + torch.square(
            y).mean() * percentage
        self.target_value_std = torch.clamp(
            torch.sqrt(self.target_value_squared_mean - torch.square(self.target_value_mean)), min=1e-6)
        y = (y - self.target_value_mean) / self.target_value_std
        return y

    def normalize_value_functions(self, value_functions):
        return (value_functions - self.target_value_mean) / self.target_value_std

    def de_normalize_target_value(self, y):
        if self.target_value_std == 0.0: return y
        y = y * self.target_value_std + self.target_value_mean
        return y

    def train(self):
        states, actions, dones, rewards = self.buffer.all_samples()
        rewards *= 0.1

        actions = torch.reshape(actions, (-1,))
        states = convert_states_to_tensors(states)

        value_functions = self.critic_network(states)
        value_functions = torch.reshape(value_functions, (-1, 1)).cuda()
        # print("Value function; mean %.2f, max %.2f, min %.2f, std %.2f" % (
        #     float(value_functions.mean()), float(value_functions.max())
        #     , float(value_functions.min()), float(value_functions.std())))

        old_log_probs = self.actor_network(states).log_prob(actions)
        old_log_probs = torch.reshape(old_log_probs, (-1, 1)).cuda()

        # value_functions = self.de_normalize_target_value(value_functions)
        y = self.compute_gae(value_functions, rewards, dones)
        # print("GAE; mean %.2f, max %.2f, min %.2f, std %.2f" % (
        #     float(y.mean()), float(y.max()), float(y.min()), float(y.std())))
        # print()

        # y = self.normalize_target_value(y)
        y = y.detach()

        old_log_probs = old_log_probs.detach()
        value_functions = value_functions.detach()
        # value_functions = self.normalize_value_functions(value_functions)

        advantage_estimation = y - value_functions
        # exit()
        self.ppo_update_split(states, actions, old_log_probs, y, advantage_estimation)
        self.buffer = ExperienceReplayBuffer(maximum_length=self.buffer_size)

    def ppo_iter(self, states, actions, log_probs, ys, advantage):
        batch_size = len(states)
        for _ in range(batch_size // self.minibatch_size):
            rand_ids = np.random.randint(0, batch_size, self.minibatch_size)
            yield states[rand_ids, :], actions[rand_ids], log_probs[rand_ids, :], ys[rand_ids, :], advantage[
                                                                                                   rand_ids, :]

    def ppo_update_split(self, states, actions, log_probs, ys, advantages):
        # actor_loss = 0
        # critic_loss = 0
        for _ in range(self.ppo_epochs):
            for state_, action_, old_log_prob_, y_, advantage_ in self.ppo_iter(states, actions,
                                                                                log_probs,
                                                                                ys,
                                                                                advantages):
                value_ = self.critic_network(state_)
                value_ = torch.reshape(value_, (-1, 1)).cuda()

                dist_ = self.actor_network(state_)
                entropy_ = dist_.entropy().mean()
                new_log_prob_ = dist_.log_prob(action_)
                new_log_prob_ = torch.reshape(new_log_prob_, (-1, 1)).cuda()

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
