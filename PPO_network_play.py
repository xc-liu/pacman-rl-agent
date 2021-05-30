import random
import os
import torch
import torch.nn as nn
from torch.distributions import Categorical

def convert_state_to_tensor(state):
    state_representation = torch.tensor(state, dtype=torch.float32)
    state_representation = torch.reshape(state_representation, (1, -1))
    return state_representation

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

    def forward(self, x):
        agent = self.agent(x)
        dist = Categorical(agent)
        return dist

class PPO:
    def __init__(self, training_agent=False):
        self.action_dim = 5
        self.state_dim = 1476
        self.training_agent = training_agent
        self.actor_network = Actor_network(self.state_dim, self.action_dim)

    def load_weights(self):
        try:
            file = 'neural-network_2.pth'
            if self.training_agent:
                agent_options = os.listdir('self-play-agents')
                file = random.choice(agent_options)
                file = 'self-play-agents/' + file
            checkpoint = torch.load(file)
            self.actor_network.load_state_dict(checkpoint['network_actor_state_dict'])
            # print("Loaded previous model")
        except:
            print("Error loading model")

    def compute_action(self, state):
        state = convert_state_to_tensor(state)
        dist = self.actor_network.forward(state)
        action = int(dist.sample().numpy())
        return action, list(dist.probs.detach().numpy()[0])



