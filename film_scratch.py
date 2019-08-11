import numpy as np
import torch
from torch import nn
import gym
import SEVN_gym
from SEVN_gym.envs import utils, wrappers


env = gym.make("SEVN-Mini-All-Shaped-v1")
house_numbers, street_names = env.sample_nearby()
house_numbers = [utils.convert_house_numbers(num) for num in house_numbers]
street_names = [utils.convert_street_name(name, env.all_street_names) for name in street_names]
src_dim = len(house_numbers[0])# + len(street_names[0])

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(src_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

# transformer_model = nn.Transformer(src_vocab, tgt_vocab, num_encoder_layers=12)
model = MLP()
torch.FloatTensor(house_numbers[0]).unsqueeze(1).transpose(0,1)

model(torch.FloatTensor(house_numbers[0]).squeeze(1))

import pdb; pdb.set_trace()
print(house_numbers)
