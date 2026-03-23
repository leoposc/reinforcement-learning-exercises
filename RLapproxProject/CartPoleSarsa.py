#!/usr/bin/env python
import gymnasium as gym
from torch import nn
import DiscreteSarsaAgent as rl
import torch

env = gym.make('CartPole-v1', max_episode_steps=100)
dimObs = env.observation_space.shape[0]

class Q(nn.Module):
    def __init__(self):
        super(Q, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(dimObs, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        return self.nn(x)


import sys
run = int(sys.argv[1]) if len(sys.argv) == 2 else None

# Play with gamma, alpha, and perhaps other pararameters:
agent = rl.Agent(Q, env.action_space.n, gamma=1, epsilon=0.3, alpha=0.0001, nEpisodes=5000, epsanneal=5000)

agent.train(env)
if run is not None:
    agent.save(f"CartPoleSarsa-{run:02d}.npy")
