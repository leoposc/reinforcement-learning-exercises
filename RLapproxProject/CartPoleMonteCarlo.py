#!/usr/bin/env python
import gymnasium as gym
from torch import nn
import DiscreteMonteCarloAgent as rl


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
        return self.nn(x)


import sys
run = int(sys.argv[1]) if len(sys.argv) == 2 else None

# Play with gamma, alpha, and perhaps other pararameters:
agent = rl.Agent(Q, env.action_space.n, gamma=1, alpha=0.005, epsilon=0.1, epsanneal=10000, nEpisodes=5000)
agent.train(env)
if run is not None:
    agent.save(f"CartPoleMonteCarlo-{run:02d}.npy")
