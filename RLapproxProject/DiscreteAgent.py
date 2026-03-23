# Base class for RL agents
# - with a separate q(s) neural net for each (discrete) action
# - using an epsilon-greedy policy
#
# The first argument of its constructor is the class (not instance)
# derived from torch.nn.Module that implements the q(s) neural nets.
# Its constructor must not take any arguments.

import torch
from random import random
import numpy as np


class Agent():
    def __init__(self, Q, nActions, alpha=0.0001,
                 epsilon=0.1, epsanneal=float('inf'), nEpisodes=25000):
        self.epsilon = epsilon
        self.epsanneal =  epsanneal # see annealeps() below
        self.q = [Q() for _ in range(nActions)]
        # One optimizer for each action's q function:
        self.optims = [torch.optim.SGD(qa.parameters(), alpha) for qa in self.q]
        self.episodes = np.zeros((nEpisodes, 3))
        # self.alpha = alpha

    # Call this to reduce epsilon.
    # Every self.epsanneal such calls, epsilon be reduced by a factor of 1/10.
    def annealeps(self):
        self.epsilon /= 10 ** (1/self.epsanneal)

    # Implements a self.epsilon-greedy policy.
    # Feel free to change or replace this method, e.g. by a softmax policy.
    def chooseAction(self, observation, action_space):
        # returns the (integer) action and its estimated value
        observation = torch.tensor(observation, dtype=torch.float32)
        with torch.no_grad():
            if random() < self.epsilon:
                action = action_space.sample()
                return action, self.q[action](observation)
            qa = [qa(observation) for qa in self.q]
            qamax = np.argmax([q.detach().numpy() for q in qa])

        return qamax, qa[qamax]

    # Perform a gradient-ascent parameter update.
    # target is the target u of the update;
    # qa = self.q[action](torch.tensor(observation)).
    # I.e., delta = target - qa.
    # Feel free remove these two self.optims[action]....() calls and
    # place them elsewhere, e.g. for updates based on full-episode gradients.
    def update(self, action, target, qa):
        self.optims[action].zero_grad()
        target_tens = torch.tensor([target], dtype = torch.float32)
        loss = torch.nn.functional.mse_loss(qa, target_tens)
        loss.backward()
        self.optims[action].step()

    # This method trains the agent on the given gymnasium environment.
    # The method for training one episode, trainEpisode(env),
    # must be implemented by a derived class.
    def train(self, env):
        for episode in range(len(self.episodes)):
            self.episodes[episode,:] = *self.trainEpisode(env), self.epsilon
            print(f"{episode=:5d}, t={self.episodes[episode,0]:3.0f}: G={self.episodes[episode,1]:6.1f} {self.epsilon=}")
            self.annealeps()

    # Call this to save data collected during training for further analysis.
    def save(self, file):
        np.save(file, self.episodes)
