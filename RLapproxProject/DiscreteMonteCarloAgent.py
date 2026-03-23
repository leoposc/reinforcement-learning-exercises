# Base class for RL agents for episodic tasks
# - derived from DiscreteAgent

import torch
import DiscreteAgent as Discrete
import numpy as np


class Agent(Discrete.Agent):
    def __init__(self, Q, nActions, gamma=1, **kwargs):
        super(Agent, self).__init__(Q, nActions, **kwargs)
        self.gamma = gamma

    # This method trains the agent for one episode on the given
    # gymnasium environment, and is called by the base class' train() method.
    # This method should repeatedly call chooseAction() and update().
    # This method must return
    #   T, the length of this episode in time steps, and
    #   G, the (discounted) return earned during this episode.
    def trainEpisode(self, env):
        state, _ = env.reset()
        terminated = truncated = False
        T = 0
        G = 0
        episode = []

        while not (terminated or truncated):
            action, qa = self.chooseAction(state, env.action_space)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            T += 1

        G = 0
        for t in range(T-1, -1, -1):
            state, action, reward = episode[t]
            G = self.gamma * G + reward
            qa = self.q[action](torch.tensor(state))
            self.update(action, G, qa)

        return T, G
