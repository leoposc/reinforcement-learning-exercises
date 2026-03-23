# Base class for RL agents for episodic tasks
# - derived from DiscreteAgent

import torch
import DiscreteAgent as Discrete
import numpy as np 

class Agent(Discrete.Agent):
    def __init__(self, Q, nActions, gamma=1, **kwargs):
        super(Agent, self).__init__(Q, nActions, **kwargs)
        self.gamma = gamma
        self.alpha = kwargs.get('alpha', 0.0001)

    # This method trains the agent for one episode on the given
    # gymnasium environment, and is called by the base class' train() method.
    # This method should repeatedly call chooseAction() and update().
    # This method must return
    #   T, the length of this episode in time steps, and
    #   G, the (discounted) return earned during this episode.
    def trainEpisode(self, env):
        state, _ = env.reset()  # Reset environment at the start
        action, qa = self.chooseAction(state, env.action_space)

        truncated = done = False
        G = 0  # Initialize return
        T = 0  # Initialize time steps
             
        action, _ = self.chooseAction(state, env.action_space)
        while not( done or truncated):
            next_state, reward, done, truncated, _ = env.step(action)
            G += reward * (self.gamma ** T)  # Accumulate discounted reward
            T += 1

            # Choose the next action
            next_action, next_qa = self.chooseAction(next_state, env.action_space)
            
            # Compute SARSA target: u = r + γ * Q(s', a')
            target = reward + (self.gamma * next_qa.item() * (not done))

            # Update the Q-value
            qa = self.q[action](torch.tensor(state))
            self.update(action, target, qa)
            
            # Move to the next state
            state = next_state
            action = next_action
            qa = next_qa
        return T, G

