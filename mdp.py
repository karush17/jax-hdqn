"""Implements the stochastic MDP environment."""

from typing import Tuple, Dict

import random
import numpy as np


class StochasticMDP:
    """Implements the stochastic MDP environment.
    
    Attributes:
        end: final state of the agent.
        current_state: current state of the agent.
        num_actions: number of allowed actions.
        num_state: number of allowed state.
        p_right: probability of moving right.
    """
    def __init__(self):
        """Initializes the MDP."""
        self.end           = False
        self.current_state = 2
        self.num_actions   = 2
        self.num_states    = 6
        self.p_right       = 0.5

    def reset(self):
        """Resets the environment."""
        self.end = False
        self.current_state = 2
        state = np.zeros(self.num_states)
        state[self.current_state - 1] = 1.
        return state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray,
                                                float, bool, Dict[str, int]]:
        """Steps the agent to the next state."""
        if self.current_state != 1:
            if action == 1:
                if random.random() < self.p_right and self.current_state < self.num_states:
                    self.current_state += 1
                else:
                    self.current_state -= 1                    
            if action == 0:
                self.current_state -= 1                
            if self.current_state == self.num_states:
                self.end = True

        state = np.zeros(self.num_states)
        state[self.current_state - 1] = 1.

        if self.current_state == 1:
            if self.end:
                return state, 1.00, True, {}
            else:
                return state, 1.00/100.00, True, {}
        else:
            return state, 0.0, False, {}
