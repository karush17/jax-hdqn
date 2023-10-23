"""Implements the replay buffer Q learning."""

from collections import deque, namedtuple

import random
import numpy as np


Batch = namedtuple(
    'Batch',
    ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer(object):
    """Implements the replay buffer object.
    
    Attributes:
        capacity: buffer capacity.
        buffer: replay buffer collection.
    """
    def __init__(self, capacity: int) -> None:
        """Initializes the buffer object."""
        self.capacity = capacity
        self.buffer   = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: np.ndarray,
             reward: np.ndarray, next_state: np.ndarray, done: bool) -> None:
        """Pushes samples to the buffer."""
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Batch:
        """Samples a batch from the buffer."""
        state, action, reward, next_state, done = zip(*random.sample(self.buffer,
                                                                     batch_size))
        return Batch(
            state=np.concatenate(state),
            action=np.array(action),
            reward=np.array(reward),
            next_state=np.concatenate(next_state),
            done=np.array(done)
        )

    def __len__(self) -> int:
        """Returns the size of the buffer."""
        return len(self.buffer)
