import random
import numpy as np
from collections import deque, namedtuple

Batch = namedtuple(
    'Batch',
    ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer   = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return Batch(
            state=np.concatenate(state),
            action=np.array(action),
            reward=np.array(reward),
            next_state=np.concatenate(next_state),
            done=np.array(done)
        )
    
    def __len__(self):
        return len(self.buffer)