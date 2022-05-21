import os
import random
from typing import Any, Sequence, Callable

import jax
import jax.numpy as jnp
import flax.linen as nn

PRNGKey = Any

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    out_dim: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size)(x)
            if i+1 < len(self.hidden_dims):
                x = self.activation(x)
        x = nn.Dense(self.out_dim)(x)
        return x
    
class Actor(nn.Module):
    hidden_dims: Sequence[int]
    out_dim: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x, train=True):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size)(x)
            if i+1 < len(self.hidden_dims):
                x = self.activation(x)
        x = nn.Dense(self.out_dim)(x)
        if not train:
            x = jnp.argmax(x)
        return x
