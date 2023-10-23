"""Implements the Neural Network models."""

from typing import Any, Sequence, Callable

import jax.numpy as jnp
import flax.linen as nn

PRNGKey = Any

class MLP(nn.Module):
    """Implements the MLP model.
    
    Attributes:
        hidden_dims: number of hidden units.
        out_dim: number of output dimensions.
        activation: activation function.
    """
    hidden_dims: Sequence[int]
    out_dim: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Executes the forward pass of the model."""
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size)(x)
            if i+1 < len(self.hidden_dims):
                x = self.activation(x)
        x = nn.Dense(self.out_dim)(x)
        return x

class Actor(nn.Module):
    """Implements the actor network.
    
    Attributes:
        hidden_dims: number of hidden dimensions.
        out_dims: number of output dimensions.
        activation: activation function.
    """
    hidden_dims: Sequence[int]
    out_dim: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray, train=True) -> jnp.ndarray:
        """Executes the forward pass of the model."""
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size)(x)
            if i+1 < len(self.hidden_dims):
                x = self.activation(x)
        x = nn.Dense(self.out_dim)(x)
        if not train:
            x = jnp.argmax(x)
        return x
