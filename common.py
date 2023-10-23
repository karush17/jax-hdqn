"""Implements the common modl train state."""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import os
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
PRNGKey = Any
Shape = Sequence[int]
Dtype = Any
InfoDict = Dict[str, float]


def default_init(scale: Optional[float] = jnp.sqrt(2)) -> jnp.ndarray:
    """Implements the neural network initialization."""
    return nn.initializers.orthogonal(scale)

@flax.struct.dataclass
class Model:
    """Implements the model for training.
    
    Attributes:
        step: train step.
        apply_fn: function for executing forward pass.
        params: trainable parameters of the network.
        tx: gradient trasnformation.
        opt_state: optimizer state during training.
    """
    step: int
    apply_fn: Callable[..., Any] = flax.struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(cls: Any,
               model_def: nn.Module,
               inputs: Sequence[jnp.ndarray],
               tx: Optional[optax.GradientTransformation] = None) -> 'Model':
        """Creates the base model object."""
        variables = model_def.init(*inputs)

        _, params = variables.pop('params')

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(step=1,
                   apply_fn=model_def.apply,
                   params=params,
                   tx=tx,
                   opt_state=opt_state)

    def __call__(self, *args, **kwargs):
        """Executes the forward pass of the model."""
        return self.apply_fn({'params': self.params}, *args, **kwargs)

    def apply_gradient(
            self,
            loss_fn: Callable[[Params], Any],
            has_aux: bool = True) -> Union[Tuple['Model', Any], 'Model']:
        """Applies the gradient transformation to parameters."""
        grad_fn = jax.grad(loss_fn, has_aux=has_aux)
        if has_aux:
            grads, aux = grad_fn(self.params)
        else:
            grads = grad_fn(self.params)

        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        new_model = self.replace(step=self.step + 1,
                                 params=new_params,
                                 opt_state=new_opt_state)
        if has_aux:
            return new_model, aux
        else:
            return new_model

    def save(self, save_path: str):
        """Saves the model parameters."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> 'Model':
        """Loads the model parameters."""
        with open(load_path, 'rb') as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)
