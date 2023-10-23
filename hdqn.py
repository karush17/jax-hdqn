"""Implements the hierarchical DQN agent."""

from typing import Any, Dict, Tuple

import functools
import numpy as np
import jax
import jax.numpy as jnp
import optax

from jax.lax import stop_gradient
from model import Actor
from common import Model


@functools.partial(jax.jit)
def _update_jit(model: Any, batch: Any) -> Tuple[Any, Dict[str, Any]]:
    """Updates the agent parameters."""
    state, action, reward, next_state, done = batch.state,\
             batch.action, batch.reward, batch.next_state, batch.done
    def loss_fn(params):
        outs = model.apply_fn({'params': params}, state)
        q_vals =  outs[jnp.arange(action.shape[0]), action]
        next_outs = model.apply_fn({'params': params}, next_state)
        next_q_vals = jnp.max(next_outs, axis=1)
        exp_q_vals = reward + 0.99*next_q_vals*(1-done)
        loss = jnp.mean(jnp.square(q_vals - stop_gradient(exp_q_vals)))
        return loss, {'loss': loss}
    new_model, info = model.apply_gradient(loss_fn)
    return new_model, info

class hDQN:
    """Implements the hierarchical DQN agent.
    
    Attributes:
        num_goals: number of goals to sample.
        num_actions: number of actions for the agent.
        lr: learning rate.
        layers: number of layers.
        model: hdqn q learning model.
        meta_model: parent model for hierarchical learning.
        rng: jax random key.
    """
    def __init__(self, num_goals, num_actions, lr, layers):
        self.num_goals = num_goals
        self.num_actions = num_actions
        self.lr = lr
        self.layers = layers

        rng = jax.random.PRNGKey(42)
        rng, model_key, meta_key = jax.random.split(rng, 3)
        model_def = Actor(self.layers, self.num_actions)
        model = Model.create(model_def, inputs=[model_key, jnp.zeros((2*self.num_goals,))],\
                 tx=optax.adam(learning_rate=self.lr))
        meta_model_def = Actor(self.layers, self.num_goals)
        meta_model = Model.create(meta_model_def, inputs=[meta_key, jnp.zeros((self.num_goals,))],\
                 tx=optax.adam(learning_rate=self.lr))

        self.model = model
        self.meta_model = meta_model
        self.rng = rng


@functools.partial(jax.jit)
def get_action(model: Any, state: np.ndarray) -> jnp.ndarray:
    """Samples the action from the agent."""
    action = model.apply_fn({'params': model.params},\
                        state, train=False)
    return action

@functools.partial(jax.jit)
def get_goal(meta_model: Any, state: np.ndarray) -> jnp.ndarray:
    """Samples the goal from the meta model."""
    goal = meta_model.apply_fn({'params': meta_model.params},\
                        state, train=False)
    return goal

def update(model: Any, buffer: Any, batch_size: int) -> Any:
    """Updates the model using gradient transformation."""
    batch = buffer.sample(batch_size)
    new_model, _ = _update_jit(model, batch)
    return new_model

def meta_update(meta_model: Any, buffer: Any, batch_size: int) -> Any:
    """Updates the meta model using graident transformation."""
    batch = buffer.sample(batch_size)
    new_model, _ = _update_jit(meta_model, batch)
    return new_model
