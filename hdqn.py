import jax
import jax.numpy as jnp
import numpy as np
import optax
import functools

from jax.lax import stop_gradient
from typing import Optional, Sequence, Tuple
from model import Actor
from common import InfoDict, Model, PRNGKey

@functools.partial(jax.jit)
def _update_jit(model, batch):
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

class hDQN(object):
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
def get_action(model, state):
    action = model.apply_fn({'params': model.params},\
                        state, train=False)
    return action

@functools.partial(jax.jit)
def get_goal(meta_model, state):
    goal = meta_model.apply_fn({'params': meta_model.params},\
                        state, train=False)
    return goal

def update(model, buffer, batch_size):
    batch = buffer.sample(batch_size)
    new_model, info = _update_jit(model, batch)
    return new_model

def meta_update(meta_model, buffer, batch_size):
    batch = buffer.sample(batch_size)
    new_model, info = _update_jit(meta_model, batch)
    return new_model


