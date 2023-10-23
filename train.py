"""Implements the main training protocols."""

import numpy as np
import math
import random
import matplotlib.pyplot as plt

from matplotlib import rc
from absl import app, flags
from hdqn import hDQN, update,meta_update, get_action, get_goal
from mdp import StochasticMDP
from buffer import ReplayBuffer

rc('font',**{'family':'serif','sans-serif':['Helvetica']}, weight='bold',
   size=20)
rc('text', usetex=True)

FLAGS = flags.FLAGS
flags.DEFINE_float('epsilon_start', 1.0, 'exploration start value')
flags.DEFINE_float('epsilon_final', 0.01, 'exploration end value')
flags.DEFINE_integer('epsilon_decay', 500, 'exploration decay')
flags.DEFINE_integer('steps', 100000, 'training steps')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_integer('batch_size', 32, 'training batch size')


def to_onehot(x: np.ndarray) -> np.ndarray:
    """Converts samples to one hot representation."""
    oh = np.zeros(6)
    oh[x-1] = 1
    return oh

def main(_) -> None:
    """Implements the main training function."""
    env = StochasticMDP()
    num_goals = env.num_states
    num_actions = env.num_actions

    replay_buffer = ReplayBuffer(10000)
    meta_replay_buffer = ReplayBuffer(10000)

    state = env.reset()
    done = False
    all_rewards = []
    episode_reward = 0
    frame_idx = 1
    epsilon_by_frame = lambda frame_idx: FLAGS.epsilon_final + (FLAGS.epsilon_start\
                     - FLAGS.epsilon_final) * math.exp(-1. *
                                                       frame_idx / FLAGS.epsilon_decay)

    agent = hDQN(num_goals, num_actions, FLAGS.learning_rate, [256])

    while frame_idx < FLAGS.steps:
        eps = epsilon_by_frame(frame_idx)
        goal = get_goal(agent.meta_model, state)
        if random.random() <= eps:
            goal = random.randrange(num_goals)

        onehot_goal = to_onehot(goal)

        meta_state = state
        extrinsic_reward = 0

        while not done and goal != np.argmax(state):
            eps = epsilon_by_frame(frame_idx)
            goal_state = np.concatenate([state, onehot_goal])
            action = get_action(agent.model, goal_state)
            if random.random() < eps:
                action = random.randrange(num_actions)
            next_state, reward, done, _ = env.step(action)

            episode_reward += reward
            extrinsic_reward += reward
            intrinsic_reward = 1.0 if goal == np.argmax(next_state) else 0.0

            replay_buffer.push(goal_state, action,
                               intrinsic_reward,
                               np.concatenate([next_state, onehot_goal]), done)
            state = next_state

            if len(replay_buffer) >= FLAGS.batch_size:
                agent.model = update(agent.model, replay_buffer,
                                     FLAGS.batch_size)
            if len(meta_replay_buffer) >= FLAGS.batch_size:
                agent.meta_model = meta_update(agent.meta_model,
                                               meta_replay_buffer,
                                               FLAGS.batch_size)
            frame_idx += 1

        meta_replay_buffer.push(meta_state, goal, extrinsic_reward, state, done)

        if done:
            state = env.reset()
            done = False
            all_rewards.append(episode_reward)
            episode_reward = 0

    n = 100
    ret_mean = [np.mean(all_rewards[i:i + n]) for i in range(0,
                                                             len(all_rewards),
                                                             n)]
    plt.figure(figsize=(10,5))
    plt.title('StochasticMDP', fontsize=20)
    plt.plot(ret_mean, color='darkorange', linewidth=4)
    plt.ylabel('Average Return')
    plt.xlabel(r'Episodes ($\times 1000$)')
    plt.tight_layout()
    plt.savefig('plot.png', dpi=600, bbox_inches='tight')


if __name__=='__main__':
    app.run(main)
