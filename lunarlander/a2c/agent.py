from __future__ import annotations

from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tqdm
from absl import logging

from lunarlander.a2c.models import Actor, Critic

EPS = np.finfo(np.float32).eps.item()
ENV = gym.wrappers.time_limit.TimeLimit


class Agent:
    """soon"""

    def __init__(self, env: ENV, alpha: float = 0.0002, gamma: float = 0.99) -> None:
        """soon"""
        self.env: ENV = env
        self.actor: Actor = Actor(alpha=alpha, num_actions=4)
        self.critic: Critic = Critic(alpha=alpha, num_out=1)
        self.max_steps: int = 1000
        self.gamma: float = gamma
        self.reward_history: list[float] = []
        self.loss_history: list[float] = []

    def _env_step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """soon"""
        state, reward, terminal, _ = self.env.step(action)
        return (
            state.astype(np.float32),
            np.array(reward, np.int32),
            np.array(terminal, np.int32),
        )

    def _tf_env_step(self, action: tf.Tensor) -> list[tf.Tensor]:
        """soon"""
        return tf.numpy_function(
            self._env_step, [action], [tf.float32, tf.int32, tf.int32]
        )

    def run_episode(
        self, initial_state: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """soon"""
        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        state = initial_state
        state_shape = state.shape
        for t in tf.range(self.max_steps):
            state = tf.expand_dims(state, 0)
            action_logits = self.actor(state)
            value = self.critic(state)
            action = tf.random.categorical(action_logits, 1)[0, 0]
            state_policy = tf.nn.softmax(action_logits)
            state, reward, terminal = self._tf_env_step(action)
            state.set_shape(state_shape)

            action_probs = action_probs.write(t, state_policy[0, action])
            values = values.write(t, tf.squeeze(value))
            rewards = rewards.write(t, reward)

            if tf.cast(terminal, tf.bool):
                break

        return (
            action_probs.stack(),
            values.stack(),
            rewards.stack(),
        )

    def _get_expected_return(
        self, rewards: tf.Tensor, standardize: bool = True
    ) -> tf.Tensor:
        """soon"""
        n = tf.shape(rewards)[0]
        targets = tf.TensorArray(dtype=tf.float32, size=n)

        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            discounted_sum = rewards[i] + self.gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            targets = targets.write(i, discounted_sum)
        targets = targets.stack()[::-1]

        if standardize:
            targets = (targets - tf.math.reduce_mean(targets)) / (
                tf.math.reduce_std(targets) + EPS
            )
        return targets

    @tf.function
    def train_episode(
        self, initial_state: tf.Tensor
    ) -> tuple[int, tuple[float, float]]:
        """soon"""
        with tf.GradientTape(persistent=True) as tape:
            action_probs, values, rewards = self.run_episode(initial_state)
            targets = self._get_expected_return(rewards)
            action_probs, values, targets = [
                tf.expand_dims(x, 1) for x in [action_probs, values, targets]
            ]
            actor_loss = self.actor.loss(action_probs, values, targets)
            critic_loss = self.critic.loss(values, targets)
        actor_gradients, critic_gradients = tape.gradient(
            [actor_loss, critic_loss],
            [self.actor.trainable_variables, self.critic.trainable_variables],
        )
        self.actor.optimizer.apply_gradients(
            zip(actor_gradients, self.actor.trainable_variables)
        )
        self.critic.optimizer.apply_gradients(
            zip(critic_gradients, self.critic.trainable_variables)
        )
        return int(tf.math.reduce_sum(rewards)), (float(actor_loss), float(critic_loss))

    def training(
        self,
        rewards_threshold: int = 120,
        min_episodes_criterion: int = 100,
        max_episodes: int = 10000,
    ) -> None:
        running_reward = 0
        episodes_reward = deque(maxlen=min_episodes_criterion)
        with tqdm.trange(max_episodes) as t:
            for e in t:
                initial_state = tf.constant(self.env.reset(), dtype=tf.float32)
                episode_reward, episode_loss = self.train_episode(initial_state)
                episodes_reward.append(episode_reward)
                self.reward_history.append(episode_reward)
                self.loss_history.append(episode_loss)
                running_reward = np.mean(episodes_reward)
                t.set_description(f"Episode {e}")
                t.set_postfix(
                    episode_reward=int(episode_reward), running_reward=running_reward
                )
                if running_reward > rewards_threshold and e >= min_episodes_criterion:
                    break
        logging.info(
            f"Solved at episode {e} with average reward of {running_reward:.2f}"
        )

    def plot_history(self) -> None:
        """soon"""
        _, ax_reward = plt.subplots(1)
        ax_reward.plot(self.reward_history, color="blue")
        ax_reward.plot(
            [i + 10 for i in range(len(self.reward_history) - 10)],
            [
                sum(self.reward_history[i : i + 10]) / 10
                for i in range(len(self.reward_history) - 10)
            ],
            color="red",
        )
        ax_reward.set_ylabel("Reward")
        ax_reward.set_xlabel("Episode")
        plt.show()

    def plot_loss(self) -> None:
        """soon"""
        _, (ax_a_loss, ax_c_loss) = plt.subplots(1, 2)
        ax_a_loss.plot([loss[0] for loss in self.loss_history], color="blue")
        ax_a_loss.set_ylabel("Actor Loss")
        ax_a_loss.set_xlabel("Update")
        ax_c_loss.plot([loss[1] for loss in self.loss_history], color="blue")
        ax_c_loss.set_ylabel("Critic Loss")
        ax_c_loss.set_xlabel("Update")
        plt.show()
