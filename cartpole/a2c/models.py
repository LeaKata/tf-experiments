from __future__ import annotations

import tensorflow as tf
from absl import logging


class Critic(tf.keras.Model):
    """soon"""

    def __init__(self, alpha: float, num_hidden_units: int) -> None:
        """soon"""
        super().__init__()
        self.common_layer: tf.keras.layers.Dense = tf.keras.layers.Dense(
            num_hidden_units, activation="relu"
        )
        self.critic: tf.keras.layers.Dense = tf.keras.layers.Dense(1)
        self.huber_loss: tf.keras.losses.Huber = tf.keras.losses.Huber(
            reduction=tf.keras.losses.Reduction.SUM
        )
        self.optimizer: tf.keras.optimizers.Adam = tf.keras.optimizers.Adam(
            learning_rate=alpha
        )
        logging.info("Critic model initialized")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """soon"""
        common_output = self.common_layer(inputs)
        return self.critic(common_output)

    def loss(self, values: tf.Tensor, targets: tf.Tensor) -> tf.Tensor:
        """soon"""
        return self.huber_loss(values, targets)


class Actor(tf.keras.Model):
    """soon"""

    def __init__(self, alpha: float, num_actions: int, num_hidden_units: int) -> None:
        """soon"""
        super().__init__()
        self.common_layer: tf.keras.layers.Dense = tf.keras.layers.Dense(
            num_hidden_units, activation="relu"
        )
        self.actor: tf.keras.layers.Dense = tf.keras.layers.Dense(num_actions)
        self.optimizer: tf.keras.optimizers.Adam = tf.keras.optimizers.Adam(
            learning_rate=alpha
        )
        logging.info(f"Actor model initialized for {num_actions} actions")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """soon"""
        common_output = self.common_layer(inputs)
        return self.actor(common_output)

    def loss(
        self, action_probs: tf.Tensor, values: tf.Tensor, targets: tf.Tensor
    ) -> tf.Tensor:
        """soon"""
        advantage = targets - values
        action_log_porbs = tf.math.log(action_probs)
        return -tf.math.reduce_sum(action_log_porbs * advantage)
