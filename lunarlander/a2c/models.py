from __future__ import annotations

import tensorflow as tf
from absl import logging


class Critic(tf.keras.Model):
    """soon"""

    def __init__(self, alpha: float, num_out: int) -> None:
        """soon"""
        super().__init__()
        self.layer_1: tf.keras.layers.Dense = tf.keras.layers.Dense(
            512, activation="relu"
        )
        self.layer_2: tf.keras.layers.Dense = tf.keras.layers.Dense(
            256, activation="relu"
        )
        self.pre_critic: tf.keras.layers.Dense = tf.keras.layers.Dense(
            256, activation="relu"
        )
        self.critic: tf.keras.layers.Dense = tf.keras.layers.Dense(
            num_out, activation="linear"
        )
        self.huber_loss: tf.keras.losses.Huber = tf.keras.losses.Huber(
            reduction=tf.keras.losses.Reduction.SUM
        )
        self.optimizer: tf.keras.optimizers.Adam = tf.keras.optimizers.Adam(
            learning_rate=alpha
        )
        logging.info(f"{self.__class__.__name__} initialized")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """soon"""
        out_layer_1 = self.layer_1(inputs)
        out_layer_2 = self.layer_2(out_layer_1)
        pre_critic = self.pre_critic(out_layer_2)
        return self.critic(pre_critic)

    def loss(self, values: tf.Tensor, targets: tf.Tensor) -> tf.Tensor:
        """soon"""
        return self.huber_loss(values, targets)


class Actor(tf.keras.Model):
    """soon"""

    def __init__(self, alpha: float, num_actions: int) -> None:
        """soon"""
        super().__init__()
        self.layer_1: tf.keras.layers.Dense = tf.keras.layers.Dense(
            512, activation="relu"
        )
        self.layer_2: tf.keras.layers.Dense = tf.keras.layers.Dense(
            256, activation="relu"
        )
        self.pre_actor: tf.keras.layers.Dense = tf.keras.layers.Dense(
            256, activation="relu"
        )
        self.actor: tf.keras.layers.Dense = tf.keras.layers.Dense(num_actions)
        self.optimizer: tf.keras.optimizers.Adam = tf.keras.optimizers.Adam(
            learning_rate=alpha
        )
        logging.info(f"{self.__class__.__name__} initialized")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """soon"""
        out_layer_1 = self.layer_1(inputs)
        out_layer_2 = self.layer_2(out_layer_1)
        pre_actor = self.pre_actor(out_layer_2)
        return self.actor(pre_actor)

    def loss(
        self, action_probs: tf.Tensor, values: tf.Tensor, targets: tf.Tensor
    ) -> tf.Tensor:
        """soon"""
        advantage = targets - values
        action_log_porbs = tf.math.log(action_probs)
        return tf.math.reduce_sum(action_log_porbs * advantage)
