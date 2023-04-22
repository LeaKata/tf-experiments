from __future__ import annotations

import numpy as np
import tensorflow as tf
from absl import logging

# Dense Tower
tower_input = tf.keras.layers.Input(shape=(8))
tower_dense_1 = tf.keras.layers.Dense(64, activation="relu")(tower_input)
tower_dense_2 = tf.keras.layers.Dense(64, activation="relu")(tower_dense_1)
tower_dense_3 = tf.keras.layers.Dense(64, activation="relu")(tower_dense_2)

# State-Value Head
qv_prediction = tf.keras.layers.Dense(4, activation="linear")(tower_dense_3)

# Policy Head
p_prediction = tf.keras.layers.Dense(4, activation="softmax")(tower_dense_3)


class QValueNN:
    """soon"""

    def __init__(self, alpha: float) -> None:
        """soon"""
        self.model = tf.keras.models.Model(
            inputs=[tower_input], outputs=[qv_prediction]
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=alpha),
            loss=tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM),
        )
        logging.info(f"{self.__class__.__name__} initialized")

    def predict(self, state: np.ndarray, training: bool = False) -> np.ndarray:
        """soon"""
        return self.model(state, training=training).numpy()

    def train_batch(self, states: np.ndarray, targets: np.ndarray) -> dict[str, float]:
        """soon"""
        return self.model.train_on_batch(states, targets, return_dict=True)


class PolicyNN:
    """soon"""

    def __init__(self, alpha: float) -> None:
        """soon"""
        self.model = tf.keras.models.Model(inputs=[tower_input], outputs=[p_prediction])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)
        logging.info(f"{self.__class__.__name__} initialized")

    def predict(self, state: np.ndarray, training: bool = False) -> None:
        """soon"""
        return self.model(state, training=training).numpy()

    def _policy_gradient(
        self,
        beta: tf.Tensor,
        states_b: tf.Tensor,
        action_values: tf.Tensor,
        policy_b: tf.Tensor,
        training: bool = True,
    ) -> list[tf.Tensor]:
        """soon"""
        with tf.GradientTape() as tape:
            policy_prediction = self.model(states_b, training=training)
            KL_distance = tf.keras.metrics.kl_divergence(policy_prediction, policy_b)
            expected_action_value = tf.math.reduce_sum(
                action_values * policy_prediction, axis=1
            )
            target = expected_action_value - beta * KL_distance
        return tape.gradient(target, self.model.trainable_variables)

    def train_batch(
        self,
        beta: float,
        states_b: np.ndarray,
        action_values: np.ndarray,
        policy_b: np.ndarray,
    ) -> None:
        """soon"""
        beta = tf.constant(beta, dtype=tf.float32)
        states_b = tf.constant(states_b, dtype=tf.float32)
        action_values = tf.constant(action_values, dtype=tf.float32)
        policy_b = tf.constant(policy_b, dtype=tf.float32)
        gradient = self._policy_gradient(beta, states_b, action_values, policy_b)
        self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
