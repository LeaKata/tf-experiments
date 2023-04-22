from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from absl import logging

from lunarlander.dqn.models import QValueNN

RNG = np.random.default_rng()
EPS = np.finfo(np.float32).eps.item()
ENV = gym.wrappers.time_limit.TimeLimit


@dataclass
class Timestep:
    """Dataclass that holds the data of one q-learning timestep.
    Attributes:
        state: The starting state of the timestep
        action: The action performed in the timestep
        reward: The reward obtained in the timestep
        q_values: The predicted q_values of the timestep
        next_state: The state resulting from 'action' in 'state'
        terminal: Timestep ends in terminal state
    """

    state: np.ndarray
    action: int
    reward: int
    q_values: np.ndarray
    next_state: np.ndarray
    terminal: bool


class ReplayMemory:
    """Simple experience replay memory.
    Experience replay memory for q-learning agents that holds data from timestep
    experience and generates batches of predefined size by randomly sampling
    from the buffer.
    Attributes:
        _capacity: Maximum number of timesteps to be stored in the memory
        _capacity_reached: Flag that indicates if the memory is filled and the
            and becomes a rolling window of the last _capacity timesteps
        _batchsize: Number of timesteps returned in a sampe batch
        _buffer: The list that stores the timestep data.
    """

    def __init__(self, capacity: int, batchsize: int) -> None:
        """Initialises the empty experience replay memory with args"""
        self._capacity: int = capacity
        self._capacity_reached: bool = False
        self._batchsize: int = batchsize
        self.batch_possible: bool = False
        self._buffer: list[Timestep] = [None] * capacity
        self._data_index: int = 0

    def add_timestep(self, timestep: Timestep) -> None:
        """
        Adds a single timestep to the memory and sets the batch_possible as well
        as the _capacitzy_reached flags if conditions are met.
        Keeps rolling window of the last _capacity number of timeslots once
        _capacity is reached.
        """
        self._buffer[self._data_index] = timestep
        self._data_index += 1
        if not self.batch_possible:
            if self._data_index >= self._batchsize:
                self.batch_possible = True
        if self._data_index >= self._capacity:
            self._capacity_reached = True
            self._data_index = 0

    def sample_batch(self) -> tuple[int, list[Timestep]]:
        """Returns a random uniform sampled batch of batchsize"""
        if not self._capacity_reached:
            batch = RNG.choice(self._buffer[: self._data_index], self._batchsize)
        else:
            batch = RNG.choice(self._buffer, self._batchsize)
        return self._batchsize, batch

    def reset(self) -> None:
        """Resets the replay memory"""
        self._data_index = 0
        self._capacity_reached = False
        self.batch_possible = False


class Agent:
    """soon"""

    def __init__(self, env: ENV, alpha: float = 0.02, gamma: float = 0.95) -> None:
        """soon"""
        self._memory: ReplayMemory = ReplayMemory(250, 30)
        self._qnn: QValueNN = QValueNN(alpha)
        self._env: ENV = env
        self.max_steps: int = 1000
        self.actions: list[int] = list(range(4))
        self.gamma: float = 0.95
        self.epsilon: float = 1
        self.epsilon_max: float = 0.8
        self.epsilon_min: float = 0.01
        self.eps_decay_rate: float = 0.99
        self.history: list[float] = []
        self.rolling_history: deque[float] = deque(maxlen=100)
        self.loss_history: list[float] = []

    def clear_history(self) -> None:
        """soon"""
        self.history.clear()
        self.rolling_history.clear()
        self.loss_history.clear()

    def epsilon_decay(self) -> None:
        """soon"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.eps_decay_rate
            logging.info(f"New epsilon value: {self.epsilon}")

    def _update(self) -> dict[str, float]:
        """soon"""
        batchsize, batch = self._memory.sample_batch()
        states = np.zeros((batchsize, 8))
        targets = np.zeros((batchsize, 4))
        for s, step in enumerate(batch):
            states[s] = step.state
            targets[s] = step.q_values
            target = step.reward
            if not step.terminal:
                next_value = self._qnn.predict(step.next_state)[0]
                targets += self.gamma * next_value[step.action]
            targets[s, step.action] = target
        return self._qnn.train_batch(states, targets)

    def _step(self, state: np.ndarray) -> np.ndarray:
        """soon"""
        q_values = self._qnn.predict(state)
        if np.random.rand() < self.epsilon:
            action = RNG.choice(self.actions)
        else:
            action = np.argmax(q_values)
        n_state, reward, terminal, _ = self._env.step(action)
        next_state = n_state[np.newaxis, :]
        self._memory.add_timestep(
            Timestep(state, action, reward, q_values, next_state, terminal),
        )
        return next_state, reward, terminal

    def _episode(self, training: bool = False, render: bool = False) -> float:
        """soon"""
        state = self._env.reset()[np.newaxis, :]
        stepcount = 0
        reward_sum = 0
        while stepcount < self.max_steps:
            stepcount += 1
            state, reward, terminal = self._step(state)
            reward_sum += reward
            if training:
                self._update()
            if render:
                self._env.render()
            if terminal:
                break
        return reward_sum, stepcount

    def train(self, episodes: int, render: bool = False) -> None:
        """soon"""
        self.clear_history()
        for e in range(episodes):
            e_reward, stepcount = self._episode(training=True, render=render)
            if self._memory._capacity_reached:
                self.loss_history.append(self._update())
                self.epsilon_decay()
            self.history.append(e_reward)
            self.rolling_history.append(e_reward)
            rolling_average = np.mean(self.rolling_history)
            logging.info(
                f"Episode {e} of {episodes} terminated with after {stepcount} "
                f"steps with a reward of: {np.round(self.history[-1], 2)} -- "
                f"rolling average reward: {np.round(rolling_average, 2)}"
            )
        if render:
            self._env.close()

    def evaluation(self, episodes: int, render: bool = False) -> None:
        """soon"""
        self.clear_history()
        for e in range(episodes):
            e_reward = self._episode(training=False, render=render)
            self.history.append(e_reward)
            self.rolling_history.append(e_reward)
            rolling_average = np.mean(self.rolling_history)
            logging.info(
                f"Episode {e} of {episodes} terminated with a reward of: "
                f"{np.round(self.history[-1], 2)} -- "
                f"rolling average reward: {np.round(rolling_average, 2)}"
            )
        if render:
            self._env.close()

    def plot_history(self, window_size: int = 100) -> None:
        """soon"""
        _, ax_reward = plt.subplots(1)
        ax_reward.plot(self.complete_reward_history, color="blue")
        ax_reward.plot(
            [
                i + window_size
                for i in range(len(self.complete_reward_history) - window_size)
            ],
            [
                sum(self.complete_reward_history[i : i + window_size]) / window_size
                for i in range(len(self.complete_reward_history) - window_size)
            ],
            color="red",
        )
        ax_reward.set_ylabel("Reward")
        ax_reward.set_xlabel("Episode")
        plt.show()

    def plot_loss(self, window_size: int = 100) -> None:
        """soon"""
        _, ax_reward = plt.subplots(1)
        ax_reward.plot(self.loss_history, color="blue")
        ax_reward.set_ylabel("Loss")
        ax_reward.set_xlabel("Update")
        plt.show()
