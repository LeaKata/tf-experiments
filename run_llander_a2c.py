from __future__ import annotations

import gym
import tensorflow as tf
from absl import logging

from lunarlander.a2c.agent import Agent


def main() -> None:
    logging.set_verbosity(logging.DEBUG)
    env = gym.make("LunarLander-v2")
    agent = Agent(env, alpha=0.0002, gamma=0.99)
    agent.training()
    agent.plot_history()
    agent.plot_loss()


if __name__ == "__main__":
    tf.config.set_visible_devices([], "GPU")
    #tf.config.run_functions_eagerly(True)  # for debugging
    main()
