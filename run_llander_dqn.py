from __future__ import annotations

import gym
import numpy as np
import tensorflow as tf
from absl import app, flags, logging

from lunarlander.dqn.agent import Agent


def main() -> None:
    logging.set_verbosity(logging.DEBUG)
    ENV = gym.make("LunarLander-v2")
    agent = Agent(ENV)
    agent.train(1000)


if __name__ == "__main__":
    main()
