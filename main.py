import numpy as np

from eight_puzzle_env import EightPuzzleEnv
from q_learning import QLearning

# create Environment
env = EightPuzzleEnv()

# create agent
agent = QLearning(env, gamma=0.9, alpha=0.1, epsilon=0.7, decay_rate=0.0001, use_random_values=False)
agent.train(100)
