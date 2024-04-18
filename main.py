import numpy as np

from eight_puzzle import EightPuzzleEnv
from q_learning import QLearning

# create Environment
env = EightPuzzleEnv(limit_count_steps=10_000, render=False, reward_type="small-penalty")

# create agent
agent = QLearning(env, gamma=0.9, alpha=0.1, epsilon=0.7, decay_rate=0.0007, use_random_values=False)
agent.train(15_000)
agent.plot_rewards(plot_after=4_000)
