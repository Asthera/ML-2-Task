from eight_puzzle import EightPuzzleEnv
from q_learning import QLearning
import time

# Define hyperparameters
# [experiment, alpha, gamma, epsilon, decay rate, episodes, use_random_values, reward_type]
hyperparameters = [
    [1, 0.1, 0.8, 0.8, 0.0008, 5000, False, "small-penalty"],
    [2, 0.1, 0.9, 0.6, 0.0006, 5000, True, "manhattan"],
    [3, 0.1, 0.99, 0.5, 0.0005, 5000, False, "hamming"],
    [4, 0.2, 0.8, 0.5, 0.0003, 5000, True, "small-penalty"],
    [5, 0.2, 0.9, 0.65, 0.0005, 5000, False, "manhattan"],
    [6, 0.2, 0.99, 0.4, 0.0001, 5000, True, "hamming"],
    [7, 0.3, 0.8, 0.5, 0.0001, 5000, False, "small-penalty"],
    [8, 0.3, 0.9, 0.45, 0.0001, 5000, True, "manhattan"],
    [9, 0.3, 0.99, 0.4, 0.0001, 5000, False, "hamming"],
    [10, 0.4, 0.8, 0.5, 0.0001, 10000, True, "small-penalty"],
    [11, 0.4, 0.9, 0.55, 0.0001, 10000, False, "manhattan"],
    [12, 0.4, 0.99, 0.8, 0.0009, 10000, True, "hamming"],
    [13, 0.5, 0.8, 0.6, 0.0005, 10000, False, "small-penalty"],
    [14, 0.5, 0.9, 0.45, 0.0005, 10000, True, "manhattan"],
    [15, 0.5, 0.99, 0.4, 0.0005, 10000, False, "hamming"]
]

for exp in hyperparameters:
    print(f"Experiment {exp[0]}")
    print("Hyperparameters:")
    print(f"Alpha: {exp[1]}")
    print(f"Gamma: {exp[2]}")
    print(f"Epsilon: {exp[3]}")
    print(f"Decay rate: {exp[4]}")
    print(f"Episodes: {exp[5]}")
    print(f"Use random values: {exp[6]}")
    print(f"Reward type: {exp[7]}")

    env = EightPuzzleEnv(limit_count_steps=10_000, render=False, reward_type=exp[7])
    agent = QLearning(env, gamma=exp[2], alpha=exp[1], epsilon=exp[3], decay_rate=exp[4], use_random_values=exp[6])

    exp_start = time.time()

    agent.train(exp[5])

    agent.save_Q_table(f"weights/q_learning/q_table_exp_{exp[0]}.npy")

    # Calculate metrics

    average_reward = agent.get_average_reward()
    max_reward = agent.get_max_reward()
    average_steps = agent.get_average_steps()
    success_rate = agent.get_success_rate()

    print(f"Average reward: {average_reward:.2f}")
    print(f"Max reward: {max_reward:.2f}")
    print(f"Average steps: {average_steps:.2f}")
    print(f"Success rate: {success_rate:.2f}")
    print(f"Experiment {exp[0]} took {time.time() - exp_start} seconds")
    print()
