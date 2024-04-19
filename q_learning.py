import numpy as np
import itertools
import time
import matplotlib.pyplot as plt


class QLearning:
    def __init__(self,
                 env,
                 gamma: float,
                 alpha: float,
                 epsilon: float,
                 decay_rate: float,
                 use_random_values: bool
                 ):

        self.env = env

        # define hyperparameters
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.use_random_values = use_random_values

        # 0: up, 1: left, 2: down, 3: right
        self.action_space = [0, 1, 2, 3]
        self.len_actions = len(self.action_space)

        self.Q = {}
        self.create_Q_table(value="random" if self.use_random_values else "zero")

        # steps, rewards to then plot it in the end
        self.steps = []
        self.rewards = []

        self.visited_states = set()

    def max_Q(self, state: tuple) -> float:
        """
        Calculate the maximum Q value for a given state

        :param state: the state
        :return: the maximum Q value
        """

        values = np.array([self.Q[state, action] for action in self.action_space])
        return np.max(values)

    def argmax_Q(self, state: tuple) -> int:
        """
        Find index of the action space == (action value) with the maximum Q value
        :param state:
        :return action:
        """
        values = np.array([self.Q[state, action] for action in self.action_space])
        return np.argmax(values)

    def train_episode(self) -> (int, float):
        """
        Train the agent for one episode
        :return:
        """
        state, info = self.env.reset()

        done = False
        truncated = False

        steps = 0
        reward_episode = 0

        start = time.time()

        while not done and not truncated:
            # choose action
            action = self.choose_action(state)
            # take action
            state_, reward, done, truncated, info = self.env.step(action)

            # update Q(s, a)
            # Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(a)(Q(s_, a)) - Q(s, a))
            self.Q[state, action] += self.alpha * (
                    reward + self.gamma * self.max_Q(state_) - self.Q[state, action])

            state = state_

            steps += 1
            reward_episode += reward
            # print(f"Reward: {reward}")

            # if steps % 100 == 0:
            #     print(f"Step {steps}")

            self.visited_states.add(state)

        self.rewards.append(reward_episode)

        print(f"\n Visited states: {len(self.visited_states)} \n")
        print(f"\n\nTime taken: {time.time() - start} seconds\n\n")

        return steps, reward_episode

    def train(self, episodes: int) -> None:
        """
        Train the agent for a number of episodes
        :param episodes:
        :return:
        """
        for e in range(episodes):
            steps, rewards = self.train_episode()
            self.decrease_epsilon()

            print("------------------------------------")
            print(f"Episode {e + 1}/{episodes}, steps: {steps}, rewards: {rewards}, epsilon: {self.epsilon}")

    def decrease_epsilon(self) -> None:
        """
        Decrease epsilon
        :return:
        """
        self.epsilon -= self.decay_rate

        if self.epsilon < 0.01:
            self.epsilon = 0.01

    def choose_action(self, state: tuple) -> int:
        """
        Choose an action based on epsilon-greedy policy
        :param state:
        :return:
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(np.arange(self.len_actions))
        else:
            return self.argmax_Q(state)

    def create_Q_table(self, value: str = "random" or "zero"):
        """
        Create the Q table with random or zero values
        :param value:
        :return:
        """

        all_possible_states = self.possible_states()

        if value == "random":
            for state in all_possible_states:
                for action in self.action_space:
                    self.Q[state, action] = np.random.rand()
            # TODO: add 0 to terminal states
        if value == "zero":
            for state in all_possible_states:
                for action in self.action_space:
                    self.Q[state, action] = 0

    def possible_states(self) -> list[tuple, ]:
        """
        Generate all possible states for 8-puzzle

        :return:
        list of tuples with all possible states
        """
        permutations = itertools.permutations(range(9))

        return list(permutations)

    def plot_rewards(self, plot_after: int = 0):
        """
        Plot the rewards

        :param
        plot_after: by default 0, plot the rewards from the beginning
        :return:
        """

        plt.plot(self.rewards[plot_after:], 'o')
        plt.show()