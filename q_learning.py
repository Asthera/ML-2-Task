import numpy as np
import itertools
import time

def all_possible_states():
    permutations = itertools.permutations(range(9))

    return np.array([np.array(p) for p in permutations])


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

        self.all_possible_states = all_possible_states()
        self.len_states = len(self.all_possible_states)

        # 0: up, 1: left, 2: down, 3: right
        self.action_space = [0, 1, 2, 3]
        self.len_actions = len(self.action_space)

        # initialize Q(s, a)
        if self.use_random_values:
            self.Q = np.random.rand(self.len_states, self.len_actions)
            # terminal when position >= 0.45 (the goal position on top of the right hill)
            idx_goal_state = self.state_to_idx(self.env.get_goal_state())
            self.Q[idx_goal_state, :] = 0
        else:
            self.Q = np.zeros((self.len_states, self.len_actions))

        # steps, rewards to then plot it in the end
        self.steps = []
        self.rewards = []

    def train_episode(self):
        state, info = self.env.reset()

        done = False
        steps = 0
        rewards = 0

        visited_states = set([])

        start = time.time()

        while not done and steps < 1000:
            # choose action
            action = self.choose_action(state)
            # take action
            state_, reward, done, truncated, info = self.env.step(action)

            state_idx = self.state_to_idx(state)
            state_idx_ = self.state_to_idx(state_)

            # update Q(s, a)
            # Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(a)(Q(s_, a)) - Q(s, a))

            self.Q[state_idx, action] += self.alpha * (
                    reward + self.gamma * np.max(self.Q[state_idx_, :]) - self.Q[state_idx, action])


            visited_states.add(int(state_idx[0]))


            state = state_

            steps += 1
            print(f"Reward: {reward}")

            if len(visited_states) % 100 == 0:
                print(f"Visited {len(visited_states)} states")
                print(f"Epsilon {self.epsilon}")
                #for i in visited_states:
                #    print(self.Q[i, :])

            if steps % 100 == 0:
                print(f"Step {steps}")

        print(f"\n\nTime taken: {time.time() - start} seconds\n\n")
    def train(self, episodes):
        for e in range(episodes):
            self.train_episode()
            self.epsilon -= self.decay_rate
            print("\n")
            print(f"Episode {e} finished")



    def idx_to_state(self, idx):
        return self.all_possible_states[idx]

    def state_to_idx(self, state):
        return np.where(np.all(self.all_possible_states == state.flatten(), axis=1))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(np.arange(self.len_actions))
        else:
            state_idx = self.state_to_idx(state)
            return np.argmax(self.Q[state_idx, :])



