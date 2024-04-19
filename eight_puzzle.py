import numpy as np
import pygame
import sys
import itertools


class EightPuzzleEnv:

    def __init__(self,
                 limit_count_steps: int | None = 10_000,
                 render: bool = False,
                 reward_type: str = "small-penalty" or "manhattan" or "hamming"
                 ):
        self.goal_state = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])

        self.action_space = [0, 1, 2, 3]

        self.prev_state = None
        self.state = None

        # (y, x) position of the empty tile
        self.empty_tile_pos = None

        # variable for store a limit for truncated episodes
        self.limit_count_steps = limit_count_steps

        self.current_step = 0

        self.reward_type = reward_type


    def reset(self) -> (tuple, dict):
        """
        Reset the environment
        1. Generate a random initial state
        2. Set the empty tile position
        3. Set the current step to 0

        :return:

        1. tuple: the new state
            example: (1, 2, 3, 4, 5, 6, 7, 0, 8)
        2. dict: empty dictionary (in later implementations, it can be used to return additional information)
        """

        # here generate a random initial state
        self.state = self.generate_state()

        self.empty_tile_pos = np.where(self.state == 0)

        self.current_step = 0

        return tuple(self.state.flatten()), {}

    def get_goal_state(self) -> np.ndarray:
        """"
        Return the goal state

        :return: the goal state
            example: np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
        """

        return self.goal_state

    def step(self, action: int) -> (tuple, float, bool, bool, dict):
        """
        Take an action and return the next state, reward, done, truncated, and info

        Increase the current step by 1

        :param action:
        :return:
        1. tuple: the new state
            example: (1, 2, 3, 4, 5, 6, 7, 0, 8)
        2. float: reward
        3. bool: done
        4. bool: truncated
        5. dict: empty dictionary (in later implementations, it can be used to return additional information)

        """

        self.prev_state = self.state.copy()
        self.move_tile(action)

        new_state = self.state.copy()
        new_state = tuple(new_state.flatten())

        self.current_step += 1

        # self.render()

        return new_state, self.get_reward(self.state), self.is_goal(self.state), self.is_truncated(), {}

    def move_tile(self, direction: int) -> None:
        """
        Move the empty tile in the direction of the action

        :param direction: 0: up, 1: down, 2: left, 3: right

        :return: None (update the state)
        """

        if direction == 0:
            if self.empty_tile_pos[0] > 0:
                self.state[self.empty_tile_pos], self.state[self.empty_tile_pos[0] - 1, self.empty_tile_pos[1]] = \
                    self.state[self.empty_tile_pos[0] - 1, self.empty_tile_pos[1]], self.state[self.empty_tile_pos]
                self.empty_tile_pos = (self.empty_tile_pos[0] - 1, self.empty_tile_pos[1])

        elif direction == 1:
            if self.empty_tile_pos[0] < 2:
                self.state[self.empty_tile_pos], self.state[self.empty_tile_pos[0] + 1, self.empty_tile_pos[1]] = \
                    self.state[self.empty_tile_pos[0] + 1, self.empty_tile_pos[1]], self.state[self.empty_tile_pos]
                self.empty_tile_pos = (self.empty_tile_pos[0] + 1, self.empty_tile_pos[1])


        elif direction == 2:
            if self.empty_tile_pos[1] > 0:
                self.state[self.empty_tile_pos], self.state[self.empty_tile_pos[0], self.empty_tile_pos[1] - 1] = \
                    self.state[self.empty_tile_pos[0], self.empty_tile_pos[1] - 1], self.state[self.empty_tile_pos]
                self.empty_tile_pos = (self.empty_tile_pos[0], self.empty_tile_pos[1] - 1)

        elif direction == 3:
            if self.empty_tile_pos[1] < 2:
                self.state[self.empty_tile_pos], self.state[self.empty_tile_pos[0], self.empty_tile_pos[1] + 1] = \
                    self.state[self.empty_tile_pos[0], self.empty_tile_pos[1] + 1], self.state[self.empty_tile_pos]
                self.empty_tile_pos = (self.empty_tile_pos[0], self.empty_tile_pos[1] + 1)

    def render(self) -> None:
        # TODO: implement this function with pygame
        print(self.state)

    def generate_state(self) -> np.ndarray:
        """
        Generate a random initial state
        :return:
        1. np.ndarray: the initial state (3x3)
        """

        state = np.random.permutation(9).reshape(3, 3)

        while not self.is_solvable(state):
            state = np.random.permutation(9).reshape(3, 3)

        return state

    def is_solvable(self, state: np.ndarray) -> bool:
        """"
        Check if the state is solvable

        :param state: the state to check
        :return: True if the state is solvable, False otherwise

        Reference: https://math.stackexchange.com/questions/293527/how-to-check-if-a-8-puzzle-is-solvable
        """

        state_copy = state.copy()
        flatten = state_copy.flatten()

        # find the position of the empty tile and delete it
        flatten = np.delete(flatten, np.where(flatten == 0))

        inversion_count = 0

        for i in range(8):
            for j in range(i + 1, 8):
                if flatten[j] > flatten[i]:
                    inversion_count += 1

        return inversion_count % 2 == 0

    def is_goal(self, state: np.ndarray) -> bool:
        """"
        Check if the state is the goal state

        :param state: the state to check
        :return: True if the state is the goal state, False otherwise
        """

        return np.array_equal(state, self.goal_state)

    def is_truncated(self) -> bool or None:
        """
        Check if the episode is truncated
          Count the number of steps and return True if it exceeds the limit
        :return:

        1. bool: True if the episode is truncated, False otherwise

        OR None if the limit_count_steps is not set
        """

        if self.limit_count_steps is None:
            raise ValueError("The limit_count_steps is not set")

        return self.current_step >= self.limit_count_steps

    def get_reward(self, state: np.ndarray) -> float:
        """
        if invalid move (array not changes)
        :return -100

        :param state:
        :return:
        """
        if np.array_equal(state, self.goal_state):
            return 100

        if np.array_equal(self.prev_state, state):
            return -100

        if self.reward_type == "manhattan":
            prev_distance = self.manhattan_distance(self.prev_state)
            current_distance = self.manhattan_distance(state)

            diff = abs(current_distance - prev_distance) / 100.0

            if current_distance < prev_distance:
                return -1 + diff

            return -1 - diff

        elif self.reward_type == "hamming":
            prev_distance = self.hamming_distance(self.prev_state)
            current_distance = self.hamming_distance(state)

            diff = abs(current_distance - prev_distance) / 10.0

            if current_distance < prev_distance:
                return -1 + diff

            return -1 - diff

        elif self.reward_type == "small-penalty":
            return -1
        else:
            raise ValueError("Invalid reward type")

    def manhattan_distance(self, state: np.ndarray) -> int:
        """"
        It one of types of reward functions

        Calculate the Manhattan distance between the current state and the goal state

        :param state: the state to calculate the distance
        :return: the Manhattan distance
        """

        distance = 0
        for i in range(3):
            for j in range(3):
                if state[i, j] != 0:
                    x, y = np.where(self.goal_state == state[i, j])
                    distance += abs(i - x) + abs(j - y)

        return distance[0]

    def hamming_distance(self, state: np.ndarray) -> int:
        """
        It one of types of reward functions

        Calculate the hamming distance between the current state and the goal state

        :param state:
        :return:
        """

        distance = 0
        size = 3

        for i in range(size):
            for j in range(size):
                # Skip the blank space in the calculation
                if state[i][j] != 0 and state[i][j] != self.goal_state[i][j]:
                    distance += 1

        return distance
