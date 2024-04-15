import numpy as np
import pygame
import sys
import itertools


class EightPuzzleEnv:

    def __init__(self, init_state=None):
        self.actions = [0, 1, 2, 3]
        self.goal_state = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])

        self.prev_state = None
        self.state = None

        self.init_state = init_state

        # (y, x) position of the empty tile
        self.empty_tile_pos = None

    def reset(self):

        # here generate a random initial state
        self.state = self.generate_state()

        self.empty_tile_pos = np.where(self.state == 0)

        return self.state, {}

    def get_goal_state(self):
        return self.goal_state

    def step(self, action):
        # action is number 0, 1, 2, 3
        # 0: up, 1: down, 2: left, 3: right
        # move the empty tile in the direction of the action

        if action not in self.actions:
            print("Invalid action")
            print("Action should be 0, 1, 2, 3")
            return

        self.prev_state = self.state.copy()

        self.move_tile(action)


        if self.is_goal(self.state):
            reward = 100

        elif np.array_equal(self.state, self.prev_state):
            reward = -100
        else:
            reward = -1

        # self.render()



        # return the next_state, reward, done, truncated, info
        return self.state, reward, self.is_goal(self.state), False, {}

    def move_tile(self, direction):

        if direction == 0:
            # move up
            if self.empty_tile_pos[0] > 0:
                self.state[self.empty_tile_pos], self.state[self.empty_tile_pos[0] - 1, self.empty_tile_pos[1]] = \
                    self.state[self.empty_tile_pos[0] - 1, self.empty_tile_pos[1]], self.state[self.empty_tile_pos]
                self.empty_tile_pos = (self.empty_tile_pos[0] - 1, self.empty_tile_pos[1])

            else:
                print("Invalid move")

        elif direction == 1:

            if self.empty_tile_pos[0] < 2:
                self.state[self.empty_tile_pos], self.state[self.empty_tile_pos[0] + 1, self.empty_tile_pos[1]] = \
                    self.state[self.empty_tile_pos[0] + 1, self.empty_tile_pos[1]], self.state[self.empty_tile_pos]
                self.empty_tile_pos = (self.empty_tile_pos[0] + 1, self.empty_tile_pos[1])

            else:
                print("Invalid move")

        elif direction == 2:
            if self.empty_tile_pos[1] > 0:
                self.state[self.empty_tile_pos], self.state[self.empty_tile_pos[0], self.empty_tile_pos[1] - 1] = \
                    self.state[self.empty_tile_pos[0], self.empty_tile_pos[1] - 1], self.state[self.empty_tile_pos]
                self.empty_tile_pos = (self.empty_tile_pos[0], self.empty_tile_pos[1] - 1)
            else:
                print("Invalid move")

        elif direction == 3:
            if self.empty_tile_pos[1] < 2:
                self.state[self.empty_tile_pos], self.state[self.empty_tile_pos[0], self.empty_tile_pos[1] + 1] = \
                    self.state[self.empty_tile_pos[0], self.empty_tile_pos[1] + 1], self.state[self.empty_tile_pos]
                self.empty_tile_pos = (self.empty_tile_pos[0], self.empty_tile_pos[1] + 1)
            else:
                print("Invalid move")

    def render(self):
        # TODO: implement this function with pygame
        print(self.state)

    def generate_state(self):
        # generate a random state
        while True:
            state = np.random.permutation(9).reshape(3, 3)
            if self.is_solvable(state):
                break

        return state

    def is_solvable(self, state):
        # check if the state is solvable
        # https://www.geeksforgeeks.org/check-instance-8-puzzle-solvable/
        # by counting the number of inversions
        # if sum of inversions is even, then the state is solvable
        flatten = state.flatten()

        # find the position of the empty tile and delete it
        flatten = np.delete(flatten, np.where(flatten == 0))

        inversion_count = 0

        for i in range(8):
            for j in range(i + 1, 8):
                if flatten[j] > flatten[i]:
                    inversion_count += 1

        print(inversion_count)

        return inversion_count % 2 == 0

    def is_goal(self, state):
        return np.array_equal(state, self.goal_state)

    def is_truncated(self):
        # TODO: implement this function
        pass

    def get_reward(self, state):
        return -self.manhattan_distance(state) / 20.0

    def manhattan_distance(self, state):
        distance = 0
        for i in range(3):
            for j in range(3):
                if state[i, j] != 0:
                    x, y = np.where(self.goal_state == state[i, j])
                    distance += abs(i - x) + abs(j - y)

        return distance
