import itertools
import time

import numpy as np


# generate all possible list of lists of size 3x3 with numbers 0-8
def generate_all_states():
    permutations = list(itertools.permutations(range(2)))

    return permutations


start = time.time()

d = {}

perm = generate_all_states()

for i in  perm:
    for j in range(3):
        d[i, j] = np.random.rand()

print(d)

# Find max value od dictionary based on fist key ex (0, 1)
values = np.array([d[(0, 1), i] for i in range(3)])

print(values, np.argmax(values), np.max(values))





print(f"Time taken: {time.time() - start} seconds")


# test manhattan distance

goal_state = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])

state = np.array([[2, 5, 3], [4, 1, 6], [7, 0, 8]])


def manhattan_distance(state, goal_state):
    distance = 0
    for i in range(3):
        for j in range(3):
            if state[i, j] != 0:
                x, y = np.where(goal_state == state[i, j])
                distance += abs(i - x) + abs(j - y)

    return distance


distance = manhattan_distance(state, goal_state)
print(distance[0], type(distance))
