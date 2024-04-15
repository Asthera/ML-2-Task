import itertools
import time

import numpy as np


# generate all possible list of lists of size 3x3 with numbers 0-8
def generate_all_states():
    permutations = list(itertools.permutations(range(2)))

    return permutations


permutation = generate_all_states()

print(permutation, type(permutation))
print(permutation[0], type(permutation[0]))


Dict = {}

for i in range(2):
    Dict[permutation[i]] = i

print(Dict)



start = time.time()

array_to_find = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
print(array_to_find)

all_states = generate_all_states()
print(all_states)

flatten_array_to_find = array_to_find.flatten()
print(flatten_array_to_find, type(flatten_array_to_find))
index = np.where(np.all(all_states == flatten_array_to_find, axis=1))
print(index)

print(f"Time taken: {time.time() - start} seconds")
print(len(all_states))

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
