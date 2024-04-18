import itertools
import time

import numpy as np


# generate all possible list of lists of size 3x3 with numbers 0-8
def generate_all_states():
    permutations = list(itertools.permutations(range(9)))

    return permutations


permutation = generate_all_states()

print(permutation, type(permutation))
print(permutation[0], type(permutation[0]))


Dict = {}

for i in range(2):
    Dict[permutation[i]] = i

print(Dict)



start = time.time()



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


# test how work set, because we need to store visited states, and it save only half of the states
visited_states = set()
for i in permutation:
    visited_states.add(i)

print(visited_states, len(visited_states))


from eight_puzzle import EightPuzzleEnv

env = EightPuzzleEnv()
print(env.reward_type)

def hamming_distance(state, goal_state):
    distance = 0
    size = len(state)
    for i in range(size):
        for j in range(size):
            # Skip the blank space in the calculation
            if state[i][j] != 0 and state[i][j] != goal_state[i][j]:
                distance += 1
    return distance

# Example usage:
current_state = [
    [2, 8, 3],
    [1, 6, 4],
    [7, 0, 5]
]

goal_state = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

print("Hamming distance:", hamming_distance(current_state, goal_state))
