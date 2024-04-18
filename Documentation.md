# Documentation


## Problem description

Have 8-puzzle game, where the goal is to move the tiles to the correct position.

So we have a 3x3 grid with 8 tiles and one empty space.

The tiles are numbered from `1` to `8` and the empty space is represented by `0`.

The Goal position is:
```
1|2|3
-+-+-
4|5|6
-+-+-
7|8|0
```

The tiles can be moved in four directions: up, down, left, right.

So, if we have the following starting position:
```
1|2|3
-+-+-
4|5|6
-+-+-
7|0|8
```

1. We need to move the `8` tile to `left`, so the `empty space (0 tile)` can be moved to the `right`. 
<br/>

And we will have our goal position.

## Observation Space

The observation is a tuple of 9 elements, where each element is a number from `0` to `8`.

For example, the starting position will be represented as:

```
1|2|3
-+-+-
4|5|6   ---> (1, 2, 3, 4, 5, 6, 7, 0, 8)
-+-+-
7|0|8
```

Observation space is `discrete` and has `9! = 362880` possible states.

## Action Space

The action is a number from `0` to `3`, where each number represents a `direction` to `move` the `empty space` `(0 tile)`.

- `0` - up
- `1` - down
- `2` - left
- `3` - right

Action space is `discrete` and has `4` possible actions.


## Generating a solvable 8-puzzle

To check is generated 8-puzzle is solvable, we need to count the number of inversions.

An inversion is when a tile precedes another tile with a lower number on it.

When counting the inversions, we need to ignore the `0` tile.

When count of the inversion is `even`, the puzzle is solvable. Otherwise, it is not solvable.

Example:
```
5|2|8
-+-+-
4|1|7
-+-+-
0|3|6
```

```
5 precedes 1,2,3,4 - 4 inversions
2 precedes 1 - 1 inversion
8 precedes 1,3,4,6,7 - 5 inversions
4 precedes 1,3 - 2 inversions
1 precedes none - 0 inversions
7 precedes 3,6 - 2 inversions
3 precedes none - 0 inversions
6 precedes none - 0 inversions

Total inversions = 4 + 1 + 5 + 2 + 0 + 2 + 0 + 0 = 14 (Even Number) 
So this puzzle is solvable.
```

Example of implementation:
```python
    def is_solvable(self, state):
    
        flatten = np.delete(flatten, np.where(flatten == 0))

        inversion_count = 0

        for i in range(8):
            for j in range(i + 1, 8):
                if flatten[j] > flatten[i]:
                    inversion_count += 1

        return inversion_count % 2 == 0
є
```
[Reference](https://math.stackexchange.com/questions/293527/how-to-check-if-a-8-puzzle-is-solvable)

## Environment Implementation Details

# TODO rewrite

- The environment is implemented in eight_puzzle.py as `EightPuzzleEnv` class.

- Reward function is implemented as `get_reward` method.
  - Have 3 types of it:
    - "simple-penalty" 
      - for each move, the reward is `-1`
      - when the puzzle is solved, the reward is `0`
      - for not valid moves, the reward is `-100`
    - "manhattan"
      - for each move, the reward computed as `manhattan distance between the current state and the goal state`
      - when the puzzle is solved, the reward is `0`
      - for not valid moves, the reward is `-100`
    - "hamming"

## Q-Learning

- The Q-Learning algorithm is implemented in q_learning.py as `QLearning` class.

### Changes made to the original algorithm

1. Experimenting with initial Q-values (random, zeros)


#### Pseudocode:


![Pseudocode](Pseudocodes/q-learning.jpg)
<p align="center">Zdroj: Sutton-Barto: Reinforcement Learning, 2nd ed., 2018</p>


### Optimizations for Q-Learning

Firstly, for Q table we used numpy array with all possible states as tuple and one more 2d array with states as integers (so firstly find index in 1d array and then in 2d array) and actions.
It was not efficient, because we had to find index in 1d array with len == 362880. So had O(n) +  complexity.