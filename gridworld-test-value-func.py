'''
Toy gridworld problem from David Silver's lecture on dynamic programming.
https://www.youtube.com/watch?v=Nd1-UUMVfz4&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=3
(~20 minutes in).

See gridworld-eval.py.  This empirically tests the cost of moving from a square
to the terminal state using a random policy.  Change START_SQUARE below to
verify the value function from gridworld-eval.py.
'''
import numpy as np
import random

# Move directions as numbers.
UP    = 0
LEFT  = 1
DOWN  = 2
RIGHT = 3

# Reward for moving from one square to another.
STEP_REWARD = -1

# Shape of the grid.
GRID_SHAPE = (4, 4)

# The number of full sweeps through the board to complete.
NUM_EPISODES = 100000

# Starting square.
START_SQUARE = [1, 2]

def main():
  values = np.zeros(NUM_EPISODES)

  for episode in range(NUM_EPISODES):
    square = START_SQUARE.copy()
    value  = 0

    while not is_terminal(square):
      value += STEP_REWARD
      move = policy()

      if move == LEFT and square[0] != 0:
        square[0] -= 1
      elif move == RIGHT and square[0] != GRID_SHAPE[0] - 1:
        square[0] += 1
      elif move == UP and square[1] != 0:
        square[1] -= 1
      elif move == DOWN and square[1] != GRID_SHAPE[1] - 1:
        square[1] += 1

    values[episode] = value
    print('Finished episode {} with a value of {}.'.format(episode + 1, value))

  print('Average value {}'.format(np.average(values)))

# Policy function (random move in one of the 4 directions).
def policy():
  return random.randint(0, 3)

def is_terminal(square):
  return square[0] == 0 and square[1] == 0 or \
    square[0] == GRID_SHAPE[0] - 1 and square[1] == GRID_SHAPE[1] - 1

if __name__ == "__main__":
  main()
