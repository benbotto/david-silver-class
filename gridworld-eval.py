'''
Toy gridworld problem from David Silver's lecture on dynamic programming.
https://www.youtube.com/watch?v=Nd1-UUMVfz4&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=3
(~20 minutes in).

The result is a value function for the gridworld, and convergence is reached at
about 407 steps.  The value at each square of the grid shows how many steps it
would take, on average, to reach a terminal state using a random policy.  For
example, if staring at square (1, 0), it will take on average 14 moves to get
to a terminal state using a random policy.
'''
import numpy as np
import random

# Random policy, so the probability of taking a move is 1/4.
MOVE_PROB = .25

# Reward for moving from one square to another.
STEP_REWARD = -1

# Shape of the grid.
GRID_SHAPE = (4, 4)

# The number of full sweeps through the board to complete.  (It would be better
# to stop when convergence is reached.)
NUM_EPISODES = 500

def main():
  # The grid world, which will be converge on the values over time.
  grid = np.zeros(GRID_SHAPE)

  for episode in range(NUM_EPISODES):
    # A copy of the grid which is used to hold all the new values during the
    # sweep.
    new_grid = np.copy(grid)

    # Sweep the board and update values.
    for x in range(GRID_SHAPE[0]):
      for y in range(GRID_SHAPE[1]):
        square = (x, y)
        new_grid[square] = calc_new_values(square, grid)

    # Now copy the updated values back to the grid.
    grid = np.copy(new_grid)

    print('Episode {}'.format(episode + 1))
    print(grid)

# Calculate the new value for a square on the grid by looking at all the
# adjacent square values.
def calc_new_values(square, grid):
  old_val = grid[square]

  # No updates on terminal squares (game over).
  if is_terminal(square):
    return 0

  # Immediate reward.
  value = STEP_REWARD

  # Reward from moving each direction times the probability of taking that
  # move.  If moving in a direction is not possible because the square is at
  # the rim of the grid, then the reward is the old value.  E.g. if at square
  # (1, 0), moving up remains in square (1, 0) and the reward is the existing
  # value.

  #Left.
  reward = old_val if square[0] == 0 else grid[(square[0] - 1, square[1])]
  value += MOVE_PROB * reward

  # Right.
  reward = old_val if square[0] == GRID_SHAPE[0] - 1 else grid[(square[0] + 1, square[1])]
  value += MOVE_PROB * reward

  # Up.
  reward = old_val if square[1] == 0 else grid[(square[0], square[1] - 1)]
  value += MOVE_PROB * reward

  # Down.
  reward = old_val if square[1] == GRID_SHAPE[1] - 1 else grid[(square[0], square[1] + 1)]
  value += MOVE_PROB * reward

  return value

# Check if the square is a terminal square.
def is_terminal(square):
  return square[0] == 0 and square[1] == 0 or \
    square[0] == GRID_SHAPE[0] - 1 and square[1] == GRID_SHAPE[1] - 1

if __name__ == "__main__":
  main()
