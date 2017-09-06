from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from Blackjack import Blackjack

# How many times to play the game.
NUM_EPISODES = 500000

NUM_PLAYER_SCORES = 10 # 12 - 21
NUM_DEALER_CARDS  = 10 # ace - 10
NUM_USABLE_ACE    = 2  # 0 or 1.
NUM_STATES        = NUM_PLAYER_SCORES * NUM_DEALER_CARDS * NUM_USABLE_ACE

# This will get populated with the average returns for each state.  The first
# index is the player's score, 12-21, the second is the dealer's visible card
# (ace-10), and the third is whether or not the player has a usable ace (0 or
# 1).  Total of 200 states.
values = [0 for _ in range(NUM_STATES)]

# A list of the returns for each state.
returns = [[] for _ in range(NUM_STATES)]

# Helper function to get the index of a state in the values or returns arrays.
def getIndex(pScore, dVisScore, hasVisibleAce):
  # 12 is the lowest score the player can have, and corresponds to index 0.
  # 1 is the lowest visible card the dealer can have, and corresponds to index 0.
  return (pScore - 12) * (NUM_DEALER_CARDS * NUM_USABLE_ACE) + (dVisScore - 1) * (NUM_USABLE_ACE) + hasVisibleAce

env = Blackjack()

for episode in range(NUM_EPISODES):
  # Reset the environment.
  obs  = env.reset()
  done = False

  # This array is used to keep track of the states that are seen in an single
  # episode.
  episodeStateIndices = [getIndex(obs[0], obs[1], obs[2])]

  print('Episode {}.'.format(episode))

  while not done:
    # Player policy is to hit if the score is less than 20.
    action = Blackjack.HIT if obs[0] < 20 else Blackjack.STAY
    obs, reward, done = env.step(action)

    if obs[0] <= 21:
      episodeStateIndices.append(getIndex(obs[0], obs[1], obs[2]))

  # For each state seen, add the new reward and update the average value.
  for ind in episodeStateIndices:
    returns[ind].append(reward)
    values[ind] = sum(returns[ind]) / len(returns[ind])

# Plot the results.
X, Y = np.meshgrid(np.arange(12, 22), np.arange(1, 11), indexing='ij')
Z_nousable = []
Z_usable   = []

for p in range(NUM_PLAYER_SCORES):
  Z_nousable.append([])
  Z_usable.append([])

  for d in range(NUM_DEALER_CARDS):
    ind = p * (NUM_DEALER_CARDS * NUM_USABLE_ACE) + d * (NUM_USABLE_ACE)
    Z_nousable[p].append(values[ind])
    Z_usable[p].append(values[ind+1])

fig = plt.figure('Without Usable Ace')
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z_nousable)

fig = plt.figure('With Usable Ace')
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z_usable)
plt.show()
