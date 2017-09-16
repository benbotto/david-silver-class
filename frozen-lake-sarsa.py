import gym
import numpy as np
import math
from random import random

# How many episodes to run.
NUM_EPISODES = 4000

# How many evaluation episodes to run.
NUM_EVAL = 1000

# The step size, usually denoted α (alpha), is sometimes called the learning rate.
STEP_SIZE = .1

# Using decaying ε-greedy (epsilon) for exploration.
GREEDYNESS_CONST = .004

# Usually denoted γ (gamma), this generally determins how much to favor future
# rewards over immediate ones.
DISCOUNT_FACTOR = .95

env = gym.make('FrozenLake-v0')

# There are 16 positions, and 4 actions (up, left, down, right).  Each entry
# in Q is indexed by a state-action pair, and the value gives the expected
# utility of taking an action from a state.
qTbl = np.zeros((env.observation_space.n, env.action_space.n))

# Helper function to pick an action using ε-greedy.
def getAction(lastObs, episode):
  if random() < math.pow(2, -episode * GREEDYNESS_CONST) or np.sum(qTbl[lastObs]) == 0:
    # Take a random action (explore).
    action = env.action_space.sample()
  else:
    # Take the best action (greedy).
    action = np.argmax(qTbl[lastObs])

  return action

wins = 0

for episode in range(NUM_EPISODES):
  if episode % 100 == 0:
    print('Starting episode {}.'.format(episode))

  done       = False
  lastObs    = env.reset()
  lastAction = getAction(lastObs, episode)

  while not done:
    #env.render()

    # Apply the last action, and get a new state and action.
    newObs, reward, done, _ = env.step(lastAction)
    newAction = getAction(lastObs, episode)

    # Update Q.
    qTbl[(lastObs, lastAction)] += STEP_SIZE * (reward + DISCOUNT_FACTOR * qTbl[(newObs, newAction)] - qTbl[(lastObs, lastAction)])
    #qTbl[(lastObs, lastAction)] += STEP_SIZE * (reward + DISCOUNT_FACTOR * np.max(qTbl[newObs]) - qTbl[(lastObs, lastAction)])

    # Keep track of the total number of wins (reward is 1 or 0).
    wins += reward

    # Set up the next round.
    lastObs    = newObs
    lastAction = newAction

print('Wins: {}. Losses: {}.'.format(wins, NUM_EPISODES - wins))
print(qTbl)

policy = []
for vals in qTbl:
  policy.append(np.argmax(vals))
print(policy)

print('Trying the policy.')
wins = 0
for episode in range(NUM_EVAL):
  if episode % 100 == 0:
    print('Starting episode {}.'.format(episode))

  obs = env.reset()
  done = False

  while not done:
    obs, reward, done, _ = env.step(policy[obs])
    wins += reward

print('Wins: {}. Losses: {}.'.format(wins, NUM_EVAL - wins))

