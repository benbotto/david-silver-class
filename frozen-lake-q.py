import gym
import numpy as np
import math
from random import random

# See the SARSA implementation for notes about these constants.
NUM_EPISODES     = 20000
NUM_EVAL         = 5
STEP_SIZE        = .8
GREEDYNESS_CONST = .003
DISCOUNT_FACTOR  = .95

env           = gym.make('FrozenLake-v0')
qTbl          = np.zeros((env.observation_space.n, env.action_space.n))
wins          = 0
winsAtEpisode = []
winner        = False
episode       = -1

def getAction(lastObs, episode):
  if random() < math.pow(2, -episode * GREEDYNESS_CONST) or np.sum(qTbl[lastObs]) == 0:
    return env.action_space.sample()
  else:
    return np.argmax(qTbl[lastObs])

# This is the same as SARSA except that Q is updated using the value of
# the maximum action for S'.
while not winner:
  episode += 1
  if episode % 100 == 0:
    print('Starting episode {}.'.format(episode))

  done    = False
  lastObs = env.reset()

  while not done:
    action = getAction(lastObs, episode)
    newObs, reward, done, _ = env.step(action)

    qTbl[(lastObs, action)] += STEP_SIZE * (reward + DISCOUNT_FACTOR * np.max(qTbl[newObs]) - qTbl[(lastObs, action)])

    wins   += reward
    lastObs = newObs

  # The task is considered complete when the agent gets 78 / 100 episodes
  # correct.
  winsAtEpisode.append(wins)

  if episode >= 100:
    last100 = winsAtEpisode[episode] - winsAtEpisode[episode - 100]

    if last100 >= 78:
      print('Won at least 78 of the last 100 episdoes. Episode: {}'.format(episode))
      winner = True

print('Wins: {}. Losses: {}.'.format(wins, NUM_EPISODES - wins))
print(qTbl)

policy = []
for vals in qTbl:
  policy.append(np.argmax(vals))
print('Trying the policy.')
print(policy)

wins = 0
for episode in range(NUM_EVAL):
  print('\n\nStarting episode {}.'.format(episode))

  obs = env.reset()
  done = False

  while not done:
    env.render()
    obs, reward, done, _ = env.step(policy[obs])
    wins   += reward

print('Wins: {}. Losses: {}.'.format(wins, NUM_EVAL - wins))

