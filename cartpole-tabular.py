import gym
import math
import numpy
import random
import time
import pdb

env = gym.make('CartPole-v0')

NUM_EPISODES = 200

# Observation ranges.
OBS_RNG_X         = env.observation_space.high[0] - env.observation_space.low[0]
OBS_RNG_X_VEL     = 2.4 - -2.4 # Space is infinite.  2.4 is the bounds, per the doc.
OBS_RNG_THETA     = env.observation_space.high[2] - env.observation_space.low[2]
OBS_RNG_THETA_VEL = math.pi - -math.pi # Space is infinite.

# Number of discrete states for each observation.  More states is
# more precise, but takes longer to learn.
OBS_STATES_X         = 1
OBS_STATES_X_VEL     = 1
OBS_STATES_THETA     = 6
OBS_STATES_THETA_VEL = 5

# Higher means more exploration.  Basically, an e-greedy method is used,
# but e is decayed over time as a function of episode.
EXPLORE_CONST = 30

def main():
  # This is the Q table, which is initialized to all zeros.  An observation, 4
  # numbers, points to a pair of expected rewards corresponding two the two
  # possible actions.
  qTbl = numpy.zeros((
    OBS_STATES_X, OBS_STATES_X_VEL, OBS_STATES_THETA, OBS_STATES_THETA_VEL,
    env.action_space.n))

  for episode in range(NUM_EPISODES):
    lastObs = obsToTuple(env.reset())

    done = False
    t    = 0 # t is the timestep.

    while not done:
      t += 1
      env.render()

      # Choose a random action occasionally, but generally choose the action that
      # is best in Q.
      if random.random() < math.pow(2, -episode / EXPLORE_CONST):
        action = env.action_space.sample()
      else:
        # Pick the index (0 or 1) with the highest value.
        action = numpy.argmax(qTbl[lastObs])

      # Apply the action.
      newObs, reward, done, _ = env.step(action)
      newObs = obsToTuple(newObs)
      #print(t, newObs, reward, done)

      # For the new observatsion, this is the highest reward (the reward, not
      # the index/action).
      bestReward = numpy.amax(qTbl[newObs])
      oldEst     = qTbl[lastObs + (action,)]

      # Update the Q table for the last observation-action pair.
      learnRate = 1 / t
      qTbl[lastObs + (action,)] = oldEst + learnRate * (reward + bestReward - oldEst)
      lastObs = newObs

      time.sleep(.02)

    print('Episode {} went for {} timesteps.'.format(episode+1, t))
    #print(qTbl)

  print('Testing the Q table for 2000 iterations...')
  obs = obsToTuple(env.reset())
  for t in range(2000):
    env.render()
    action = numpy.argmax(qTbl[obs])
    obs, _, _, _  = env.step(action)
    obs = obsToTuple(obs)
    print('Timestep {}.'.format(t))
    time.sleep(.02)

'''
  A step function that converts a continuous observation (obs) to a
  discrete integer index in the range [0, numSteps).
'''
def makeDiscrete(obsRange, numSteps, obs):
  stepSize = obsRange / numSteps
  min      = -(obsRange / 2)

  for i in range(numSteps):
    if obs < min + (i + 1) * stepSize:
      return i
  return numSteps - 1

'''
  Convert the observed state to a tuple of discrete integers.  This tuple
  can the index the qTbl.
'''
def obsToTuple(obs):
  xInd  = makeDiscrete(OBS_RNG_X,         OBS_STATES_X,         obs[0])
  xvInd = makeDiscrete(OBS_RNG_X_VEL,     OBS_STATES_X_VEL,     obs[1])
  tInd  = makeDiscrete(OBS_RNG_THETA,     OBS_STATES_THETA,     obs[2])
  tvInd = makeDiscrete(OBS_RNG_THETA_VEL, OBS_STATES_THETA_VEL, obs[3])

  return (xInd, xvInd, tInd, tvInd)

if __name__ == "__main__":
  main()

