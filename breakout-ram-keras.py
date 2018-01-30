import gym
import math
import numpy as np
import random
import tensorflow as tf
import time

env = gym.make('Breakout-ram-v0')

ACT_SIZE           = env.action_space.n
LEARN_RATE         = 0.0025
REP_SIZE           = 600000
REP_BATCH_SIZE     = 32
REP_LASTOBS        = 0
REP_ACTION         = 1
REP_REWARD         = 2
REP_NEWOBS         = 3
REP_DONE           = 4
GAMMA              = .99
EPSILON_MIN        = .1
EPSILON_DECAY_OVER = 500000
TEST_INTERVAL      = 100

def getEpsilon(totalT):
  return max((EPSILON_MIN - 1) / EPSILON_DECAY_OVER * totalT + 1, EPSILON_MIN)

def main():
  # Define the network model.
  model = tf.keras.models.Sequential()

  #model.add(tf.keras.layers.Flatten(input_shape=env.observation_space.shape))
  model.add(tf.keras.layers.Dense(14, input_shape=env.observation_space.shape, activation="relu"))
  model.add(tf.keras.layers.Dense(256, activation="relu"))
  model.add(tf.keras.layers.Dense(ACT_SIZE, activation="linear"))

  '''
  model.add(tf.keras.layers.Lambda(lambda x: x / 255.0, input_shape=inputShape))
  model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=8, strides=4, activation="relu"))
  model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, activation="relu"))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(256, activation="relu"))
  model.add(tf.keras.layers.Dense(ACT_SIZE, activation="linear"))
  '''

  '''
  model.add(tf.keras.layers.MaxPooling2D())
  model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=4, activation="relu"))
  model.add(tf.keras.layers.MaxPooling2D())
  model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"))
  model.add(tf.keras.layers.MaxPooling2D())
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(512, activation="relu"))
  model.add(tf.keras.layers.Dense(ACT_SIZE, activation="linear"))
  '''

  opt = tf.keras.optimizers.Adam(lr=LEARN_RATE)

  model.compile(loss="mean_squared_error", optimizer=opt, metrics=["mean_squared_error"])
  model.summary()

  # Array for holding past results.  This is "replayed" in the network to help
  # train.
  replay    = []
  episode   = 0
  maxReward = -1000000
  totalT    = 0

  while episode < 10000000:
    episode      += 1
    done          = False
    t             = 0 # t is the timestep.
    randCt        = 0
    episodeReward = 0
    lastObs       = env.reset()

    while not done:
      t      += 1
      totalT += 1

      env.render()

      # Choose a random action occasionally so that new paths are explored.
      epsilon = getEpsilon(totalT)

      if np.random.rand() < epsilon and episode % TEST_INTERVAL != 0:
        action  = env.action_space.sample()
        randCt += 1
      else:
        # Run the inputs through the network to predict an action and get the Q
        # table (the estimated rewards for the current state).
        Q = model.predict(np.array([lastObs]))
        #print('Q: {}'.format(Q))

        # Action is the index of the element with the hightest predicted reward.
        action = np.argmax(Q)

      # Apply the action.
      newObs, reward, done, _ = env.step(action)
      episodeReward += reward
      reward = np.sign(reward)
      #print('t, newobs, reward, done')
      #print(t, newObs, reward, done)

      if episode % TEST_INTERVAL != 0:
        # Save the result for replay.
        replay.append((lastObs, action, reward, newObs, done))

        if len(replay) > REP_SIZE:
          replay.pop(np.random.randint(REP_SIZE + 1))
          
        # Create training data from the replay array.
        batch = random.sample(replay, min(len(replay), REP_BATCH_SIZE))

        # Predictions from the old states, which will be updated to act as the
        # training target.
        target = model.predict(np.array([rep[REP_LASTOBS] for rep in batch]))
        newQ   = model.predict(np.array([rep[REP_NEWOBS] for rep in batch]))

        for i in range(len(batch)):
          if batch[i][REP_DONE]:
            target[i][batch[i][REP_ACTION]] = batch[i][REP_REWARD]
          else:
            target[i][batch[i][REP_ACTION]] = batch[i][REP_REWARD] + GAMMA * np.max(newQ[i])

        model.train_on_batch(np.array([rep[REP_LASTOBS] for rep in batch]), target)

        '''
        X     = []
        Y     = []

        for i in range(len(batch)):
          X.append(batch[i]['lastObs'])
          oldQ = model.predict(np.array([batch[i]['lastObs']]))
          newQ = model.predict(np.array([batch[i]['newObs']]))
          target = np.copy(oldQ)[0] # Not needed, I think.  Just use oldQ...

          if batch[i]['done']:
            target[batch[i]['action']] = batch[i]['reward']
          else:
            target[batch[i]['action']] = batch[i]['reward'] + GAMMA * np.max(newQ)
          Y.append(target)

        model.train_on_batch(np.array(X), np.array(Y))
        '''

      lastObs = newObs

      #time.sleep(.02)

    if episodeReward > maxReward:
      maxReward = episodeReward

    print('Episode {} went for {} timesteps, {} total.  {} rand acts.  Episode reward: {}.  Best reward: {}.  Epsilon: {}'
      .format(episode, t, totalT, randCt, episodeReward, maxReward, epsilon))

if __name__ == "__main__":
  main()

