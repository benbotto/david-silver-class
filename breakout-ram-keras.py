import gym
import math
import numpy as np
import random
import tensorflow as tf
import time
from SumTree import SumTree

env = gym.make('Breakout-ram-v0')

ACT_SIZE            = env.action_space.n
LEARN_RATE          = 0.0002
REP_SIZE            = 1000000
REP_BATCH_SIZE      = 32
REP_LASTOBS         = 0
REP_ACTION          = 1
REP_REWARD          = 2
REP_NEWOBS          = 3
REP_DONE            = 4
GAMMA               = .99
EPSILON_MIN         = .1
EPSILON_DECAY_OVER  = 1000000
EPSILON_DECAY_RATE  = (EPSILON_MIN - 1) / EPSILON_DECAY_OVER
TEST_INTERVAL       = 100
TARGET_UPD_INTERVAL = 10000
MODEL_FILE_NAME     = "weights_breakout-ram-keras__ddqn_prio_2018_02_02_02_23.h5"

def getEpsilon(totalT):
  return max(EPSILON_DECAY_RATE * totalT + 1, EPSILON_MIN)

def buildModel():
  # Define the network model.
  model = tf.keras.models.Sequential()

  model.add(tf.keras.layers.Lambda(lambda x: x / 255.0, input_shape=env.observation_space.shape))
  model.add(tf.keras.layers.Dense(128, activation="relu"))
  model.add(tf.keras.layers.Dense(128, activation="relu"))
  model.add(tf.keras.layers.Dense(128, activation="relu"))
  model.add(tf.keras.layers.Dense(128, activation="relu"))
  model.add(tf.keras.layers.Dense(ACT_SIZE, activation="linear"))
  opt = tf.keras.optimizers.RMSprop(lr=LEARN_RATE)

  model.compile(loss="mean_squared_error", optimizer=opt)

  return model

def updateTargetModel(model, targetModel):
  targetModel.set_weights(model.get_weights())
  model.save(MODEL_FILE_NAME)

def main():
  model = buildModel()
  model.summary()

  targetModel = buildModel()
  updateTargetModel(model, targetModel)

  # Array for holding past results.  This is "replayed" in the network to help
  # train.
  replay    = SumTree(REP_SIZE)
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
      randAct = False

      env.render()

      # Choose a random action occasionally so that new paths are explored.
      epsilon = getEpsilon(totalT)

      if np.random.rand() < epsilon and episode % TEST_INTERVAL != 0:
        action  = env.action_space.sample()
        randCt += 1
      elif episode % TEST_INTERVAL != 0:
        # Run the inputs through the network to predict an action and get the Q
        # table (the estimated rewards for the current state).
        Q = targetModel.predict(np.array([lastObs]))
        print('Q: {}'.format(Q))

        # Action is the index of the element with the hightest predicted reward.
        action = np.argmax(Q)
      else:
        # Run the inputs through the network to predict an action and get the Q
        # table (the estimated rewards for the current state).
        Q = model.predict(np.array([lastObs]))
        print('Q: {}'.format(Q))

        # Action is the index of the element with the hightest predicted reward.
        action = np.argmax(Q)

      # Apply the action.
      newObs, reward, done, _ = env.step(action)
      episodeReward += reward
      reward = np.sign(reward)
      #print('t, newobs, reward, done')
      #print(t, newObs, reward, done)

      if episode % TEST_INTERVAL != 0:
        # Save the result for replay.  Priority is set to 1 here (high replay
        # prob) and updated on the forward pass.
        replay.add(1.0, (lastObs, action, reward, newObs, done))

        if replay.size >= REP_BATCH_SIZE:
          # Create training data from the replay array.
          segment = replay.total() / REP_BATCH_SIZE
          batch   = []
          indices = []

          for i in range(REP_BATCH_SIZE):
            print('{}, {}, {}, {}, {}'.format(segment, i, segment * i, segment * (i + 1), np.random.uniform(segment * i, segment * (i + 1))))
            (ind, p, data) = replay.get(np.random.uniform(segment * i, segment * (i + 1)))
            batch.append(data)
            indices.append(ind)
            print(ind, p, data)

          # Predictions from the old states, which will be updated to act as the
          # training target.
          # Using Double DQN.
          target = model.predict(np.array([rep[REP_LASTOBS] for rep in batch]))
          newQ   = targetModel.predict(np.array([rep[REP_NEWOBS] for rep in batch]))
          actSel = model.predict(np.array([rep[REP_NEWOBS] for rep in batch]))

          for i in range(len(batch)):
            act = np.argmax(actSel[i])

            if batch[i][REP_DONE]:
              target[i][batch[i][REP_ACTION]] = batch[i][REP_REWARD]
            else:
              target[i][batch[i][REP_ACTION]] = batch[i][REP_REWARD] + GAMMA * newQ[i][act]

            error = np.abs(target[i][batch[i][REP_ACTION]] - (batch[i][REP_REWARD] + GAMMA * newQ[i][act]))
            p = (error + .01) ** .6
            replay.update(indices[i], p)

          mse = model.train_on_batch(np.array([rep[REP_LASTOBS] for rep in batch]), target)
          print(mse)

          if totalT % TARGET_UPD_INTERVAL == 0:
            print("Updating target model.")
            updateTargetModel(model, targetModel)

      lastObs = newObs

      #time.sleep(.02)

    if episodeReward > maxReward:
      maxReward = episodeReward

    print('Episode {} went for {} timesteps, {} total.  {} rand acts.  Episode reward: {}.  Best reward: {}.  Epsilon: {}'
      .format(episode, t, totalT, randCt, episodeReward, maxReward, epsilon))

if __name__ == "__main__":
  main()

