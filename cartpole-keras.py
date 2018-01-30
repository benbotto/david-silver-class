import gym
import math
import numpy as np
import random
import tensorflow as tf
import time

env = gym.make('CartPole-v0')

OBS_SIZE           = len(env.observation_space.low)
ACT_SIZE           = env.action_space.n
LEARN_RATE         = 0.001
REP_SIZE           = 5000
REP_BATCH_SIZE     = 32
REP_LASTOBS        = 0
REP_ACTION         = 1
REP_REWARD         = 2
REP_NEWOBS         = 3
REP_DONE           = 4
GAMMA              = .99
FIN_TRAIN_SOLVE_CT = 3
EPSILON            = .25

def main():
  # Define the network model.
  model = tf.keras.models.Sequential()

  model.add(tf.keras.layers.Dense(14, input_dim=OBS_SIZE, activation="relu"))
  model.add(tf.keras.layers.Dense(22, activation="relu"))
  model.add(tf.keras.layers.Dense(ACT_SIZE, activation="linear"))

  opt = tf.keras.optimizers.Adam(lr=LEARN_RATE)

  model.compile(loss="mean_squared_error", optimizer=opt, metrics=["mean_squared_error"])

  # Array for holding past results.  This is "replayed" in the network to help
  # train.
  replay     = []
  solveCt    = 0
  seqSolveCt = 0
  episode    = 0

  while seqSolveCt < 100:
    episode += 1
    done     = False
    t        = 0 # t is the timestep.
    randCt   = 0
    lastObs  = env.reset()

    while not done:
      t += 1
      env.render()

      # Run the inputs through the network to predict an action and get the Q
      # table (the estimated rewards for the current state).
      Q = model.predict(np.array([lastObs]))
      #print('Q: {}'.format(Q))

      # Action is the index of the element with the hightest predicted reward.
      action = np.argmax(Q)

      # Choose a random action occasionally so that new paths are explored.
      if np.random.rand() < EPSILON and seqSolveCt < FIN_TRAIN_SOLVE_CT:
        action  = env.action_space.sample()
        randCt += 1

      # Apply the action.
      newObs, reward, done, _ = env.step(action)
      #print('t, newobs, reward, done')
      #print(t, newObs, reward, done)

      if done and t == 200:
        solveCt    += 1
        seqSolveCt += 1
      elif done:
        seqSolveCt = 0
      
      # Save the result for replay.
      replay.append((lastObs, action, reward, newObs, done))

      if len(replay) > REP_SIZE:
        replay.pop(np.random.randint(REP_SIZE + 1))
        
      if seqSolveCt < FIN_TRAIN_SOLVE_CT:
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

      lastObs = newObs

      #time.sleep(.02)

    print('Episode {} went for {} timesteps.  {} rand acts.  {} solves and {} sequential.'.format(episode, t, randCt, solveCt, seqSolveCt))

if __name__ == "__main__":
  main()

