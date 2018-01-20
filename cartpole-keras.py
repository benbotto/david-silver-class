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
REP_SIZE           = 100
REP_BATCH_SIZE     = 20
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
      replay.append({
        'lastObs': lastObs,
        'newObs' : newObs,
        'reward' : reward,
        'done'   : done,
        'action' : action
      })

      if len(replay) > REP_SIZE:
        replay.pop(np.random.randint(REP_SIZE + 1))
        
      if seqSolveCt < FIN_TRAIN_SOLVE_CT:
        # Create training data from the replay array.
        batch = random.sample(replay, min(len(replay), REP_BATCH_SIZE))
        X     = []
        Y     = []

        for i in range(len(batch)):
          X.append(batch[i]['lastObs'])
          oldQ = model.predict(np.array([batch[i]['lastObs']]))
          newQ = model.predict(np.array([batch[i]['newObs']]))
          target = np.copy(oldQ)[0] # Not needed, I think.  Just use oldQ...

          if batch[i]['done']:
            target[batch[i]['action']] = -1
          else:
            target[batch[i]['action']] = batch[i]['reward'] + GAMMA * np.max(newQ)
          Y.append(target)

        model.train_on_batch(np.array(X), np.array(Y))

      lastObs = newObs

      #time.sleep(.02)

    print('Episode {} went for {} timesteps.  {} rand acts.  {} solves and {} sequential.'.format(episode, t, randCt, solveCt, seqSolveCt))

if __name__ == "__main__":
  main()

