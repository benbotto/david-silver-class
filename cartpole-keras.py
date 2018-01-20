import gym
import math
import numpy as np
import tensorflow as tf
import random
import time
import pdb

env = gym.make('CartPole-v0')

OBS_SIZE       = len(env.observation_space.low)
ACT_SIZE       = env.action_space.n
NUM_EPISODES   = 200000
LEARN_RATE     = 0.01
REP_SIZE       = 1000
REP_BATCH_SIZE = 100
GAMMA          = .99

# Higher means more exploration.  Basically, an e-greedy method is used,
# but e is decayed over time as a function of episode.
EXPLORE_CONST = 2000

def main():
  # Define the network model.
  model = tf.keras.models.Sequential()

  model.add(tf.keras.layers.Dense(12, input_dim=OBS_SIZE, activation="relu"))
  model.add(tf.keras.layers.Dense(24, activation="relu"))
  model.add(tf.keras.layers.Dense(ACT_SIZE, activation="sigmoid"))

  opt = tf.keras.optimizers.Adam(lr=LEARN_RATE)

  model.compile(loss="mean_squared_error", optimizer=opt, metrics=["mean_squared_error"])

  # Array for holding past results.  This is "replayed" in the network to help
  # train.
  replay = []

  for episode in range(NUM_EPISODES):
    lastObs = env.reset()
    done    = False
    t       = 0 # t is the timestep.

    while not done:
      t += 1
      #env.render()

      # Run the inputs through the network to predict an action and get the Q
      # table (the estimated rewards for the current state).
      Q = model.predict(np.array([lastObs]))
      #print('Q: {}'.format(Q))

      # Action is the index of the element with the hightest predicted reward.
      action = np.argmax(Q)

      # Choose a random action occasionally so that new paths are explored.
      if random.random() < math.pow(2, -episode / EXPLORE_CONST):
        action = env.action_space.sample()

      # Apply the action.
      #print('Applying action {}.'.format(action[0]))
      newObs, reward, done, _ = env.step(action)
      #print('t, newobs, reward, done')
      #print(t, newObs, reward, done)

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
      
      # Create training data from the replay array.
      batch = random.sample(replay, min(len(replay), REP_BATCH_SIZE))
      X     = []
      Y     = []

      for i in range(len(batch)):
        X.append(np.array([batch[i]['lastObs']]))
        oldQ = model.predict(np.array([batch[i]['lastObs']]))
        newQ = model.predict(np.array([batch[i]['newObs']]))
        target = np.copy(oldQ) # Not needed, I think.  Just use oldQ...
        target[0, batch[i]['action']] = batch[i]['reward'] + GAMMA * np.max(newQ)
        Y.append(target)
      print(X)
      model.train_on_batch(X, Y)

      '''
      # Now get Q' by feeding the new observation through the network.
      newQ = model.predict(np.array([newObs]))
      #print('New Q')
      #print(newQ)

      # Of the Q values, this one is the max (e.g. the best estimated reward).
      bestQ = np.max(newQ)
      #print('Best Q')
      #print(bestQ)

      # Update the Q table for the last action.  This is the new target.
      #gamma = 1 / t
      gamma = .99
      Q[0, action] = reward + gamma * bestQ
      #print('Target Q')
      #print(Q)

      # Update the weights (train) using the updated Q target.
      model.fit(x=np.array([lastObs]), y=Q, batch_size=1, epochs=1, verbose=0)
      '''

      lastObs = newObs

      #time.sleep(.02)

    print('Episode {} went for {} timesteps.'.format(episode+1, t))

if __name__ == "__main__":
  main()

