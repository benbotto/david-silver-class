import gym
import math
import numpy as np
import tensorflow as tf
import random
import time
import pdb

env = gym.make('CartPole-v0')

OBS_SIZE     = len(env.observation_space.low)
ACT_SIZE     = env.action_space.n
NUM_EPISODES = 4000
LEARN_RATE   = 0.01

# Higher means more exploration.  Basically, an e-greedy method is used,
# but e is decayed over time as a function of episode.
EXPLORE_CONST = 30

def main():
  tf.reset_default_graph()

  # Define the inputs, which consist of the four observations (x, x velocity,
  # pole angle, pole angle velocity).
  x = tf.placeholder(shape=[1, OBS_SIZE], dtype=tf.float32)

  # These are the weights which are what will be trained.
  #W  = tf.Variable(tf.random_uniform([OBS_SIZE, ACT_SIZE]))
  #W  = tf.Variable(tf.random_uniform([OBS_SIZE, 10]))
  W  = tf.Variable(tf.random_uniform([OBS_SIZE, 1]))
  #H  = tf.nn.relu(tf.matmul(x, W))
  H  = tf.nn.softmax(tf.matmul(x, W))
  W2 = tf.Variable(tf.random_uniform([1, ACT_SIZE]))

  # The action prediction is the maximum output (there are two, one for left,
  # one for right).
  #Qout       = tf.matmul(x, W)
  Qout       = tf.matmul(H, W2)
  actPredict = tf.argmax(Qout, 1)

  # Sum of squares difference is used to find the loss and descend toward
  # a solution.
  nextQ       = tf.placeholder(shape=[1, ACT_SIZE], dtype=tf.float32)
  loss        = tf.reduce_sum(tf.square(nextQ - Qout))
  #loss        = tf.square(nextQ - Qout)
  trainer     = tf.train.GradientDescentOptimizer(learning_rate=LEARN_RATE)
  updateModel = trainer.minimize(loss)

  # Graph defined -- the session moves it to the device for execution.
  init = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init)

    for episode in range(NUM_EPISODES):
      lastObs = env.reset()
      done    = False
      t       = 0 # t is the timestep.

      while not done:
        t += 1
        env.render()

        # Run the inputs through the network to predict an action and get the Q
        # table.  (Both action and Q are 2D arrays with a single row, per the
        # graph defintion above.)
        action, Q = sess.run([actPredict, Qout], feed_dict={x: [lastObs]})
        #print('Q')
        #print(Q)

        # Choose a random action occasionally so that new paths are explored.
        if random.random() < math.pow(2, -episode / EXPLORE_CONST):
          action[0] = env.action_space.sample()

        # Apply the action.
        #print('Applying action {}.'.format(action[0]))
        newObs, reward, done, _ = env.step(action[0])
        #print('t, newobs, reward, done')
        #print(t, newObs, reward, done)

        # Now get Q' by feeding the new observation through the network.
        newQ = sess.run(Qout, feed_dict={x: [newObs]})
        #print('New Q')
        #print(newQ)

        # Of the Q values, this one is the max (e.g. the best estimated reward).
        bestReward = np.max(newQ)
        #print('Best Q')
        #print(bestQ)

        # Update the Q table for the last action.  This is the new target.
        #gamma = 1 / t
        gamma = .99
        #gamma = .01
        oldEst = Q[0, action[0]]
        Q[0, action[0]] = reward + gamma * bestReward
        #Q[0, action[0]] = oldEst + gamma * (reward + bestReward - oldEst)
        #print('Target Q')
        #print(Q)

        # Update the weights (train) using the updated Q target.
        t1 = sess.run(updateModel, feed_dict={x: [lastObs], nextQ: Q})
        #print(t2)

        lastObs = newObs

        #time.sleep(.02)

      print('Episode {} went for {} timesteps.'.format(episode+1, t))

if __name__ == "__main__":
  main()

