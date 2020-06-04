import numpy as np
import random
from windy_gridworld import WindyGridworldEnv

# The number of episodes to run.
NUM_EPISODES = 100000

# For e-greedy (chance of choosing a random action).  Epsilon decays over time.
EPSILON_START = 0.5
EPSILON_END = 0

# Learning rate.
ALPHA = 0.5

# Discount factor.
GAMMA = 1

# Number of rows and columns in the windy gridworld.
NUM_ROWS = 7
NUM_COLS = 10

'''
 ' Given start and stop points and a time period, get a linear
 ' annealing rate.
'''
def get_annealing_rate(start, stop, time):
  return (stop - start) / time

'''
 ' Given an annealing rate, a starting point, and a time, get a leanearly
 ' annealed value.
'''
def get_annealed_value(rate, start, time):
  return rate * time + start

def policy(env, q, state, episode):
  anneal_rate = get_annealing_rate(EPSILON_START, EPSILON_END, NUM_EPISODES)
  epsilon = get_annealed_value(anneal_rate, EPSILON_START, episode)

  if random.random() < epsilon:
    # Random action (explore).
    return env.action_space.sample()
  else:
    # Greedy action (index of the action with the maximum value).
    return np.argmax(q[state])

def print_policy(q):
  actions = ('U', 'R', 'D', 'L')
  policy_grid = np.empty(NUM_ROWS * NUM_COLS, 'S')

  for state in range(NUM_ROWS * NUM_COLS):
    policy_grid[state] = actions[np.argmax(q[state])]

  print(np.reshape(policy_grid, (NUM_ROWS, NUM_COLS)))

def main():
  env = WindyGridworldEnv()
  t = 0

  # State-action function (table).  70 states with 4 actions (up, right, down,
  # left) per state, so the shape is (70, 4).
  q = np.zeros((NUM_ROWS * NUM_COLS, env.action_space.n))

  for episode in range(NUM_EPISODES):
    state = env.reset()
    action = policy(env, q, state, episode)

    #env.render()
    done = False

    while not done:
      state_prime, reward, done, _ = env.step(action)
      action_prime = policy(env, q, state_prime, episode)

      #print("S: {} A: {} R {} S': {} A': {} Done: {}".format(
      #  state, action, reward, state_prime, action_prime, done))
      #env.render()

      q[(state, action)] += ALPHA * (reward + GAMMA * q[(state_prime, action_prime)] - q[(state, action)])
      #q[(state, action)] += ALPHA * (reward + GAMMA * np.amax(q[state_prime]) - q[(state, action)])
      #print_policy(q)

      state = state_prime
      action = action_prime
      t += 1

    print_policy(q)
    print("Episode: {} Timestep: {}".format(episode + 1, t))

if __name__ == "__main__":
  main()
