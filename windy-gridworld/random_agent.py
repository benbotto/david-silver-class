import numpy as np
from windy_gridworld import WindyGridworldEnv

# The number of episodes to run.
NUM_EPISODES = 1

def main():
  env = WindyGridworldEnv()

  for episode in range(NUM_EPISODES):
    last_obs = env.reset()
    done = False

    env.render()

    while not done:
      act = env.action_space.sample()
      new_obs, reward, done, _ = env.step(act)

      print("Last obs: {} New obs: {} Reward: {} Done: {}".format(last_obs, new_obs, reward, done))
      env.render()
      last_obs = new_obs

if __name__ == "__main__":
  main()
