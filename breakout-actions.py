import gym
import numpy as np
import time

env = gym.make('Breakout-ram-v0')

env.reset()

#for _ in range(100):
#  env.step(env.action_space.sample())

print("starting")

for i in range(100):
  env.render()
  #env.step(0) # NOOP.
  #env.step(1) # Fire.
  #env.step(2) # Right.
  #env.step(3) # Left
  time.sleep(.5)
