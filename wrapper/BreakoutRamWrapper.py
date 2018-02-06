import gym

class BreakoutRamWrapper:

  '''
   ' Init.
  '''
  def __init__(self):
    self.env = gym.make('Breakout-ram-v0')
    self.action_space = self.env.action_space
    self.observation_space = self.env.observation_space

  def reset(self):
    return self.env.reset()

  def render(self):
    return self.env.render()

  def step(self, action):
    new_obs, reward, done, info = self.env.step(action)

    if info['ale.lives'] != 5:
      done   = True
      reward = -1

    return new_obs, reward, done, info

