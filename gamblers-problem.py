'''
A gambler has the opportunity to make bets on the outcomes of a sequence of
coin flips. If the coin comes up heads, he wins as many dollars as he has
staked on that flip; if it is tails, he loses his stake. The game ends when the
gambler wins by reaching his goal of $100, or loses by running out of money. On
each flip, the gambler must decide what portion of his capital to stake, in
integer numbers of dollars. This problem can be formulated as an undiscounted,
episodic, finite MDP. The state is the gambler’s capital, s ∈ {1, 2, ..., 99},
and the actions are stakes, a ∈ {0, 1, ..., min(s, 100 − s)}. The reward is
zero on all transitions except those on which the gambler reaches his goal,
when it is +1. The state-value function then gives the probability of winning
from each state. A policy is a mapping from levels of capital to stakes. The
optimal policy maximizes the probability of reaching the goal. Let p_h denote
the probability of the coin coming up heads. If p_h is known, then the entire
problem is known and it can be solved, for instance, by value iteration.

Excercise 4.9: Implement value iteration for the gambler’s problem and solve it
for ph = 0.25 and ph = 0.55. In programming, you may find it convenient to
introduce two dummy states corresponding to termination with capital of 0 and
100, giving them values of 0 and 1 respectively. Show your results graphically,
as in Figure 4.6. Are your results stable as θ → 0?
'''
import numpy as np

NUM_STATES = 100 # 100 states, corresponding to the gambler's capital.
p_h        = .4  # Probability of heads coming up.

# The value at each state is initialized to zero, except at the dummy terminal
# state 0, which is a loss, and (100) which wins.
values = np.zeros(NUM_STATES + 1)
values[NUM_STATES] = 1

# This will hold the optimal policy once the value function is done.
policy = np.zeros(NUM_STATES + 1)

delta = 1
EPSILON = .001

while delta > EPSILON:
  delta = 0

  for s in range(1, NUM_STATES):
    temp = values[s]

    # The actions are the stakes (how much the gambler can bet) which depends
    # on the current state (the amount the gambler has).  She can bet up to as
    # much as she has, so long as winning would not exceed the goal (100).
    actions = np.arange(min(s, NUM_STATES - s) + 1)
    actVals = []

    for a in actions:
      # p_h probability that the gambler wins (heads) and gets "a" more dollars.
      win  = p_h * values[s + a]
      # (1 - p_h) probability that the gambler loses and loses "a" dollars.
      lose = (1 - p_h) * values[s - a]
      actVals.append(win + lose)

    # Update the value associated with this state using the best action.
    values[s] = max(actVals)
    delta = max(delta, abs(temp - values[s]))

    #print(values)
    #print(delta)

# Print plot data for the value function.
#print("Value function.")
#print("Capital,Value")

#for s in range(NUM_STATES + 1):
#  print("{},{}".format(s,values[s]))

# Evaluate the policy.
for s in range(1, NUM_STATES):
  actions = np.arange(min(s, NUM_STATES - s) + 1)
  actVals = []

  for a in actions:
    actVals.append(p_h * values[s + a] + (1 - p_h) * values[s - a])

  #policy[s] = np.argmax(actVals)

  maxInd = 0
  maxV = -1
  for i in range(len(actVals)):
    #if actVals[i] >= maxV:
    if actVals[i] >= maxV:
      maxV = actVals[i]
      maxInd = i

  policy[s] = maxInd
print(policy)

#print("Policy")
print("Capital,Policy")
for s in range(NUM_STATES + 1):
  print("{},{}".format(s,policy[s]))

