Blackjack is a popular casino card game. The object is to
obtain cards the sum of whose numerical values is as great as possible without
exceeding 21. All face cards count as 10, and the ace can count either as 1 or
as 11. We consider the version in which each player competes independently
against the dealer. The game begins with two cards dealt to both dealer and
player. One of the dealer’s cards is faceup and the other is facedown. If the
player has 21 immediately (an ace and a 10-card), it is called a natural. He
then wins unless the dealer also has a natural, in which case the game is a
draw. If the player does not have a natural, then he can request additional
cards, one by one (hits), until he either stops (sticks) or exceeds 21 (goes bust).
If he goes bust, he loses; if he sticks, then it becomes the dealer’s turn. The
dealer hits or sticks according to a fixed strategy without choice: he sticks on
any sum of 17 or greater, and hits otherwise. If the dealer goes bust, then
the player wins; otherwise, the outcome—win, lose, or draw—is determined
by whose final sum is closer to 21.

Playing blackjack is naturally formulated as an episodic finite MDP. Each
game of blackjack is an episode. Rewards of +1, −1, and 0 are given for
winning, losing, and drawing, respectively. All rewards within a game are
zero, and we do not discount (γ = 1); therefore these terminal rewards are
also the returns. The player’s actions are to hit or to stick. The states depend
on the player’s cards and the dealer’s showing card. We assume that cards
are dealt from an infinite deck (i.e., with replacement) so that there is no
advantage to keeping track of the cards already dealt. If the player holds an
ace that he could count as 11 without going bust, then the ace is said to be
usable. In this case it is always counted as 11 because counting it as 1 would
make the sum 11 or less, in which case there is no decision to be made because,
obviously, the player should always hit. Thus, the player makes decisions on
the basis of three variables: his current sum (12–21), the dealer’s one showing
card (ace–10), and whether or not he holds a usable ace. This makes for a
total of 200 states.

Consider the policy that sticks if the player’s sum is 20 or 21, and other-
wise hits. To find the state-value function for this policy by a Monte Carlo
approach, one simulates many blackjack games using the policy and averages
the returns following each state. Note that in this task the same state never
recurs within one episode, so there is no difference between first-visit and
every-visit MC methods. In this way, we obtained the estimates of the state-
value function shown in Figure 5.2. The estimates for states with a usable ace
are less certain and less regular because these states are less common. In any
event, after 500,000 games the value function is very well approximated.
Although we have complete knowledge of the environment in this task, it
would not be easy to apply DP policy evaluation to compute the value function.
DP methods require the distribution of next events—in particular, they require
the quantities p('0|s, a) and r(s, a, s')—and it is not easy to determine these for
blackjack. For example, suppose the player’s sum is 14 and he chooses to stick.
What is his expected reward as a function of the dealer’s showing card? All of
these expected rewards and transition probabilities must be computed before
DP can be applied, and such computations are often complex and error-prone.
In contrast, generating the sample games required by Monte Carlo methods is
easy. This is the case surprisingly often; the ability of Monte Carlo methods
to work with sample episodes alone can be a significant advantage even when
one has complete knowledge of the environment’s dynamics.
