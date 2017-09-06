from random import randint;

class Blackjack:
  STAY = 0
  HIT  = 1

  def __init__(self):
    # Becomes True when the game is over.
    self.done = False

    # A deck of cards.
    self.deck = []

    # 1 - 9, 4 each.
    for i in range(1, 10):
      for j in range(4):
        self.deck.append(i)

    # 10, jack, queen, king, all worth 10, 4 of each card.
    for i in range(16):
      self.deck.append(10)

    # The dealer's cards.
    self.dealerCards = []

    # The player's cards.
    self.playerCards = []

    self.reset()

  # Get a random card from the deck uniformly, with replacement.
  def getRandomCard(self):
    return self.deck[randint(0, len(self.deck) - 1)]

  # Reset the game and return the initial state (see step()).
  def reset(self):
    self.done        = False
    self.dealerCards = [self.getRandomCard(), self.getRandomCard()]
    self.playerCards = [self.getRandomCard(), self.getRandomCard()]

    # Player gets cards until the sum is at least 12 (below this score the
    # player should always hit because it's not possible to bust).
    while self.getPlayerScore() < 12:
      self.hitPlayer()

    return [
      self.getPlayerScore(),
      self.getDealerVisibleScore(),
      1 if self.playerHasUsableAce() else 0]

  # Sum up the score of cards.  If countUsableAce is False, just the sum of the
  # cards, not accounting for usable aces, otherwise add 10 if the cards have a
  # usable ace.
  def _getScore(self, cards, countUsableAce=True):
    score = 0

    for card in cards:
      score += card

    if countUsableAce and self._hasUsableAce(cards):
      score += 10

    return score

  # Check if the cards has a usable ace.
  def _hasUsableAce(self, cards):
    # Player has at least one ace and playing it as 11 wouldn't cause a bust.
    return 1 in cards and self._getScore(cards, False) + 10 <= 21

  # Get the player's score.  If countUsableAce is False, just the sum of the
  # cards, not accounting for usable aces, otherwise add 10 if the player has a
  # usable ace.
  def getPlayerScore(self, countUsableAce=True):
    return self._getScore(self.playerCards, countUsableAce)

  # Get the dealer's score.
  def getDealerScore(self, countUsableAce=True):
    return self._getScore(self.dealerCards, countUsableAce)

  # Check if the player has a usable ace.
  def playerHasUsableAce(self):
    return self._hasUsableAce(self.playerCards)

  # Check if the dealer has a usable ace.
  def dealerHasUsableAce(self):
    return self._hasUsableAce(self.dealerCards)

  # Give the player another card.
  def hitPlayer(self):
    self.playerCards.append(self.getRandomCard())

  # Give the dealer another card.
  def hitDealer(self):
    self.dealerCards.append(self.getRandomCard())

  # Get the player's cards.
  def getPlayerCards(self):
    return self.playerCards

  # Get the dealer's cards.
  def getDealerCards(self):
    return self.dealerCards

  # Get the dealer's visible card value, 1-10 (i.e. excluding the first card).
  def getDealerVisibleScore(self):
    return self.dealerCards[1]

  # Check if the player is bust.
  def playerIsBust(self):
    return self.getPlayerScore() > 21

  # Check if the dealer is bust.
  def dealerIsBust(self):
    return self.getDealerScore() > 21

  # Step the environment (hit or stay) and return the new state, the reward (if
  # any), and a flag indicating whether or not the game is done.  The state is
  # a numpy array containing the player score, the dealer's one visible card,
  # and whether or not the player has an ace (0 or 1).
  def step(self, action):
    if self.done:
      raise RuntimeException('step() called but the game has ended.')

    if action == Blackjack.HIT:
      self.hitPlayer()

      # If the player went bust, the game is over and the player loses.
      if self.playerIsBust():
        self.done = True
        reward    = -1
      else:
        # No reward yet because the game isn't done.
        reward = 0
    else:
      # Player decided to stay.
      self.done = True

      # The dealer sticks if his score is 17 or more.
      while self.getDealerScore() < 17:
        self.hitDealer()

      # If the dealer went bust then the game is over and the player wins.
      if self.dealerIsBust():
        reward = 1
      else:
        # Reward is based on the scores.
        pScore = self.getPlayerScore()
        dScore = self.getDealerScore()

        if pScore == dScore:
          reward = 0
        elif pScore < dScore:
          reward = -1
        else:
          reward = 1

    return [
      self.getPlayerScore(),
      self.getDealerVisibleScore(),
      1 if self.playerHasUsableAce() else 0], reward, self.done

