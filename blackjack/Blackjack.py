import numpy as np
from random import randint;
from enum import IntEnum

class Blackjack:
  STAY = 0
  HIT  = 1

  def __init__(self):
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

  # Reset the game.
  def reset(self):
    self.dealerCards = [self.getRandomCard(), self.getRandomCard()]
    self.playerCards = [self.getRandomCard(), self.getRandomCard()]

    # Player gets cards until the sum is at least 12 (below this score the
    # player should always hit because it's not possible to bust).
    while self.getPlayerScore() < 12:
      self.hitPlayer()

  # Sum up the score of cards.  If countUsableAce is False, just the sum of the
  # cards, not accounting for usable aces, otherwise add 10 if the cards have a
  # usable ace.
  def _getScore(self, cards, countUsableAce=True):
    score = 0

    for card in cards:
      score += card

    if countUsableAce and self.playerHasUsableAce():
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

