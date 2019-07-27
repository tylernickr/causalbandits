import numpy as np
from pyro import sample
import pyro.distributions as dist


class StandardThomspon(object):
  """
  Standard Thompson Sampling.

  parameters:
    bandits: a bandit class with .pull method

  methods:
    sample_bandits(n): sample and train on n pulls.
    initialize():  reset these variables to their blank states, if we want to re-run the algorithm
    
  attributes:
    N: the cumulative number of samples
    choices: the historical choices as an (N,) array
    bb_score: the historical score as an (N,) array
  """

  def __init__(self, bandits):

    self.bandits = bandits
    n_bandits = len(self.bandits)
    self.wins = np.zeros(n_bandits)
    self.trials = np.zeros(n_bandits)
    self.N = 0
    self.choices = []
    self.bb_score = []

  def sample_bandits(self, n=1):

    bb_score = np.zeros(n)
    choices = np.zeros(n)
    
    for k in range(n):
      # sample from bandit probabilities priors and choose bandit that maximizes its Beta distr.
      choice = np.argmax([sample('arm'+str(i), dist.Beta(1+self.wins[i],1+self.trials[i]-self.wins[i])) for i in range(len(self.bandits))])

      # sample the chosen bandit
      result = self.bandits.pull(choice)

      # update priors and score
      self.wins[choice] += result
      self.trials[choice] += 1
      bb_score[k] = result
      self.N += 1
      choices[k] = choice

    self.bb_score = np.r_[self.bb_score, bb_score]   # Note that each time we pull more arms, we append the results
    self.choices = np.r_[self.choices, choices]      # If we want to re-run the algo, the results will be appended

    return

  def initialize(self):  # if we want to re-run the algorithm from scratch
        
    n_bandits = len(self.bandits)
    self.wins = np.zeros(n_bandits)
    self.trials = np.zeros(n_bandits)
    self.N = 0
    self.choices = []
    self.bb_score = []
    
    return



class CausalThomspon(object):
  """
  Causal Thompson Sampling.

  Need to finish....
  
  """
  def __init__(self, bandits):

    self.bandits = bandits
    
    

  