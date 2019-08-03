import numpy as np
from pyro import sample
import pyro.distributions as dist
from torch import tensor


class BaseSampler(object):

    @staticmethod
    def _get_init_array(n):
        return np.zeros(n)

    def __init__(self, bandits, environment):
        self.bandits = bandits
        self.environment = environment
        self.wins = self.__class__._get_init_array(len(self.bandits))
        self.trials = self.__class__._get_init_array(len(self.bandits))
        self.N = 0
        self.choices = []
        self.current_scores = []
        self.observed_envs = []

    def initialize(self):
        self.wins = self.__class__._get_init_array(len(self.bandits))
        self.trials = self.__class__._get_init_array(len(self.bandits))
        self.N = 0
        self.choices = []
        self.current_scores = []
        self.observed_envs = []

    def sample_bandits(self, n=1):
        for k in range(n):
            self._pull_bandit()

    def _pull_bandit(self):
        # This method will do nothing in the base class
        # Implement it in actual implementations of BaseSampler
        pass


class StandardThomspon(BaseSampler):
    """
    Standard Thompson Sampling.

    parameters:
    bandits: a bandit class with .pull method

    methods:
    sample_bandits(n): sample and train on n pulls.
    initialize():  reset these variables to their blank states, if we want to re-run the algorithm
    
    attributes:
    N: the cumulative number of samples
    choices: the current choices as an (N,) array
    current_scores: the current scores as an (N,) array
    """
    def _pull_bandit(self):
        # get env for this round
        observed_env = self.environment.observe()

        # sample from bandit probabilities priors and choose bandit that maximizes its Beta distr.
        choice = np.argmax(
            [sample('arm' + str(i), dist.Beta(1 + self.wins[i], 1 + self.trials[i] - self.wins[i])) for i in
             range(len(self.bandits))])

        # sample the chosen bandit
        result = self.bandits.pull(choice, observed_env)

        # update priors and score
        self.wins[choice] += result
        self.trials[choice] += 1
        self.current_scores.append(result)
        self.N += 1
        self.choices.append(choice)
        self.observed_envs.append(observed_env)


class CausalThomspon(BaseSampler):

    @staticmethod
    def _get_init_array(n):
        return np.array([
            np.array([np.zeros(n), np.zeros(n)]),
            np.array([np.zeros(n), np.zeros(n)])
        ])

    def cond_prob_y(self, *args):
        # Calculate the prob of Y=1 given the conditional: X=x (Drunk = d, and Blinking = b)
        if len(args) == 1:  # meanining only condition on x
            x = args[0]
            y_vals = [self.current_scores[i] for i, x_val in enumerate(self.choices) if x_val == x]
        else:
            x, d, b = args
            y_vals = [self.current_scores[i] for i, (x_val, env) in enumerate(zip(self.choices, self.observed_envs))
                      if x_val == x and env['drunk'] == d and env['blinking'] == b]
        if len(y_vals) == 0:
            return 0
        else:
            prob = sum(y_vals) / len(y_vals)
            return prob

    def cond_prob_db(self, d, b, x):
        # Calculate the prob of Drunk=d & Blink=b given the conditional: X=x
        no_events = np.sum([1 for i, (x_val, env) in enumerate(zip(self.choices, self.observed_envs))
                            if x_val == x and env['drunk'] == d and env['blinking'] == b])
        no_outcomes = np.sum([1 for i, x_val in enumerate(self.choices) if x_val == x])

        if no_outcomes == 0:
            return 0
        else:
            prob = no_events / no_outcomes
            return prob

    def _pull_bandit(self):
        # get env for this round
        observed_env = self.environment.observe()
        drunk = observed_env['drunk']
        blinking = observed_env['blinking']

        # Get the intuition for this trial:
        # Based on Bareinboim et. al., this is just the xor function of drunk and blinking
        intuition = tensor(int(bool(drunk) ^ bool(blinking)))  # xor(drunk, blinking)

        # Estimate the payout for the counter-intuition: E(Y_(X=x')|X=x)
        counter_intuition = abs(intuition - 1)
        Q1 = np.sum([self.cond_prob_y(counter_intuition, drunk_val, blink_val)
                     * self.cond_prob_db(drunk_val, blink_val, counter_intuition)
                     for drunk_val in [0, 1] for blink_val in [0, 1]])

        # Estimate the payout for the intuition (posterior predictive): P(y|X=x)
        Q2 = self.cond_prob_y(intuition)

        w = [1, 1]  # initialize weights (per the paper)
        bias = 1 - abs(Q1 - Q2)  # weighting strength (per the paper)

        if Q1 > Q2:
            w[intuition] = bias  # per the paper
        else:
            w[counter_intuition] = bias

        # Get the #successes and # failures for each machine given the intuition:
        # Since we store the successes as wins[drunk][blinking] I want to get the possible drunkess and
        # blinkness that would yield our intuition (2 posibilities since we are doing inverse of xor)
        env_given_intuition = [[drunk, blinking],
                               [abs(drunk - 1), abs(blinking - 1)]]

        # Thus, env_given_intuitio[k] corresponds to the drunk&blink values
        # that yield that intuition
        wins = sum([self.wins[drunk][blinking] for drunk, blinking in env_given_intuition])
        trials = sum([self.trials[drunk][blinking] for drunk, blinking in env_given_intuition])
        alpha = 1 + wins
        beta = 1 + trials - wins

        # Choose arm:
        choice = np.argmax([sample('arm1', dist.Beta(alpha[0], beta[0])) * w[0],
                            sample('arm2', dist.Beta(alpha[1], beta[1])) * w[1]])

        # sample the chosen bandit
        result = self.bandits.pull(choice, observed_env)

        # update priors and score
        self.wins[drunk][blinking][choice] += result
        self.trials[drunk][blinking][choice] += 1
        self.current_scores.append(result)
        self.N += 1
        self.choices.append(choice)
        self.observed_envs.append(observed_env)




