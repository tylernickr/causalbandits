import numpy as np
from pyro import sample
from pyro import do
import pyro.distributions as dist
from time import time

DRUNK = 'drunk'
BLINKING = 'blinking'


class BaseSampler(object):
    """
    A base class with standard methods used by more complex samplers.

    parameters:
    bandits: a bandit class with .pull method
    environment: an environment that can be .observe for variables
    """

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

    def update_parameters(self, choice, result, observed_env):
        self._update_wins_trials(choice, result, observed_env)
        self.current_scores.append(result)
        self.N += 1
        self.choices.append(choice)
        self.observed_envs.append(observed_env)

    def _update_wins_trials(self, choice, result, observed_env):
        self.wins[choice] += result
        self.trials[choice] += 1

    def sample_bandits(self, n=1):
        for k in range(n):
            self._pull_bandit()

    def _pull_bandit(self):
        observed_env = self.environment.observe()
        algo_choice = self._select_arm(observed_env)
        model = self.bandits.gambler_model
        intervention_model = do(model, {'arm':algo_choice})
        result = intervention_model(observed_env)
        self.update_parameters(algo_choice, result, observed_env)

    def _select_arm(self, observed_env):
        return 0
    

class StandardThomspon(BaseSampler):

    def _select_arm(self, observed_env):
        choice = np.argmax([
                sample('arm' + str(i), dist.Beta(1 + self.wins[i], 1 + self.trials[i] - self.wins[i]))
                for i in range(len(self.bandits))
            ])
        return choice


class CausalThomspon(BaseSampler):

    @staticmethod
    def _get_init_array(n):
        return np.array([
            np.array([np.zeros(n), np.zeros(n)]),
            np.array([np.zeros(n), np.zeros(n)])
        ])

    def _update_wins_trials(self, choice, result, observed_env):
        drunk = observed_env[DRUNK]
        blinking = observed_env[BLINKING]
        self.wins[drunk][blinking][choice] += result
        self.trials[drunk][blinking][choice] += 1

    # Calculate P(Y=1 | X=x, Drunk = d, Blinking = b)
    def _cond_prob_y(self, x, d=-1, b=-1):
        if d == b == -1:  # meanining only condition on x
            y_vals = [self.current_scores[i] for i, x_val in enumerate(self.choices) if x_val == x]
        else:
            y_vals = [self.current_scores[i] for i, (x_val, env) in enumerate(zip(self.choices, self.observed_envs))
                      if x_val == x and env[DRUNK] == d and env[BLINKING] == b]

        if len(y_vals) == 0:
            return 0
        else:
            return sum(y_vals) / len(y_vals)

    # Calculate P(Drunk=d & Blink=b | X=x)
    def _cond_prob_db(self, d, b, x):
        no_events = np.sum([1 for i, (x_val, env) in enumerate(zip(self.choices, self.observed_envs))
                            if x_val == x and env[DRUNK] == d and env[BLINKING] == b])
        no_outcomes = np.sum([1 for i, x_val in enumerate(self.choices) if x_val == x])

        if no_outcomes == 0:
            return 0
        else:
            return no_events / no_outcomes

    def _select_arm(self, observed_env):
        # Get environment variables that we care about
        drunk = observed_env[DRUNK]
        blinking = observed_env[BLINKING]

        # Get the intuition for this trial:
        # Based on Bareinboim et. al., this is just the xor function of drunk and blinking
        intuition = int(bool(drunk) ^ bool(blinking))  # xor(drunk, blinking)

        # Estimate the payout for the counter-intuition: E(Y_(X=x')|X=x)
        counter_intuition = abs(intuition - 1)
        Q1 = np.sum([self._cond_prob_y(counter_intuition, drunk_val, blink_val)
                     * self._cond_prob_db(drunk_val, blink_val, counter_intuition)
                     for drunk_val in [0, 1] for blink_val in [0, 1]])

        # Estimate the payout for the intuition (posterior predictive): P(y|X=x)
        Q2 = self._cond_prob_y(intuition)

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

        # Thus, env_given_intuition[k] corresponds to the drunk&blink values
        # that yield that intuition
        wins = sum([self.wins[drunk][blinking] for drunk, blinking in env_given_intuition])
        trials = sum([self.trials[drunk][blinking] for drunk, blinking in env_given_intuition])
        alpha = 1 + wins
        beta = 1 + trials - wins

        # Choose arm:
        choice = np.argmax([sample('arm1', dist.Beta(alpha[0], beta[0])) * w[0],
                            sample('arm2', dist.Beta(alpha[1], beta[1])) * w[1]])

        return choice



