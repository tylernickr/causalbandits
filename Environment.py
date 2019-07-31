from torch import tensor
from pyro import sample
import pyro.distributions as dist


class Environment(object):

    def __init__(self, env_probabilities):
        self.env_probabilities = env_probabilities

    def observe(self):
        env = {}
        for key, value in self.env_probabilities.items():
            env[key] = sample(key, dist.Categorical(tensor([1 - value, value])))
        return env
