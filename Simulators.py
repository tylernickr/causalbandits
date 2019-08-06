from torch import tensor
from pyro import sample
import pyro.distributions as dist


class BanditSimulator(object):

    def __init__(self, reward_probabilities):
        self.reward_probabilities = reward_probabilities

    def pull(self, bandit_index, environment):
        reward_probability = self.reward_probabilities[bandit_index]
        reward_tensor = tensor([1 - reward_probability, reward_probability])
        reward = sample('arm'+str(bandit_index), dist.Categorical(reward_tensor))
        return reward
            
    def __len__(self):
        return len(self.reward_probabilities)


class ConfoundingBanditSimulator(object):

    def __init__(self, reward_probabilities):
        self.reward_probabilities = reward_probabilities

    def gambler_model(self, environment):  
        drunk = int(environment["drunk"])
        blinking = int(environment["blinking"])
        intuition = int(bool(drunk) ^ bool(blinking))  # xor(drunk, blinking)

        # Select arm based on intuition
        arm = sample('arm',dist.Delta(tensor(intuition)))

        # Get Reward
        reward_probability = self.reward_probabilities[arm][drunk][blinking]
        reward = sample('reward',dist.Bernoulli(reward_probability))

        return reward

    
    def __len__(self):
        return len(self.reward_probabilities)






