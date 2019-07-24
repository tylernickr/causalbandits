from torch import tensor
from pyro import sample, distributions


class SimpleSimulator(object):

    def __init__(self, reward_probabilities):
        self.reward_probabilities = reward_probabilities

    def pull(self, bandit_index):
        reward_probability = self.reward_probabilities[bandit_index]
        reward_tensor = tensor([1 - reward_probability, reward_probability])
        reward = sample('arm' + str(bandit_index), distributions.Categorical(reward_tensor))
        return reward


if __name__ == '__main__':
    probs = [.33, .33, .8]
    simulator = SimpleSimulator(probs)

    for arm in range(0, len(probs)):
        arm_results = []
        for trial in range(0, 100):
            arm_results.append(int(simulator.pull(arm)))
        print("Bandit " + str(arm) + ": " + str(sum(arm_results) / len(arm_results)))
