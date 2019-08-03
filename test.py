from Simulators import ConfoundingBanditSimulator
from Environment import Environment
from Samplers import CausalThomspon

if __name__ == '__main__':
    env_probs = {
        'drunk': .5,
        'blinking': .5
    }
    env = Environment(env_probs)
    bandit_probs = [
        [
            [.1, .5],
            [.4, .2]
        ],
        [
            [.5, .1],
            [.2, .4]
        ]
    ]
    simulator = ConfoundingBanditSimulator(bandit_probs)

    agent = CausalThomspon(simulator, env)
    agent.sample_bandits(1000)
    print(agent.choices)
    print(agent.wins.sum() / agent.trials.sum())

