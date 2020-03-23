import numpy as np

from method import Method


class GeneratingMethod(Method):
    n_random = 100

    def __int__(self, data, bet_choices, n_random, strategy):
        super().__init__(data=data, bet_choices=bet_choices)
        self.n_random = n_random
        self.strategy = strategy

    # distribution instead of unit bet
    def run(self, prob=None):
        bets = np.ones((self.length, self.n_random), dtype=int)
        for i in range(self.length):
            if prob is None:
                bets[i] = np.random.choice(self.bet_choices, size=self.n_random)
            else:
                bets[i] = np.random.choice(self.bet_choices, size=self.n_random, p=prob[i])
        return bets.T
