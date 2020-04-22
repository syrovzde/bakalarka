import numpy as np

import method as method


class KL(method.Method):
    """
    Class to calculate KL-divergence(distance) of stretegy from random one
    """

    def __init__(self, data, bet_choices):
        super().__init__(data=data, bet_choices=bet_choices)

    def run(self, margin='basic'):
        odds = self.find_fair_odds(method=margin)
        n_bets = self.data['results'].count()
        outcomes = np.zeros(n_bets)
        for i in range(n_bets):
            outcomes[i] = odds[i, self.data['results'].iloc[i]]
        kl = np.sum(np.log(outcomes)) / n_bets
        eff = 1 - np.minimum(kl / np.log(self.bet_choices), 1)
        return eff
