import numpy as np

import method as method


class KL(method.Method):
    """
    Class to calculate KL-divergence(distance) of stretegy from random one
    """

    def __init__(self, data, bet_choices):
        super().__init__(data=data, bet_choices=bet_choices)

    def run(self, margin='basic',odds=None,results=None):
        if odds is None:
            odds = self.find_fair_odds(method=margin)
        if results is None:
            results = self.data['results']
        n_bets = results.count()
        outcomes = np.zeros(n_bets)
        for i in range(n_bets):
            outcomes[i] = odds[i, results.iloc[i]]
        kl = np.sum(np.log(outcomes)) / n_bets
        eff = 1 - np.minimum(kl / np.log(self.bet_choices), 1)
        return eff
