import numpy as np
import pandas

import method


class Bettor:

    def __init__(self, data, fraction, bankroll, iterations, threshold):
        self.data = data
        self.fraction = fraction
        self.bankroll = bankroll
        self.iterations = iterations
        self.threshold = threshold

    def bet(self, odds):
        generation = method.Method(self.data, bet_choices=3)
        np_results = pandas.DataFrame.to_numpy(self.data['results'])
        count = 0
        for i in range(self.iterations):
            count += 1
            my_bet = np.random.choice(generation.bet_choices)
            bet_amount = self.fraction * self.bankroll
            self.bankroll = (1 - self.fraction) * self.bankroll
            if my_bet == np_results[i]:
                self.bankroll = self.bankroll + bet_amount * odds[i, my_bet]
            if self.bankroll < self.threshold:
                self.bankroll = 0
                break
        return count
