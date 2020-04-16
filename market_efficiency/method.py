import numpy as np
import pandas

import utils.odds2probs as odds2probs

AWAYWIN = 2
DRAW = 0
HOMEWIN = 1
YES = 1
NO = 0


class Method:
    """ kind of abstract class to all the methods """

    def __init__(self, data, bet_choices):
        self.data = data
        self.length = len(data)
        self.bet_choices = bet_choices

    def run(self):
        return NotImplementedError

    def plot(self):
        return NotImplementedError

    def compareBets(self, bets):
        results = pandas.Series.to_numpy(self.data['results'], dtype=int)
        result = results == bets
        return result

    def evaluateStrategy(self, bets, odds, results):
        strategies = np.ones(bets.shape)
        if np.ndim(bets) == 1:
            n = 1
            for i in range(strategies.shape[0]):
                strategies[i] = odds[i, bets[i]]
        else:
            n = bets.shape[1]
            for i in range(n):
                strategies[:, i] = odds[i, bets[:, i]]
        strategies = strategies * results - 1
        return strategies, n

    def evaluate(self, bets, odds):
        results = self.compareBets(bets)
        strategies, n = self.evaluateStrategy(bets, odds, results)
        # in case of evaluating multiple strategies at the time
        if n != 1:
            profit = np.mean(strategies, axis=1)
        else:
            profit = strategies
        return profit

    def find_fair_odds(self, method='basic'):
        odds = self.find_odds()
        prob = odds2probs.implied_probabilities(odds, method, normalize=True)
        if np.isnan(prob['probabilities']).any():
            prob['probabilities'] = np.nan_to_num(prob['probabilities'], nan=1 / self.bet_choices)
        return 1 / prob['probabilities']

    def find_odds(self):
        odds = np.ones((self.length, self.bet_choices))
        if self.bet_choices == 2:
            odds[:, YES] = self.data[str(YES)]
            odds[:, NO] = self.data[str(NO)]
        if self.bet_choices == 3:
            odds[:, DRAW] = self.data[str(DRAW)]
            odds[:, HOMEWIN] = self.data[str(HOMEWIN)]
            odds[:, AWAYWIN] = self.data[str(AWAYWIN)]
        if self.bet_choices > 3:
            return NotImplementedError
        return odds

    def get_probabilities(self):
        prob = np.ones((self.length, self.bet_choices))
        """prob[:,DRAW] = self.data['pD']
        prob[:,HOMEWIN] = self.data['pH']
        prob[:,AWAYWIN] = self.data['pA']"""
        prob[:, DRAW] = 0
        prob[:, HOMEWIN] = 0
        prob[:, AWAYWIN] = 1
        return prob

    def getWinnerProbabilites(self, probabilities):
        bets_count = np.shape(probabilities)[0]
        winnerProbabilities = np.ones(bets_count)
        for i in range(bets_count):
            winnerProbabilities[i] = probabilities[i, self.data['results'].iloc[i]]
        return winnerProbabilities
