import numpy as np
import pandas

import utils.odds2probs as odds2probs

AWAYWIN = 1
DRAW = 0
HOMEWIN = 2
CLOSED = ['Draw_close', 'Away_close', 'Home_close']
START = ['Draw_start', 'Away_start', 'Home_start']
MARKET_1X2 = ['X', '2', '1']
MARKET_OU = ['Over', 'Under']
MARKET_AH = ['2', '1']
MARKET_DC = ['1X', '12', 'X2']
MARKET_BTS = ['YES', 'NO']


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

    def find_fair_odds(self, method='basic', odds_columns=None):
        odds = self.find_odds(odds_columns)
        prob = odds2probs.implied_probabilities(odds, method, normalize=True)
        return 1 / prob['probabilities']

    def find_odds(self, odds_columns):
        odds = np.ones((self.length, self.bet_choices))
        odds[:, DRAW] = self.data[odds_columns[DRAW]]
        odds[:, HOMEWIN] = self.data[odds_columns[HOMEWIN]]
        odds[:, AWAYWIN] = self.data[odds_columns[AWAYWIN]]
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
