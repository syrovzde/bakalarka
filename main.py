import numpy as np
import pandas

import Bettor
import method
from KL import KL
from generating_method import GeneratingMethod
from regression import Regression


def getResults(bets):
    winner = bets['HSC'] != bets['ASC']
    home = bets['HSC'] > bets['ASC']
    results = home.astype(int) + winner.astype(int)
    return results


def random(data, probabilities, fair=True, closed=True, margin='basic', odds=None):
    generating = GeneratingMethod(data=data, bet_choices=3)
    bets = generating.run(probabilities)
    if odds is None:
        odds = find_correct_odds(fair, closed, margin, generating)
    return generating.evaluate(bets=bets, odds=odds)


def randomSingle(data, number, fair=True, closed=True, margin='basic', odds=None):
    generating = GeneratingMethod(data=data, bet_choices=3)
    bets = number * np.ones(generating.data['results'].count(), dtype=np.int)
    if odds is None:
        odds = find_correct_odds(fair, closed, margin, generating)
    return generating.evaluate(bets, odds=odds)


def find_correct_odds(fair, closed, margin, generating):
    if fair:
        if closed:
            odds = generating.find_fair_odds(method=margin, odds_columns=method.CLOSED)
        else:
            odds = generating.find_fair_odds(method=margin, odds_columns=method.START)
    else:
        if closed:
            odds = generating.find_odds(odds_columns=method.CLOSED)
        else:
            odds = generating.find_odds(odds_columns=method.START)
    return odds


def betFavourite(data, fair=True, closed=True, margin="basic", odds=None):
    generating = method.Method(data, bet_choices=3)
    if odds is None:
        odds = find_correct_odds(fair, closed, margin, generating)
    favourite = np.argmin(odds, axis=1)
    return generating.evaluate(bets=favourite, odds=odds)


def betUnderdog(data, fair=True, closed=True, margin="basic", odds=None):
    generating = method.Method(data, bet_choices=3)
    if odds is None:
        odds = find_correct_odds(fair, closed, margin, generating)
    underdog = np.argmax(odds, axis=1)
    return generating.evaluate(bets=underdog, odds=odds)


def bettorWithBankroll(data, fraction, bankroll, iterations, threshold, odds):
    bettor = Bettor.Bettor(data=data, fraction=fraction, bankroll=bankroll, iterations=iterations, threshold=threshold)
    iteration = bettor.bet(odds)
    return bettor.bankroll, iteration


def bettorsWithBankroll(data, count, fraction, bankroll, iterations, threshold, odds):
    bankrolls = np.zeros(count)
    iteration = np.zeros(count)
    for i in range(count):
        bankrolls[i], iteration[i] = bettorWithBankroll(data, fraction=fraction, bankroll=bankroll,
                                                        iterations=iterations, threshold=threshold, odds=odds)
    return bankrolls, iteration


def devideBet(data, margin="basic", prob=None):
    generating = method.Method(data, bet_choices=3)
    odds = generating.find_fair_odds(margin)
    if prob is None:
        prob = 1 / 3 * np.ones(odds.shape)  # prob = generating.get_probabilities()  # prob = 1/odds
    n_bet = data['results'].count()
    np_results = pandas.DataFrame.to_numpy(data['results'])
    eval = np.zeros(prob.shape)
    for i in range(n_bet):
        eval[i, np_results[i]] = odds[i, np_results[i]] * prob[i, np_results[i]]
    print((np.sum(eval) - n_bet) / n_bet)
    return eval


def kl_divergence(data, closed=True, margin='basic'):
    kl = KL(data=data, bet_choices=3)
    return kl.run(closed=closed, margin=margin)


def regression(data, odds):
    linear = Regression(data=data, bet_choices=3)
    lin = linear.run(model="Linear", odds=odds)
    log = linear.run(model="Logistic", odds=odds)
    return lin, log
