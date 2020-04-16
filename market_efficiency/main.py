import numpy as np
import pandas

import market_efficiency.Bettor as Bettor
import market_efficiency.method as method
import re
import pandas as pd
from market_efficiency.KL import KL
from market_efficiency.generating_method import GeneratingMethod
from market_efficiency.regression import Regression


def digit_sum(n):
    sum = 0
    for i in n:
        sum += int(i)
    return sum


def bts_results(results):
    pom = results.str.split("-", expand=True)
    tmp = pom[1] == ''
    ASC = pom[1].astype(int)
    HSC = pom[0].astype(int)
    bet_result = np.logical_and(ASC != 0, HSC != 0)
    return bet_result.astype(int)


def tennis_ou_results(results, total):
    sets = results.str.replace("\D+", "", regex=True)
    totals = []
    pom = results.str.findall("[^^].\d")
    for index, value in sets.items():
        pom = digit_sum(value)
        totals.append(pom)
    return pandas.Series(totals)


def ou_results(results, total, sport):
    total = total.str.findall("\d{1,3}\.*\d*").str[0]
    if sport == 'tennis':
        return tennis_ou_results(results, total)
    pom = results.str.split("-", expand=True)
    ASC = pom[1]
    HSC = pom[0]
    match_total = ASC.astype(int) + HSC.astype(int)
    total = total.astype(str).astype(float)
    return (match_total > total).astype(int)


def getResults(results, market, total=None, sport='football'):
    if market == "1x2":
        return x12market_results(results)
    if market == "ou":
        return ou_results(results, total, sport)
    if market == "bts":
        return bts_results(results)


def x12market_results(results):
    pom = results.str.split("-", expand=True)
    HSC = pom[0]
    ASC = pom[1]
    winner = HSC != ASC
    home = HSC < ASC
    results = home.astype(int) + winner.astype(int)
    return results


def random_dict(dict):
    return random(dict['data'], dict['probabilities'], dict['fair'], dict['margin'], dict['odds'], dict['bet_choices'],
                  dict['n_random'])


def random(data, probabilities=None, fair=True, margin='basic', odds=None, bet_choices=3, n_random=100):
    generating = GeneratingMethod(data=data, bet_choices=bet_choices)
    if odds is None:
        odds = find_correct_odds(fair, margin, generating)
    if probabilities is None:
        probabilities = 1 / odds
    generating.n_random = n_random
    bets = generating.run(probabilities)
    return generating.evaluate(bets=bets, odds=odds)


def bet_home_dict(dict):
    return randomSingle(dict['data'], method.HOMEWIN, dict['fair'], dict['margin'], dict['odds'], dict['bet_choices'])


def bet_away_dict(dict):
    if dict['bet_choices'] == 2:
        return None
    return randomSingle(dict['data'], method.AWAYWIN, dict['fair'], dict['margin'], dict['odds'], dict['bet_choices'])


def bet_draw_dict(dict):
    return randomSingle(dict['data'], method.DRAW, dict['fair'], dict['margin'], dict['odds'], dict['bet_choices'])


def random_single_dict(dict):
    return randomSingle(dict['data'], dict['number'], dict['fair'], dict['margin'], dict['odds'], dict['bet_choices'])


def randomSingle(data, number, fair=True, margin='basic', odds=None, bet_choices=3):
    generating = GeneratingMethod(data=data, bet_choices=bet_choices)
    if odds is None:
        odds = find_correct_odds(fair, margin, generating)
    bets = number * np.ones(odds.shape[0], dtype=np.int)
    return generating.evaluate(bets, odds=odds)


def find_correct_odds(fair, margin, generating):
    if fair:
        odds = generating.find_fair_odds(method=margin)
    else:
        odds = generating.find_odds()
    return odds


def bet_favourite_dict(dict):
    return betFavourite(dict['data'], dict['fair'], dict['margin'], dict['odds'], dict['bet_choices'])


def betFavourite(data, fair=True, margin="basic", odds=None, bet_choices=3):
    generating = method.Method(data, bet_choices=bet_choices)
    if odds is None:
        odds = find_correct_odds(fair, margin, generating)
    favourite = np.argmin(odds, axis=1)
    return generating.evaluate(bets=favourite, odds=odds)


def bet_Underdog_dict(dict):
    return betUnderdog(dict['data'], dict['fair'], dict['margin'], dict['odds'], dict['bet_choices'])


def betUnderdog(data, fair=True, margin="basic", odds=None, bet_choices=3):
    generating = method.Method(data, bet_choices=bet_choices)
    if odds is None:
        odds = find_correct_odds(fair, margin, generating)
    underdog = np.argmax(odds, axis=1)
    return generating.evaluate(bets=underdog, odds=odds)


def bettorWithBankroll(data, fraction, bankroll, iterations, threshold, odds):
    bettor = Bettor.Bettor(data=data, fraction=fraction, bankroll=bankroll, iterations=iterations, threshold=threshold)
    iteration = bettor.bet(odds)
    return bettor.bankroll, iteration


def bettors_with_bankroll_dict(dict):
    return bettorsWithBankroll(dict['data'], dict['count'], dict['fraction'], dict['bankroll'], dict['iterations'],
                               dict['threshold'], dict['odds'])

def bettorsWithBankroll(data, count, fraction, bankroll, iterations, threshold, odds):
    bankrolls = np.zeros(count)
    iteration = np.zeros(count)
    for i in range(count):
        bankrolls[i], iteration[i] = bettorWithBankroll(data, fraction=fraction, bankroll=bankroll,
                                                        iterations=iterations, threshold=threshold, odds=odds)
    return bankrolls, iteration


def devide_bet_dict(dict):
    return devide_bet(dict['data'], dict['margin'], dict['prob'], dict['bet_choices'])


def devide_bet(data, margin="basic", prob=None, bet_choices=3):
    generating = method.Method(data, bet_choices=bet_choices)
    odds = generating.find_fair_odds(margin)
    if prob is None:
        prob = 1 / bet_choices * np.ones(
            odds.shape)  # prob = 1 / 3 * np.ones(odds.shape)  # prob = generating.get_probabilities()  # prob = 1/odds
    n_bet = data['results'].count()
    np_results = pandas.DataFrame.to_numpy(data['results'])
    eval = np.zeros(prob.shape)
    for i in range(n_bet):
        eval[i, np_results[i]] = odds[i, np_results[i]] * prob[i, np_results[i]]
    #print(np.max(eval,axis=0))
    print((np.sum(eval) - n_bet) / n_bet)
    return eval


def kl_divergence_dict(dict):
    return kl_divergence(dict['data'], dict['margin'])


def kl_divergence(data, margin='basic', bet_choices=3):
    kl = KL(data=data, bet_choices=bet_choices)
    return kl.run(margin=margin)


def regression_dict(dict):
    return regression(dict['data'], dict['odds'])


def regression(data, bet_choices=3, fair=True, margin='basic'):
    linear = Regression(data=data, bet_choices=bet_choices)
    odds = find_correct_odds(fair=fair, margin=margin, generating=linear)
    lin = linear.run(model="Linear", odds=odds)
    log = linear.run(model="Logistic", odds=odds)
    return lin, log

