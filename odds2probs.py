import numpy as np
from scipy.optimize import brentq as root


# Source in R: https://github.com/opisthokonta/implied/blob/master/R/implied_probabilities.R

def margin_basic(odds: np.ndarray):
    res = implied_probabilities(odds, method="basic")
    return res['margin']


def implied_probs(odds: np.ndarray, method='basic', normalize=True):
    res = implied_probabilities(odds, method, normalize=True)
    return res['probabilities']


def implied_probabilities(odds: np.ndarray, method='basic', normalize=True):
    if (odds < 1).any():
        return False

    # Prepare the list that will be returned.
    result = {'probabilities': -1, 'margin': -1}

    # Some useful quantities
    n_matches = odds.shape[0]
    n_outcomes = odds.shape[1]

    # Inverted odds and margins
    inverted_odds = 1 / odds
    inverted_odds_sum = np.sum(inverted_odds, axis=1)
    inverted_odds_sum = inverted_odds_sum[:, None]
    result['margin'] = (inverted_odds_sum - 1) / (inverted_odds_sum)  # corrected margin definition

    if method == 'basic':
        result['probabilities'] = inverted_odds / inverted_odds_sum

    elif method == 'shin':

        zvalues = np.zeros(n_matches)  # The proportion of insider trading.
        probs = np.zeros((n_matches, n_outcomes))

        for ii in range(n_matches):
            try:
                resroot = root(shin_solvefor, 0, 0.4, args=inverted_odds[ii,])
            except Exception as exc:
                print("Cannot convert odds to probs: " + ",".join(map(str, odds[ii])))
                continue
            zvalues[ii] = resroot
            probs[ii,] = shin_func(zz=resroot, io=inverted_odds[ii,])

        result['probabilities'] = probs
        result['zvalues'] = zvalues

    elif method == 'wpo':
        # Margin Weights Proportional to the Odds.
        # Method from the Wisdom of the Crowds pdf.
        fair_odds = (n_outcomes * odds) / (n_outcomes - (result['margin'] * odds))
        result['probabilities'] = 1 / fair_odds
        result['specific_margins'] = (result['margin'] * fair_odds) / n_outcomes

    elif (method == 'or'):

        odds_ratios = np.zeros(n_matches)
        probs = np.zeros((n_matches, n_outcomes))

        for ii in range(n_matches):
            try:
                resroot = root(or_solvefor, 0.05, 5, args=inverted_odds[ii,])
            except Exception as exc:
                print("Cannot convert odds to probs: " + ",".join(map(str, odds[ii])))
                continue
            odds_ratios[ii] = resroot
            probs[ii,] = or_func(cc=resroot, io=inverted_odds[ii,])

        result['probabilities'] = probs
        result['odds_ratios'] = odds_ratios

    elif method == 'power':

        probs = np.zeros((n_matches, n_outcomes))
        exponents = np.zeros(n_matches)

        for ii in range(n_matches):
            try:
                resroot = root(pwr_solvefor, 0.001, 1, args=inverted_odds[ii,])
            except Exception as exc:
                print("Cannot convert odds to probs: " + ",".join(map(str, odds[ii])))
                continue
            exponents[ii] = resroot
            probs[ii,] = pwr_func(nn=resroot, io=inverted_odds[ii,])

        result['probabilities'] = probs
        result['exponents'] = exponents

    ## do a final normalization to make sure the probabilites
    ## sum to 1 without rounding errors.
    if normalize:
        norm = np.sum(result['probabilities'], axis=1)[:, None]
        np.warnings.filterwarnings('ignore')  # division by zero (negative margin odds)
        try:
            result['probabilities'] = result['probabilities'] / norm
        except Warning:
            pass

    return result


#########################################################
# Internal functions used to transform probabilities
# and be used with uniroot.
#########################################################

# Calculate the probabilities usin Shin's formula, for a given value of z.
# io = inverted odds.
def shin_func(zz, io):
    bb = sum(io)
    return (np.sqrt(zz ** 2 + 4 * (1 - zz) * (((io) ** 2) / bb)) - zz) / (2 * (1 - zz))


# the condition that the sum of the probabilites must sum to 1.
# Used with uniroot.
def shin_solvefor(zz, io):
    tmp = shin_func(zz, io)
    return 1 - sum(tmp)  # 0 when the condition is satisfied.


# Calculate the probabilities using the odds ratio method,
# for a given value of the odds ratio cc.
# io = inverted odds.
def or_func(cc, io):
    return io / (cc + io - (cc * io))


# The condition that the sum of the probabilites must sum to 1.
# This function calulates the true probability, given bookmaker
# probabilites xx, and the odds ratio cc.
def or_solvefor(cc, io):
    tmp = or_func(cc, io)
    return sum(tmp) - 1


# power function.
def pwr_func(nn, io):
    return io ** (1 / nn)


# The condition that the sum of the probabilites must sum to 1.
# This function calulates the true probability, given bookmaker
# probabilites xx, and the inverse exponent. nn.
def pwr_solvefor(nn, io):
    tmp = pwr_func(nn, io)
    return sum(tmp) - 1
