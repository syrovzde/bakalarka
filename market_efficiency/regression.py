import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from market_efficiency.method import Method


class Regression(Method):

    def run(self, model, odds):
        """
        :param model: "Linear for linear regression", Logistic for logistic regression
        :param odds: odds for matches
        :return:
        """
        market_probabilities = 1 / odds
        n_bets = np.shape(market_probabilities)[0]
        favourites = np.argmin(market_probabilities, axis=1)
        y = np.array((self.data['results'] == favourites), dtype=np.uint8)
        bettors = self.getWinnerProbabilites(self.get_probabilities())
        market = self.getWinnerProbabilites(market_probabilities)
        A = np.ones((2, n_bets))
        A[0] = bettors
        A[1] = market
        if model == 'Linear':
            return linear(A.T, y, bettors, n_bets)
        elif model == 'Logistic':
            return logistic(A.T, y)
        return None


def logistic(A, y):
    """
    Simple logistic regression
    Creates 2 models:
        First result = a*market_probability + e
    :param A: n*p * matrix n is number of bets used for regression and p is number of parameteres
    :param y: vector of results 0 zero if favourite lost 1 if won
    :param market:
    :param bettors:
    :param n_bets:
    :return:
    """
    b_model = LogisticRegression().fit(A, y)
    predicted_b = b_model.predict(A)

    without_b = A[:, 1].reshape(-1, 1)
    without_b_model = LogisticRegression().fit(without_b, y)
    predicted_without_b = without_b_model.predict(without_b)
    log_mle_b = log_loss(y, predicted_b)
    log_mle_without_b = log_loss(y, predicted_without_b)
    lr = 2 * (log_mle_without_b - log_mle_b)
    return stats.chi2(1).cdf(lr)


def linear(A, y, bettors, n_bets):
    p = LinearRegression().fit(A, y)
    bettor_weight = p.coef_[0]
    predicted = p.predict(A)
    squares = np.sqrt(np.sum(np.square(predicted - y)) / (n_bets - 2))
    bettor_squares = np.sqrt(np.sum(np.square(bettors - np.average(bettors))))
    SE = squares / bettor_squares
    T = bettor_weight / SE
    return stats.t.cdf(T, n_bets - 2)
