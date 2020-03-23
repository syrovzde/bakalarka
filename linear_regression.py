import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from method import Method


class Linear(Method):

    def run(self, model, odds):
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
            return linear(A.T, y, market, bettors, n_bets)
        elif model == 'Logistic':
            return logistic(A.T, y, market, bettors, n_bets)
        return None


def logistic(A, y, market, bettors, n_bets):
    b_model = LogisticRegression().fit(A, y)
    predicted_b = b_model.predict(A)

    without_b = A[:, 1].reshape(-1, 1)
    without_b_model = LogisticRegression().fit(without_b, y)
    predicted_without_b = without_b_model.predict(without_b)
    log_mle_b = log_loss(y, predicted_b)
    log_mle_without_b = log_loss(y, predicted_without_b)
    lr = 2 * (log_mle_without_b - log_mle_b)
    return stats.chi2(1).cdf(lr)


def linear(A, y, market, bettors, n_bets):
    p = LinearRegression().fit(A, y)
    bettor_weight = p.coef_[0]
    predicted = p.predict(A)
    squares = np.sqrt(np.sum(np.square(predicted - y)) / (n_bets - 2))
    bettor_squares = np.sqrt(np.sum(np.square(bettors - np.average(bettors))))
    SE = squares / bettor_squares
    T = bettor_weight / SE
    return stats.t.cdf(T, n_bets - 2)
