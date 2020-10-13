import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from method import Method


class Regression(Method):

    def run(self, model, odds,results):
        """
        :param model: "Linear for linear regression", Logistic for logistic regression
        :param odds: odds for matches
        :return:
        """
        market_probabilities = 1 / odds
        n_bets = np.shape(market_probabilities)[0]
        if results is None:
            results = self.data['results']
        #choose whether home or away yeam will bhe considered
        tmp = np.random.randint(0, 3, results.size())
        outcomes = np.array(tmp == results,dtype=int)
        favourites = np.argmax(market_probabilities, axis=1)
        favourites = np.array(favourites == tmp, dtype=int)
        market = market_probabilities[:, tmp]
        home = np.array(tmp == 1, dtype=int)
        A = np.ones((4, n_bets))
        A[1,:] = market
        A[2,:] = home
        A[3,:] = favourites
        y = outcomes
        if model == 'Linear':
            return linear(A.T, y)
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
    :return: p value
    """
    b_model = LogisticRegression().fit(A, y)
    predicted_b = b_model.predict(A)

    without_b = A[:, 1].reshape(-1, 1)
    without_b = A[[0,1],:]
    without_b_model = LogisticRegression().fit(without_b, y)
    predicted_without_b = without_b_model.predict(without_b)
    log_mle_b = log_loss(y, predicted_b)
    log_mle_without_b = log_loss(y, predicted_without_b)
    lr = - 2 * (log_mle_without_b - log_mle_b)
    p = 1-2*stats.chi2.cdf(lr,2)
    return p


def linear(A, y):
    n_bets = A.shape()[0]
    p = LinearRegression().fit(A, y)
    bettor_weight = p.coef_[0]
    predicted = p.predict(A)
    squares = np.sqrt(np.sum(np.square(predicted - y)) / (n_bets - 2))
    bettor_squares = np.sqrt(np.sum(np.square(bettors - np.average(bettors))))
    SE = squares / bettor_squares
    T = bettor_weight / SE
    p = 1-2*stats.t.cdf(T,n_bets-2)
    return p
