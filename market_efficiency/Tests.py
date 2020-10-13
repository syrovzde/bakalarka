import helper_functions as hlp
import regression
import pandas as pd
import main
import pickle
import matplotlib.pyplot as plt
from utils.odds2probs import implied_probs

class Tests:
    def __init__(self, closed_odds, opening_odds, market, schema, test_list=None, margin='basic',
                 fair=True, closed_results = None,opening_results=None,Bookmaker = "",png_path="",txt_path="",pickle_path=""):
        self.png_path = png_path
        self.txt_path = txt_path
        self.pickle_path = pickle_path
        self.closed = closed_odds
        self.opening = opening_odds
        self.closed_results = closed_results
        self.opening_results = opening_results
        self.fair_closed = 1/implied_probs(self.closed, method=margin, normalize=True)
        self.fair_open = 1/implied_probs(self.opening, method=margin, normalize=True)
        self.market = market
        self.schema = schema
        self.name = schema+ "_" + market + "_" + Bookmaker
        self.test_list = test_list
        self.margin = margin
        self.market_choices = {'1x2': 3, 'ah': 2, 'ou': 2, 'bts': 2, 'dc': 3,'ha':2}
        self.fair = fair
        self.simple_func_list = [hlp.bet_favourite_dict, hlp.bet_Underdog_dict,hlp.bet_home_dict, hlp.bet_away_dict,hlp.bet_draw_dict]
        self.results = {}

    def kl_test(self):
        if self.market == 'ah' or self.market == 'ha':
            return None,None
        closed_kl = hlp.kl_divergence(self.closed, margin=self.margin, bet_choices=self.market_choices[self.market], odds=self.fair_closed,results=self.closed_results)
        opened_kl = hlp.kl_divergence(self.opening, margin=self.margin, bet_choices=self.market_choices[self.market], odds =self.fair_open,results=self.opening_results)
        self.results['opened_kl'] = opened_kl
        self.results['closed_kl'] = closed_kl
        return opened_kl, closed_kl

    """def regression_test(self):
        closed_reg = regression.Regression(data=self.closed, bet_choices=self.market_choices[self.market])
        opened_reg = regression.Regression(data=self.opening, bet_choices=self.market_choices[self.market])
        #print(closed_reg.run(model="Linear", odds=self.fair_closed,results=self.closed_results))
        #print(opened_reg.run(model='Linear', odds=self.fair_open,results=self.opening_results))
        return closed_reg"""

    def get_default_dict_simple(self, opening=True, asian=False):
        if opening:
            return {'fair': self.fair, 'margin': self.margin, 'odds': self.opening, 'data': None,
                    'bet_choices': self.market_choices[self.market], 'asian': asian,'results':self.opening_results}
        else:
            return {'fair': self.fair, 'margin': self.margin, 'odds': self.closed, 'data': None,
                    'bet_choices': self.market_choices[self.market], 'asian': asian,'results':self.closed_results}

    def get_default_dict_random(self,opening=True,asian=False):
        if opening:
            return {'fair': self.fair, 'margin': self.margin, 'odds': self.opening, 'data': None, 'bet_choices': self.market_choices[self.market],
                'n_random': 100, 'probabilities': None, 'asian': asian,'results':self.opening_results}
        else:
            return {'fair': self.fair, 'margin': self.margin, 'odds': self.closed, 'data': None,
                    'bet_choices': self.market_choices[self.market], 'n_random': 100, 'probabilities': None,
                    'asian': asian,'results':self.closed_results}

    def simple_tests(self):
        opened_result = []
        closed_result = []
        asian = self.market == 'ah' or self.market == 'ha'
        open_dict = self.get_default_dict_simple(asian=asian,opening=True)
        closed_dict = self.get_default_dict_simple(asian=asian,opening=False)
        favourite_open, underdog_open, home_open, away_open, draw_open = main.multiple_tests(self.simple_func_list,
                                                                                             open_dict)
        favourite_closed, underdog_closed, home_closed, away_closed, draw_closed = main.multiple_tests(
            self.simple_func_list, closed_dict)
        if self.market == 'bts':
            opened_result = pd.DataFrame(
                {'max': favourite_open, 'min': underdog_open, 'both_scored': home_open,
                 'both_did_not_score': draw_open})
            closed_result = pd.DataFrame(
                {'max': favourite_closed, 'min': underdog_closed, 'both_scored': home_closed,
                 'both_did_not_score': draw_closed})
        if self.market == 'ah':
            opened_result = pd.DataFrame(
                {'favourite': favourite_open, 'underdog': underdog_open, 'handicapped_team': home_open,
                 'not_handicapped_team': draw_open})
            closed_result = pd.DataFrame(
                {'favourite': favourite_closed, 'underdog': underdog_closed, 'handicapped_team': draw_closed,
                 'not_handicapped_team': home_closed})
        if self.market == 'ha':
            opened_result = pd.DataFrame(
                {'favourite': favourite_open, 'underdog': underdog_open, 'home': home_open,
                 'away': draw_open})
            closed_result = pd.DataFrame(
                {'favourite': favourite_closed, 'underdog': underdog_closed,'home': home_closed, 'away': draw_closed,
                 })
        if self.market == '1x2':
            opened_result = pd.DataFrame(
                {'favourite': favourite_open, 'underdog': underdog_open, 'home': home_open, 'away': away_open,
                 'draw': draw_open})
            closed_result = pd.DataFrame(
                {'favourite': favourite_closed, 'underdog': underdog_closed, 'home': home_closed, 'away': away_closed,
                 'draw': draw_closed})
        if self.market == 'ou':
            opened_result = pd.DataFrame(
                {'max': favourite_open, 'min': underdog_open, 'under': draw_open, 'over': home_open})
            closed_result = pd.DataFrame(
                {'max': favourite_closed, 'min': underdog_closed, 'under': draw_closed, 'over': home_closed})
        self.results['opened_simple'] = opened_result
        self.results['closed_simple'] = closed_result
        return opened_result, closed_result

    def devide_bet_test(self):
        if self.market == 'ah' or self.market == 'ha':
            return None
        opened = hlp.devide_bet(odds=self.opening, margin=self.margin, bet_choices=self.market_choices[self.market],results=self.opening_results)
        closed = hlp.devide_bet(odds=self.closed, margin=self.margin, bet_choices=self.market_choices[self.market],results=self.closed_results)
        opened = opened[opened != 0]
        closed = closed[closed != 0]

        df = pd.DataFrame({'closed': closed, 'open': opened})
        self.results['devide'] = df
        return opened, closed

    def random_tests(self):
        asian = self.market == 'ah' or self.market == 'ha'
        closed_dict = self.get_default_dict_random(asian=asian)
        opening_dict = self.get_default_dict_random(asian=asian)
        closed = hlp.random_dict(closed_dict)
        open = hlp.random_dict(opening_dict)
        df = pd.DataFrame({'closed': closed, 'open': open})
        self.results['random'] = df
        return df

    def test(self):
        for test in self.test_list:
            test(self)

    def save(self):
        if Tests.random_tests in self.test_list:
            plt.figure()
            self.results['random'].boxplot()
            plt.savefig(self.png_path + self.name+"_random.png")
        if Tests.simple_tests in self.test_list:
            plt.figure()
            self.results['opened_simple'].cumsum().plot()
            plt.title(self.name + "opening")
            plt.savefig(self.png_path + self.name+ "_opened_simple.png")
            plt.figure()
            self.results['closed_simple'].cumsum().plot()
            plt.title(self.name + 'closing')
            plt.savefig(self.png_path + self.name+'_closed_simple.png')
        if Tests.devide_bet_test in self.test_list:
            if self.market != 'ah' and self.market != 'ha':
                plt.figure()
                self.results['devide']['closed'].cumsum().plot()
                plt.savefig(self.png_path + self.name+'_closed_devide.png')
                plt.figure()
                self.results['devide']['open'].cumsum().plot()
                plt.savefig(self.png_path + self.name+'_open_devide.png')
        if Tests.kl_test in self.test_list:
            if self.market != 'ah' and self.market != 'ha':
                f = open(self.txt_path + self.name+".txt",'w')
                try:
                    f.write("Opened kl is:" + str(self.results['opened_kl']) + ", closed kl is:" + str(self.results['closed_kl']))
                except:
                 pass
                f.close()
