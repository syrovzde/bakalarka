from typing import Dict, Any, Union

import market_efficiency.main
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import market_efficiency.main
import matplotlib.pyplot as plt
import market_efficiency.regression as regression
import time

random = market_efficiency.main.random_dict
kl = market_efficiency.main.kl_divergence_dict
favourite = market_efficiency.main.bet_favourite_dict
underdog = market_efficiency.main.bet_Underdog_dict
home = market_efficiency.main.bet_home_dict
away = market_efficiency.main.bet_away_dict
draw = market_efficiency.main.bet_draw_dict
simple_tests_func_list = [favourite, underdog, home, away, draw]
default_dtb = "postgresql+pg8000://postgres:1234@localhost:5432/tmp"
default_schema = "football"
market_choices = {'1x2': 3, 'ah': 2, 'ou': 2, 'bts': 2, 'dc': 3}


def get_default_dict_simple(data, margin='basic', fair=True, bet_choices=3):
    return {'fair': fair, 'margin': margin, 'odds': None, 'data': data, 'bet_choices': bet_choices}


def get_default_dict_random(data, margin='basic', fair=True, bet_choices=3, n_random=100, probabilities=None,
                            odds=None):
    return {'fair': fair, 'margin': margin, 'odds': odds, 'data': data, 'bet_choices': bet_choices,
            'n_random': n_random, 'probabilities': probabilities}


def random_tests(market, margin='basic', schema=default_schema, opened=None, closed=None):
    if opened is None and closed is None:
        closed, opened = load_data(market=market, schema=schema)
    closed_dict = get_default_dict_random(closed, margin=margin, bet_choices=market_choices[market])
    opening_dict = get_default_dict_random(opened, margin=margin, bet_choices=market_choices[market])
    closed = market_efficiency.main.random_dict(closed_dict)
    open = market_efficiency.main.random_dict(opening_dict)
    df = pd.DataFrame({'closed': closed, 'open': open})
    return df


def get_select_script(market, schema=default_schema):
    if market == '1x2':
        return 'SELECT "Matches"."MatchID","1", "X" as "0", "2", "Result","Bookmaker" FROM {y}.' '"Matches" ' \
               'INNER JOIN {y}."Odds_1x2" on "Odds_1x2"."MatchID" = "Matches"."MatchID" ' \
               'LIMIT 1000000 '.format(y=schema)
    if market == 'ah':
        return 'SELECT "Matches"."MatchID","1" as "0", "2" as "1", "Result","Bookmaker" FROM {y}.' '"Matches" ' \
               'INNER JOIN {y}."Odds_ah" on "Odds_ah"."MatchID" = "Matches"."MatchID" ' \
               'LIMIT 1000000 '.format(y=schema)
    if market == 'bts':
        return 'SELECT "Matches"."MatchID","YES" as "1", "NO" as "0", "Result","Bookmaker" FROM {y}.' '"Matches" ' \
               'INNER JOIN {y}."Odds_bts" on "Odds_bts"."MatchID" = "Matches"."MatchID" ' \
               'LIMIT 1000000 '.format(y=schema)
    if market == 'ou':
        return 'SELECT "Matches"."MatchID","Under" as "0", "Over" as "1","Total", "Result","Bookmaker","Details" FROM {y}.' '"Matches" ' \
               'INNER JOIN {y}."Odds_ou" on "Odds_ou"."MatchID" = "Matches"."MatchID" ' \
               'LIMIT 1000000 '.format(y=schema)
    if market == 'dc':
        return 'SELECT "Matches"."MatchID","1X" as "0","12" as "1","X2" as "2", "Result","Bookmaker" FROM {y}.' '"Matches" ' \
               'INNER JOIN {y}."Odds_dc" on "Odds_dc"."MatchID" = "Matches"."MatchID" ' \
               'LIMIT 1000000 '.format(y=schema)


def simple_tests(market="1x2", fair=True, margin='basic', schema=default_schema, opened=None, closed=None):
    if opened is None and closed is None:
        opened, closed = load_data(dtb_url=default_dtb, schema=schema, market=market)
    open_dict = get_default_dict_simple(opened, bet_choices=market_choices[market], fair=fair, margin=margin)
    closed_dict = get_default_dict_simple(closed, bet_choices=market_choices[market], fair=fair, margin=margin)
    arg_list_open = [open_dict, open_dict, open_dict, open_dict, open_dict]
    arg_list_closed = [closed_dict, closed_dict, closed_dict, closed_dict, closed_dict]
    favourite, underdog, home, away, draw = multiple_tests(simple_tests_func_list, arg_list_open)
    opened = pd.DataFrame({'favourite': favourite, 'underdog': underdog, 'home': home, 'away': away, 'draw': draw})
    favourite, underdog, home, away, draw = multiple_tests(simple_tests_func_list, arg_list_closed)
    closed = pd.DataFrame({'favourite': favourite, 'underdog': underdog, 'home': home, 'away': away, 'draw': draw})
    return opened, closed


def devide_bet_test(opened, closed, market, margin):
    opened = market_efficiency.main.devide_bet(data=opened, margin=margin, bet_choices=market_choices[market])
    closed = market_efficiency.main.devide_bet(data=closed, margin=margin, bet_choices=market_choices[market])
    return opened, closed

def plot_simple_tests(opened, closed):
    opened.cumsum().plot()
    closed.cumsum().plot()
    plt.show()


def multiple_tests(test_list, argument_lists):
    results = []
    for func, args in zip(test_list, argument_lists):
        results.append(func(args))
    return results


def load_data(schema, market, dtb_url=default_dtb):
    conn = create_engine(dtb_url)
    select_script = get_select_script(market=market, schema=schema)
    df = pd.read_sql_query(select_script, con=conn)
    df = df.dropna()
    df = df[df['Result'] != '-']
    df = df[df['Result'] != "---"]
    df = df.reset_index()
    if market_choices[market] == 2:
        df = df[np.logical_and(df['1'].astype(float) >= 1, df['0'].astype(float) >= 1)]
    else:
        df = df[
            np.logical_and.reduce([df['1'].astype(float) >= 1, df['0'].astype(float) >= 1, df['2'].astype(float) >= 1])]
    if 'Total' in df.columns:
        if schema == "tennis":
            df['results'] = market_efficiency.main.getResults(df['Details'], market, df['Total'], sport=schema)
            print(df['results'])
        else:
            df['results'] = market_efficiency.main.getResults(df['Result'], market, df['Total'], sport=schema)
    else:
        df['results'] = market_efficiency.main.getResults(df['Result'], market)
    closed, opening = devide_closed_opening(df)
    return closed, opening


def devide_closed_opening(df):
    length = len(df.index)
    opening_indexes = np.empty(length, dtype=np.uint32)
    closed_indexes = np.empty(length, dtype=np.uint32)
    adding_index = 0
    opening_indexes[0] = 0
    last_closed = 0
    last_bookmaker = df.iloc[0]['Bookmaker']
    last_match_id = df.iloc[0]['MatchID']
    for row in zip(df.iloc[1:]['Bookmaker'], df.iloc[1:]['MatchID'], df.iloc[1:].index):
        if row[0] != last_bookmaker or last_match_id != row[1]:
            opening_indexes[adding_index + 1] = row[2]
            closed_indexes[adding_index] = last_closed
            adding_index += 1
        last_closed = row[2]
        last_match_id = row[1]
        last_bookmaker = row[0]
    closed_indexes[adding_index] = last_closed
    closed = df.loc[closed_indexes[0:adding_index + 1]]
    opening = df.loc[opening_indexes[0:adding_index + 1]]
    return closed, opening


def kl_test(market, schema=default_schema, margin='basic', closed=None, opening=None):
    if opening is None and closed is None:
        closed, opening = load_data(dtb_url=default_dtb, schema=schema, market=market)
    closed_kl = market_efficiency.main.kl_divergence(closed, margin=margin, bet_choices=market_choices[market])
    opened_kl = market_efficiency.main.kl_divergence(opening, margin=margin, bet_choices=market_choices[market])
    return opened_kl, closed_kl


def regression_test(market='1x2', schema=default_schema, margin='basic', dtb=default_dtb, closed=None, opening=None):
    if closed is None and opening is None:
        closed, opening = load_data(schema=schema, market=market, dtb_url=dtb)
    closed_reg = regression.Regression(data=closed, bet_choices=market_choices[market])
    closed_odds = closed_reg.find_fair_odds(margin)
    opened_reg = regression.Regression(data=opening, bet_choices=market_choices[market])
    opened_odds = opened_reg.find_fair_odds(margin)
    # print(closed_reg.run(model = "Linear",odds = closed_odds))
    # print(opened_reg.run(model = 'Linear',odds=opened_odds))
    return closed_reg


def all_tests(schema=default_schema, market='1x2', margin='basic'):
    results = {}
    closed, opening = load_data(schema=schema, market=market, dtb_url=default_dtb)
    print("loaded, starting to test")
    """results['simple'] = simple_tests(market=market,fair=True,margin=margin,schema=schema,opened = opening,closed=closed)
    print("simple_done")
    results['kl'] = kl_test(market=market,schema = schema,margin=margin,opening=opening,closed=closed)
    print("kl_done")
    results['random'] = random_tests(market=market,schema=schema,margin=margin,opened=opening,closed=closed)
    print("random_done")
    results['regression'] = regression_test(market=market,schema=schema,margin=margin,closed=closed,opening=opening)
    print('regression_done')"""
    results['devide'] = devide_bet_test(opened=opening, closed=closed, market=market, margin=margin)
    print("devide_done")
    return results


def test(test_list, test_args, dtb_url=default_dtb, schema="Football", market="1x2"):
    closed, opening = load_data(dtb_url, schema)
    return multiple_tests(test_list, test_args)


def tmp(a, b):
    return a, b

if __name__ == '__main__':
    closed, opening = load_data(dtb_url="postgresql+pg8000://postgres:1234@localhost:5432/tmp", market='ou',
                                schema='tennis')  # opened,closed = simple_tests(market='bts')  # plot_simple_tests(opened,closed)  # print(opened)  # print(closed)  # print(simple_tests(arg_list,market='bts'))  # print(np.average(simple_tests(pom, test_list=simple_tests_func_list)[0]))  #all_tests(market='1x2',margin='wpo',schema=default_schema)
