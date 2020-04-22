import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import helper_functions as hlp
import regression as regression

random = hlp.random_dict
kl = hlp.kl_divergence_dict
favourite = hlp.bet_favourite_dict
underdog = hlp.bet_Underdog_dict
home = hlp.bet_home_dict
away = hlp.bet_away_dict
draw = hlp.bet_draw_dict
simple_tests_func_list = [favourite, underdog, home, away, draw]
default_dtb = "postgresql+pg8000://postgres:1234@localhost:5432/tmp"
default_schema = "football"
market_choices = {'1x2': 3, 'ah': 2, 'ou': 2, 'bts': 2, 'dc': 3}


def get_default_dict_simple(data, margin='basic', fair=True, bet_choices=3, asian=False):
    return {'fair': fair, 'margin': margin, 'odds': None, 'data': data, 'bet_choices': bet_choices, 'asian': asian}


def get_default_dict_random(data, margin='basic', fair=True, bet_choices=3, n_random=100, probabilities=None, odds=None,
                            asian=False):
    return {'fair': fair, 'margin': margin, 'odds': odds, 'data': data, 'bet_choices': bet_choices,
            'n_random': n_random, 'probabilities': probabilities, 'asian': asian}


def random_tests(market, margin='basic', schema=default_schema, opened=None, closed=None):
    if opened is None and closed is None:
        closed, opened = load_data(market=market, schema=schema)
    asian = market == 'ah'
    closed_dict = get_default_dict_random(closed, margin=margin, bet_choices=market_choices[market], asian=asian)
    opening_dict = get_default_dict_random(opened, margin=margin, bet_choices=market_choices[market], asian=asian)
    closed = hlp.random_dict(closed_dict)
    open = hlp.random_dict(opening_dict)
    df = pd.DataFrame({'closed': closed, 'open': open})
    return df


def get_select_script(market, schema=default_schema):
    if market == '1x2':
        return 'SELECT "Matches"."MatchID","1", "X" as "0", "2", "Result","Bookmaker" FROM {y}.' '"Matches" ' \
               'INNER JOIN {y}."Odds_1x2" on "Odds_1x2"."MatchID" = "Matches"."MatchID" ' \
               'LIMIT 1000000 '.format(y=schema)
    if market == 'ah':
        return 'SELECT "Matches"."MatchID","1" , "2" as "0", "Result","Bookmaker","Handicap" FROM {y}.' '"Matches" ' \
               'INNER JOIN {y}."Odds_ah" on "Odds_ah"."MatchID" = "Matches"."MatchID" ' \
               'LIMIT 3000000'.format(y=schema)
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
    asian = market == 'ah'
    open_dict = get_default_dict_simple(opened, bet_choices=market_choices[market], fair=fair, margin=margin,
                                        asian=asian)
    closed_dict = get_default_dict_simple(closed, bet_choices=market_choices[market], fair=fair, margin=margin,
                                          asian=asian)
    arg_list_open = [open_dict, open_dict, open_dict, open_dict, open_dict]
    arg_list_closed = [closed_dict, closed_dict, closed_dict, closed_dict, closed_dict]
    favourite_open, underdog_open, home_open, away_open, draw_open = multiple_tests(simple_tests_func_list,
                                                                                    arg_list_open)
    favourite_closed, underdog_closed, home_closed, away_closed, draw_closed = multiple_tests(simple_tests_func_list,
                                                                                              arg_list_closed)
    if market == 'bts':
        opened_result = pd.DataFrame({'favourite': favourite_open, 'underdog': underdog_open, 'both_scored': home_open,
                                      'both_did_not_score': draw_open})
        closed_result = pd.DataFrame(
            {'favourite': favourite_closed, 'underdog': underdog_closed, 'both_scored': home_closed,
             'both_did_not_score': draw_closed})
    if market == 'ah':
        opened_result = pd.DataFrame(
            {'favourite': favourite_open, 'underdog': underdog_open, 'handicapped_team': home_open,
             'not_handicapped_team': draw_open})
        closed_result = pd.DataFrame(
            {'favourite': favourite_closed, 'underdog': underdog_closed, 'both_scored': draw_closed,
             'both_did_not_score': home_closed})
    if market == '1x2':
        opened_result = pd.DataFrame(
            {'favourite': favourite_open, 'underdog': underdog_open, 'home': home_open, 'away': away_open,
             'draw': draw_open})
        closed_result = pd.DataFrame(
            {'favourite': favourite_closed, 'underdog': underdog_closed, 'home': home_closed, 'away': away_closed,
             'draw': draw_closed})
    if market == 'ou':
        opened_result = pd.DataFrame(
            {'favourite': favourite_open, 'underdog': underdog_open, 'under': draw_open, 'over': home_open})
        closed_result = pd.DataFrame(
            {'favourite': favourite_closed, 'underdog': underdog_closed, 'under': draw_closed, 'over': home_closed})

    return opened_result, closed_result


def devide_bet_test(opened, closed, market, margin):
    opened = hlp.devide_bet(data=opened, margin=margin, bet_choices=market_choices[market])
    closed = hlp.devide_bet(data=closed, margin=margin, bet_choices=market_choices[market])
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
    if market_choices[market] == 2:
        df = df[np.logical_and(df['1'].astype(float) >= 1, df['0'].astype(float) >= 1)]
    else:
        df = df[
            np.logical_and.reduce([df['1'].astype(float) >= 1, df['0'].astype(float) >= 1, df['2'].astype(float) >= 1])]
    df = df.reset_index()
    if market == 'ou':
        df['results'] = hlp.getResults(df['Result'], market, df['Total'], sport=schema)
    elif market == 'ah':
        df['results'] = hlp.getResults(results=df['Result'], market=market, handicap=df['Handicap'], sport=schema)
    else:
        df['results'] = hlp.getResults(df['Result'], market)
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
    asian = market == 'ah'
    closed_kl = hlp.kl_divergence(closed, margin=margin, bet_choices=market_choices[market])
    opened_kl = hlp.kl_divergence(opening, margin=margin, bet_choices=market_choices[market])
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
    results['simple'] = simple_tests(market=market, fair=False, margin=margin, schema=schema, opened=opening,
                                     closed=closed)
    results['simple'][0].cumsum().plot()
    plt.title("opened")
    results['simple'][1].cumsum().plot()
    plt.title("closed")
    results['kl'] = kl_test(market=market,schema = schema,margin=margin,opening=opening,closed=closed)
    print("closed odds kl is {closed} opened odds is {opened}".format(closed=results['kl'][1], opened=results['kl'][0]))
    results['random'] = random_tests(market=market,schema=schema,margin=margin,opened=opening,closed=closed)
    results['random'].boxplot()
    plt.title("random")
    results['regression'] = regression_test(market=market,schema=schema,margin=margin,closed=closed,opening=opening)
    print('regression_done')
    results['devide'] = devide_bet_test(opened=opening, closed=closed, market=market, margin=margin)
    print("devide_done")
    plt.show()
    return results


def test(test_list, test_args, dtb_url=default_dtb, schema="Football", market="1x2"):
    closed, opening = load_data(dtb_url, schema)
    return multiple_tests(test_list, test_args)


def main(argv, argc):
    if argc == 4:
        schema = argv[1]
        market = argv[2]
        margin = argv[3]
        return all_tests(schema=schema, market=market, margin=margin)
    else:
        return all_tests()


if __name__ == '__main__':
    main(sys.argv, len(sys.argv))
