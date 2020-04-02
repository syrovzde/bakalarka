from typing import Dict, Any, Union

import market_efficiency.main
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import market_efficiency.main
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


def get_default_dict(data, margin='basic', fair=False):
    return {'fair': fair, 'margin': margin, 'odds': None, 'data': data}


def simple_tests(arg_list=None):
    opened, closed = load_data(default_dtb, default_schema)
    if arg_list is None:
        open_dict = get_default_dict(opened)
        closed_dict = get_default_dict(closed)
        arg_list_open = [open_dict, open_dict, open_dict, open_dict, open_dict]
        arg_list_closed = [closed_dict, closed_dict, closed_dict, closed_dict, closed_dict]
        favourite, underdog, home, away, draw = multiple_tests(simple_tests_func_list, arg_list_open)
        print(type(favourite[0]))
        opened = pd.DataFrame({'favourite': favourite, 'underdog': underdog, 'home': home, 'away': away, 'draw': draw})
        favourite, underdog, home, away, draw = multiple_tests(simple_tests_func_list, arg_list_closed)
        closed = pd.DataFrame({'favourite': favourite, 'underdog': underdog, 'home': home, 'away': away, 'draw': draw})
    return opened, closed


def plot_simple_tests(opened, closed):
    pass


def multiple_tests(test_list, argument_lists):
    results = []
    for func, args in zip(test_list, argument_lists):
        results.append(func(args))
    return results


def load_data(dtb_url, schema):
    conn = create_engine(dtb_url)
    select_script = 'SELECT "Matches"."MatchID","1", "X", "2", "Result","Bookmaker" FROM {y}.' '"Matches" ' \
                    'INNER JOIN {y}."Odds_1x2" on "Odds_1x2"."MatchID" = "Matches"."MatchID" ' \
                    'LIMIT 1000000 '.format(y=schema)
    start = time.time()
    df = pd.read_sql_query(select_script, con=conn)
    df = df.dropna()
    df['results'] = market_efficiency.main.getResults(df['Result'])
    start = time.time()
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


def test(test_list, test_args, dtb_url=default_dtb, schema="Football"):
    closed, opening = load_data(dtb_url, schema)  # return multiple_tests(test_list,test_args)

if __name__ == '__main__':
    closed, opening = load_data("postgresql+pg8000://postgres:1234@localhost:5432/tmp", schema='football')
    arg_list = get_default_dict(data=closed)
    pom = [arg_list, arg_list, arg_list, arg_list, arg_list]
    print(np.average(simple_tests(pom, test_list=simple_tests_func_list)[0]))
