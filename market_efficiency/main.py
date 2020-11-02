import config
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import helper_functions as hlp
import Tests
import argparse
import sqlalchemy
import time
import utils.odds2probs as marg
from scipy.optimize import linprog
import config

default_dtb = "postgresql+pg8000://postgres:1234@localhost:5432/tmp"
default_schema = "football"
market_choices = {'1x2': 3, 'ah': 2, 'ou': 2, 'bts': 2, 'dc': 3,'ha':2}

schemas = ['football','baseball','basketball','hockey','handball','volleyball']
markets = ['1x2','bts','ou','ah','ha']
default_bookmakers = [ 'bet-at-home', 'bwin', 'Unibet', 'BetVictor', 'Tipsport.cz', 'Betsafe', 'Pinnacle', 'Chance.cz', 'GoldBet' ,'188BET']
#Bookmakers = ['Pinnacle']



def get_select_script(market, schema=default_schema):
    """
    Loads data from database
    :param market:
    :param schema:
    :return:
    """
    if market == '1x2':
        return 'SELECT "Matches"."MatchID","1", "X" as "0", "2", "Result","Bookmaker" FROM {y}.' '"Matches" ' \
               'INNER JOIN {y}."Odds_1x2" on "Odds_1x2"."MatchID" = "Matches"."MatchID" ' \
               'LIMIT 100000  '.format(y=schema)
    if market == 'ah':
        return 'SELECT "Matches"."MatchID","1" , "2" as "0", "Result","Bookmaker","Handicap" FROM {y}.' '"Matches" ' \
               'INNER JOIN {y}."Odds_ah" on "Odds_ah"."MatchID" = "Matches"."MatchID" ' \
               'LIMIT 100000'.format(y=schema)
    if market == 'bts':
        return 'SELECT "Matches"."MatchID","YES" as "1", "NO" as "0", "Result","Bookmaker" FROM {y}.' '"Matches" ' \
               'INNER JOIN {y}."Odds_bts" on "Odds_bts"."MatchID" = "Matches"."MatchID" ' \
               'LIMIT 100000 '.format(y=schema)
    if market == 'ou':
        return 'SELECT "Matches"."MatchID","Under" as "0", "Over" as "1","Total", "Result","Bookmaker","Details" FROM {y}.' '"Matches" ' \
               'INNER JOIN {y}."Odds_ou" on "Odds_ou"."MatchID" = "Matches"."MatchID" ' \
               'LIMIT 100000 '.format(y=schema)
    if market == 'dc':
        return 'SELECT "Matches"."MatchID","1X" as "0","12" as "1","X2" as "2", "Result","Bookmaker" FROM {y}.' '"Matches" ' \
               'INNER JOIN {y}."Odds_dc" on "Odds_dc"."MatchID" = "Matches"."MatchID" ' \
               'LIMIT 100000 '.format(y=schema)
    if market == 'ha':
        return 'SELECT "Matches"."MatchID","1" , "2" as "0", "Result","Bookmaker" FROM {y}.' '"Matches" ' \
               'INNER JOIN {y}."Odds_ha" on "Odds_ha"."MatchID" = "Matches"."MatchID" ' \
               'LIMIT 100000'.format(y=schema)


def multiple_tests(test_list, argument):
    results = []
    for func in test_list:
        results.append(func(argument))
    return results


def load_data(schema=None, market=None, dtb_url=default_dtb,to_numpy = False,csv=False,csv_file=""):
    """
    Loads and parses data
    :param schema:
    :param market:
    :param dtb_url:
    :param to_numpy:
    :param csv:
    :param csv_file:
    :return:
    """
    if not csv:
        conn = create_engine(dtb_url)
        select_script = get_select_script(market=market, schema=schema)
        try:
            df = pd.read_sql_query(select_script, con=conn)
        except sqlalchemy.exc.ProgrammingError as e:
            return None,None,None,None
    else:
        df = pd.read_csv(csv_file)
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
        #select only whole number totals
        df = df[df['Total']%1 != 0.0]
        df = df.reset_index()
        df['results'] = hlp.getResults(df['Result'], market, df['Total'], sport=schema)
    elif market == 'ah':
        df['results'] = hlp.getResults(results=df['Result'], market=market, handicap=df['Handicap'], sport=schema,from_string =not csv)
        df = df.dropna()
    elif market == '1x2':
        df['results'] = hlp.getResults(df['Result'], market)
    else:
        df['results'] = hlp.getResults(df['Result'], market)
    if df['results'] is None:
        return None, None, None, None
    closed, opening,closed_results,opening_results = devide_closed_opening(df)
    if to_numpy:
        if market_choices[market] == 3:
            closed = closed[['0','1','2']].to_numpy(dtype=np.float)
            opening = opening[['0','1','2']].to_numpy(dtype=np.float)
        if market_choices[market] == 2:
            closed = closed[['0','1']].to_numpy(dtype=np.float)
            opening = opening[['0','1']].to_numpy(dtype=np.float)
    return closed,opening,closed_results,opening_results


def round_to_decimal(x2,ou_odds,ah_odds,bts_odds,places=4):
    return np.around(x2,decimals=places),np.around(ou_odds,decimals=places),\
           np.around(ah_odds,decimals=places),np.around(bts_odds,decimals=places),


def to_odds(dataframe,market):
    global tmp
    if market_choices[market] == 3:
        tmp = dataframe[['0', '1', '2']].to_numpy(dtype=np.float)
    if market_choices[market] == 2:
        tmp = dataframe[['0', '1']].to_numpy(dtype=np.float)
    return tmp


def probability_test_load_data(csv_file,points = 10,places=4):
    """
    Probability test implementation
    :param csv_file:
    :param points:
    :param places:
    :return:
    """
    tries = 100
    iterations = 0
    found = 0
    not_found = 0
    method = 'basic'
    df = pd.read_csv(csv_file)
    grouped  = df.groupby(['MatchID','1','2','X'])
    start = time.time()
    home, draw, away = hlp.x_market_table(points=points)
    scored, not_scored = hlp.bts_market_table(points=points)
    default_A = np.concatenate([home,draw,away,scored,not_scored])
    c = -np.ones(points**2)
    for name,group in grouped:
        if iterations == tries:
            break
        one, two = hlp.ah_result_table(handicap=group['Handicap'].unique(), points=points)
        over, under = hlp.ou_result_table(totals=group['Total'].unique(), points=points)
        x2_prop = marg.implied_probs(group[['1','X','2']].iloc[0].to_numpy(),method=method,normalize=False)
        bts_prop = marg.implied_probs(group[['YES','NO']].iloc[0].to_numpy(),method=method,normalize=False)
        pm = group['Over'].unique()
        mv = group['Under'].unique()
        ou_odds = np.array([pm,mv]).T
        if pm.size != mv.size:
            continue
        ou_prop = marg.implied_probs(ou_odds,method=method,normalize=True)
        pm = group['AH1'].unique()
        mv = group['AH2'].unique()
        ah_odds = np.array([pm, mv]).T
        if pm.size != mv.size:
            continue
        ah_prop = marg.implied_probs(ah_odds, method=method, normalize=True)
        if ou_prop is None or bts_prop is None or  ah_prop is None:
            continue
        ou_prop = ou_prop.T.reshape(-1)
        ah_prop = ah_prop.T.reshape(-1)
        x2_prop,ou_odds,ah_prop,bts_prop = round_to_decimal(x2=x2_prop,bts_odds=bts_prop,ah_odds=ah_prop,ou_odds=ou_prop,places=places)
        #b = np.concatenate([x2_prop,bts_prop, ou_prop, ah_prop])
        b = np.concatenate([x2_prop,bts_prop])
        b = np.around(b,decimals=places)
        #A = np.concatenate([default_A,over,under,one,two])
        A = default_A
        sol = linprog(c=c,A_eq=A,b_eq=b,bounds=(0,1))
        if sol['status'] == 0:
            found += 1
        else:
            not_found += 1
        iterations += 1
    print("Succesfull: {found},Unsuccessfull: {not_found}".format(found=found,not_found=not_found))


def devide_closed_opening(df):
    """
    Selects opening and closing odds from dataframe
    :param df:
    :return:
    """
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
    return closed, opening ,df['results'].loc[closed_indexes[0:adding_index + 1]],df['results'].loc[opening_indexes[0:adding_index + 1]]


def test(dtb_url=default_dtb, schema="Football", market="1x2",margin='basic',csv=True,csvfile="",bookmakers=None):
    if bookmakers is None:
        bookmakers = default_bookmakers
    closed, opening,_,_ = load_data(dtb_url=dtb_url,
                                    schema=schema,market=market,to_numpy=False,csv=csv,csv_file=csvfile)
    if closed is None:
        return
    for bookmaker in bookmakers:
        closed_pom = closed[closed['Bookmaker']==bookmaker]
        opening_pom = opening[opening['Bookmaker']==bookmaker]
        closed_odds = to_odds(closed_pom,market=market)
        closed_results = closed_pom['results']
        opening_odds = to_odds(opening_pom,market=market)
        opening_results = opening_pom['results']
        test = Tests.Tests(closed_odds=closed_odds, opening_odds=opening_odds, market=market, schema=schema,margin=margin,
                           opening_results=opening_results,closed_results=closed_results,Bookmaker=bookmaker,
                           test_list=config.default_list,txt_path=config.txt_path,pickle_path=config.pickle_path,
                           png_path=config.png_path)
        test.test()
        test.save()



def all_csv_test(csv_files,csv_markets,csv_schemas,bookmakers = None):
    for file,market,schema in zip(csv_files,csv_markets,csv_schemas):
        file = "csv_data\\" + file
        test(csv=True,market=market,schema=schema,csvfile=file,bookmakers=bookmakers)
    return

"""def main(args):
    if args['all_markets']:
        for schema in schemas:
            for market in markets:
                if args['all_tests']:
                    test(schema=schema,market=market)
                else:
                    return None
    return None"""

if __name__ == '__main__':
    """
    Starting point of the test. All the presented tests
    for football 1x2 market subset included in football_1x2_for_test.csv 
    """
    #csv_files = ['baseball_ou.csv']
    #csv_markets = ["ou"]
    #csv_schemas = ["baseball"]
    #csv_files = ['baseball_ah.csv', 'baseball_ou.csv', 'baseball_ha.csv', 'basketball_1x2.csv', 'basketball_ah.csv',
    #             'basketball_ou.csv', 'basketball_ha.csv', 'football_ha.csv', 'football_1x2.csv', 'football_ah.csv',
    #             'football_ou.csv', 'football_bts.csv', 'handball_1x2.csv', 'handball_ah.csv', 'hockey_ou.csv',
    #             'hockey_1x2.csv', 'hockey_ha.csv', 'hockey_ah.csv', 'hockey_bts.csv', 'volleyball_ah.csv',
    #             'volleyball_ha.csv']
    #csv_schemas = ['baseball', 'baseball', 'baseball', 'basketball', 'basketball', 'basketball', 'basketball',
    #               'football', 'football', 'football', 'football', 'football', 'handball', 'handball', 'hockey',
    #               'hockey', 'hockey', 'hockey', 'hockey', 'volleyball']
    #csv_markets = ['ah', 'ou', 'ha', '1x2', 'ah', 'ou', 'ha', 'ha', '1x2', 'ah', 'ou', 'bts', '1x2', 'ah', 'ou', '1x2',
    #               'ha', 'ah', 'bts', 'ah', 'ha']
    #all_csv_test(csv_files=csv_files,csv_schemas=csv_schemas,csv_markets=csv_markets)
    #print("results are available in directories {txt},{png}".format(txt=config.txt_path,png = config.png_path))
    #probability_test_load_data(csv_file="test_probability.csv")
    """parser = argparse.ArgumentParser(description="Market efficiency program testing")
    parser.add_argument('--schema',help='sport')
    parser.add_argument('--market',help='type of market')
    parser.add_argument('--margin',help='type of sport')
    parser.add_argument('--all_markets',help='Test through all markets',action='store_true')
    parser.add_argument('--all_tests',help='Use all available tests',action='store_true')
    parser.add_argument('--kl',help='Include kl divergence test',action='store_true')
    parser.add_argument('--simple',help='Include simple tests',action='store_true')
    parser.add_argument('--random',help='Include random_tests',action='store_true')
    parser.add_argument('--devide',help='Include devide bet',action='store_true')
    parser.add_argument('--regression',help='Include regression tests',action='store_true')
    args = vars(parser.parse_args())
    main(args)"""

