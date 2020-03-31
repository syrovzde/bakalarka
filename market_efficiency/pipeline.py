import market_efficiency.main
import pandas as pd
from sqlalchemy import create_engine
import time


def load_data(dtb_url, schema, table_name):
    conn = create_engine(dtb_url)
    select_script = 'SELECT "1", "X", "2", "Result","Bookmaker" FROM {y}."Matches" INNER JOIN {y}."Odds_1x2" on "Odds_1x2"."MatchID" = "Matches"."MatchID" LIMIT 1000000 '.format(
        y=schema)
    print(select_script)
    # df = pd.read_sql_table(table_name='Matches',schema='football',con=conn)
    # print()
    start = time.time()
    df = pd.read_sql_query(select_script, con=conn)
    df = df.dropna()
    print(time.time() - start)
    print(df['Bookmaker'])
    df['results'] = market_efficiency.main.getResults(df['Result'])
    return df


if __name__ == '__main__':
    load_data("postgresql+pg8000://postgres:1234@localhost:5432/tmp", schema='football', table_name=None)
