import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import scipy.stats as stats
default_select_script = 'SELECT * FROM  public."matches"'
default_dtb_url = "postgresql+pg8000://postgres:1234@localhost:5432/bettors"


def load_data(dtb_url=None,csv_file = None,select_script=None):
    global df
    if csv_file is not None:
        df = pd.read_csv(csv_file)
    else:
        if dtb_url is None:
            dtb_url = default_dtb_url
        conn = create_engine(dtb_url)
        if select_script is None:
            select_script = default_select_script
        df = pd.read_sql_query(select_script, con=conn)
    print()
    return df.groupby("Name")

def count_deposit(group):
    positive = group[group['Amount_got'] > 0]
    negative = group[group['Amount_got'] < 0]
    negative_tmp= negative['Amount_got'].abs()
    positive_tmp = (positive['Amount_got']+1)/positive['Odds']
    pom = positive_tmp.sum() + negative_tmp.sum()
    return pom

def count_roi(group):
    deposit = count_deposit(group)
    return (group['Amount_got'].sum())/deposit + 1


def parse_bettor(name,group):
    print(group['Amount_got'])
    #print(name + " mean of odds is "  + str(group["Odds"].mean()) + " and amount won is " + str(group['Amount_got'].mean()) + " sample size is "
    #     + str(group['Odds'].count()))
    return

def t_test(name,group):
    roi = count_roi(group)
    average_odds = group['Odds'].mean()
    sigma = np.sqrt((roi * (average_odds - roi)))
    if np.isnan(sigma) or sigma == 0:
        return None,None,None,None
    count = group['Odds'].count()
    t = np.sqrt(count) * (roi - 1) / sigma
    p = 1 - stats.t.cdf(abs(t), count)
    return p,t,count,average_odds

def parse_data(groups):
    for name,group in groups:
        p,t,count,average_odds = t_test(name,group)
        if p is None:
            continue
        average_odds = np.around(average_odds,decimals=3)
        p  = np.around(p,decimals=3)
        t = np.around(t,decimals=3)
        if p< 0.01:
            print("we reject the null hypothesis bettor shows some skill and average odd were {k}".format(k=group['Odds'].count()))
        else:
            print("null hypothesis cannot be rejected. Bettor seems to have ROI 1 or "
                  "lower on our significance level and average odds were {k}".format(k=group["Odds"].count()))
    return None

"""
Starting point of program. T-test on all the data available in bettors.csv
"""
if __name__ == '__main__':
    df = load_data(csv_file="bettors.csv")
    parse_data(df)