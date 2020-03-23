import datetime
import sys

import numba
import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy.dialects.postgresql import insert


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


def shin_probabilities(odds):
    p = 1 / odds

    p_plus = np.sum(p, 1)
    p_minus = p[:, 0] - p[:, 1]

    z = ((p_plus - 1) * (p_minus ** 2 - p_plus)) / (p_plus * (p_minus ** 2 - 1))

    p1 = p[:, 0]

    shin_probs = (np.sqrt(z ** 2 + 4 * (1 - z) * ((p1 ** 2) / np.sum(p, 1))) - z) / (2 * (1 - z))

    return np.column_stack([shin_probs, 1 - shin_probs])


def upsert_table(df, table_name, db, schema, update_on_conflict=False, update_cols=None, unique_cols=None):
    if df.empty:
        return

    try:
        df.to_sql(table_name, db, index=False, schema=schema)  # try to create nonexistent table and set it up first
        if not unique_cols:
            unique_cols = '","'.join(list(df.columns))
        db.execute(
            'ALTER TABLE "' + schema + '"."' + table_name + '" ADD CONSTRAINT ' + table_name.lower() + '_unique_rows UNIQUE("' + unique_cols + '");')
        return
    except Exception as e:
        pass  # table already exists, just "upsert" it

    database = pd.io.sql.pandasSQL_builder(db, schema=schema)

    sql_table = pd.io.sql.SQLTable(table_name, database, frame=df, index=False, if_exists='fail', prefix='pandas',
                                   index_label=None, schema=None, keys=None, dtype=None)

    keys, data_list = sql_table.insert_data()
    data = [{k: v for k, v in zip(keys, row)} for row in zip(*data_list)]

    stmt = insert(sql_table.table, values=data)
    if update_on_conflict:
        excluded = dict(stmt.excluded)
        to_be_updated = {col: excluded[col] for col in update_cols}
        stmt = stmt.on_conflict_do_update(constraint=f'{table_name.lower()}_unique_rows', set_=to_be_updated)
    else:
        stmt = stmt.on_conflict_do_nothing()

    try:
        db.execute(stmt)
    except sqlalchemy.exc.ProgrammingError as e:
        pass
    except Exception as e:
        print("SQL upsert failed: " + str(e), file=sys.stderr)
        pass


@numba.guvectorize(['float64, float64, float64[:,:], float64, float64[:]'], '(),(),(n,m),()->()', nopython=True)
def _interpolate_2d(x, y, grid, step, res):
    x = x / step
    y = y / step

    x0 = int(x)
    x1 = x0 + 1
    y0 = int(y)
    y1 = y0 + 1

    Ia = grid[x0, y0]
    Ib = grid[x0, y1]
    Ic = grid[x1, y0]
    Id = grid[x1, y1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    res[0] = wa * Ia + wb * Ib + wc * Ic + wd * Id


def interpolate_2d(x, y, grid, step):
    x = x / step
    y = y / step

    x0 = int(x)
    x1 = x0 + 1
    y0 = int(y)
    y1 = y0 + 1

    Ia = grid[x0, y0]
    Ib = grid[x0, y1]
    Ic = grid[x1, y0]
    Id = grid[x1, y1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id
