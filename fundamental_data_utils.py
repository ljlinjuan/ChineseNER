import numpy as np
import pandas as pd
from chameqs_common.utils.date_util import date_to_int


def fundamental_drop_duplicates(df: pd.DataFrame, col_sort: list, col_keep_last: str) -> pd.DataFrame:
    df = df.sort_values(col_sort + [col_keep_last])
    df = df.drop_duplicates(col_sort, keep='last')
    return df


def fundamental_pivot(df: pd.DataFrame, values: str, columns: str, index: str, fillna=True) -> pd.DataFrame:
    df = df.pivot(values=values, columns=columns, index=index)
    if fillna:
        df = df.asfreq("D").fillna(method="pad")
    df.columns = df.columns.astype(int)
    df.index = [date_to_int(i) for i in df.index]
    return df.fillna(method="pad")

