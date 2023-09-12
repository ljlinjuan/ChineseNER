import logging

import pandas as pd
from chameqs_common.data import query
from chameqs_common.data.calendar import get_trade_date_after
from chameqs_common.data.tradable_filter.base import TradableFilter
from chameqs_common.data.universe import Universe
from chameqs_common.utils.decorator import timeit


class STTradableFilter(TradableFilter):
    # NOTE: This is predictable! Can be applied to calculate both factor and execution tradable symbols.
    # The tradable filter is calculated on factor date and applied on execution date.
    # Because the after ST date can be forecasted. 
    # So if you want 20 days after ST to be tradable, you should set 19 days in init parameter.
    def __init__(self, after_st_trade_days: int) -> None:
        super().__init__()
        self.after_st_trade_days = after_st_trade_days

    @timeit(logging.INFO)
    def get_filter_df(self, universe: Universe) -> pd.DataFrame:
        query_str = "select toYYYYMMDD(entry_dt, 'Asia/Shanghai'), toYYYYMMDD(remove_dt, 'Asia/Shanghai'), toInt32(s_info_windcode) from wind.asharest where (isNull(remove_dt) or toYYYYMMDD(remove_dt, 'Asia/Shanghai') >= {}) {}".format(\
            get_trade_date_after(universe.from_date, self.after_st_trade_days * -1),\
            "and s_info_windcode in ({})".format(universe.wind_common.wind_codes_query) if len(universe.wind_common.wind_codes_query) > 0 else "")
        logging.debug(query_str)
        filter_df = pd.DataFrame(True, index=universe.wind_common.trade_dates, columns=universe.wind_common.wind_codes.columns)
        st_list = query.get_clickhouse_client("wind").execute(query_str)
        for st in st_list:
            entry_date = st[0]
            remove_date = st[1]
            wind_code = st[2]
            if remove_date == None:
                filter_df.loc[filter_df.index >= entry_date, wind_code] = False
            else:
                # Remove companies out of ST within some trade days
                filter_df.loc[(filter_df.index >= entry_date) & (filter_df.index <= get_trade_date_after(remove_date, self.after_st_trade_days)), wind_code] = False
        return filter_df
