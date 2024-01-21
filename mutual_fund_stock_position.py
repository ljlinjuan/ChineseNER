import os
from typing import Tuple

import numpy as np
import pandas as pd
import datetime
from chameqs_common.data.query import Query
from chameqs_common.data.universe import Universe
from chameqs_common.utils.date_util import int_to_date
from chameqs_common.factor.single_factor.factor_definition.single_factor import SingleFactor, SingleFactorsFactory
from chameqs_red.factor.single_factor.factor_definition.mutual_fund import mutual_fund_holdings_data as mf_data


class MutualFundHoldingsFactory(SingleFactorsFactory):
    def __init__(self, universe: Universe, processor: list[str], q: list[float], window=list[int]):
        super().__init__()
        self.__prepare_common_data(universe, q)
        self.parameters_combination = self._gen_parameters_iter_product(
            processor=processor, q=q, window=window)
        self.current_params = next(self.parameters_combination, False)

    def __prepare_common_data(self, universe: Universe, q):
        # Fund_style: {2001010101000000:普通股票型基,
        #              2001010201000000: 偏股混合型基金,
        #              2001010202000000:平衡混合型基金,
        #              200101020400000:灵活配置型基金}

        fund_list = Query().from_("wind", "chinamutualfundsector").on("s_info_sector")\
            .in_([2001010101000000, 2001010201000000, 2001010202000000, 200101020400000])\
            .select("distinct(f_info_windcode) as f_info_windcode").df()
        fund_list = fund_list["f_info_windcode"].apply(lambda x:x.split(".")[0]).to_list()

        trade_dates_with_window_buffer = universe.get_trade_dates_with_window_buffer()
        df = Query().from_("wind", "chinamutualfundstockportfolio").\
                on("ann_date").range(int(trade_dates_with_window_buffer[0]), int(trade_dates_with_window_buffer[-1])).on("s_info_windcode").in_(fund_list). \
                order("ann_date", "asc").\
                order("s_info_windcode", "asc").\
                select("ann_date, f_prt_enddate, f_prt_stkvaluetonav, "
                       "s_info_stockwindcode, s_info_windcode").df()
        df = df[df["s_info_stockwindcode"].apply(lambda x: x.split(".")[1] in ["SZ", "SH"])]
        df["s_info_stockwindcode"] = df["s_info_stockwindcode"].apply(lambda x: int(x.split(".")[0]))

        df[["f_prt_stkvaluetonav"]] = df[["f_prt_stkvaluetonav"]].astype(float)
        df = df.rename(columns={"f_prt_stkvaluetonav": "values"})
        tot_weight, tot_weight_ana_date = mf_data.get_total_stock_portfolio_weight(df)
        first_ann_port = mf_data.get_top10_stock_portfolio_weight(df)
        total_stock_portfolio_weight_composite = mf_data.get_total_stock_portfolio_weight_composite(tot_weight, tot_weight_ana_date, first_ann_port.drop_duplicates())

        mutual_fund_stock_position_data = {}
        for each_q in q:
            factor = mf_data.get_mutual_fund_stock_position_quantile(universe, total_stock_portfolio_weight_composite, each_q)
            factor = universe.align_df_with_trade_dates_with_window_buffer_and_symbols(factor)
            mutual_fund_stock_position_data[each_q] = factor

        self.universe = universe
        self.mutual_fund_stock_position_data = mutual_fund_stock_position_data

    def _has_next_factor(self) -> bool:
        return self.current_params

    def _get_next_factor(self) -> SingleFactor:
        factor = MutualFundHoldingsFactor(
            self.universe, self.mutual_fund_stock_position_data, **self.current_params)
        self.current_params = next(self.parameters_combination, False)
        return factor


class MutualFundHoldingsFactor(SingleFactor):
    def __init__(self, universe: Universe, mutual_fund_stock_position_data: dict[pd.DataFrame],
                 processor: str, q: float, window: int):
        super().__init__(f"mutualfundholdings_holdingratio_{q}_{processor}_{window}",
                         [f"MutualFundHoldings"])
        self.universe = universe
        self.mutual_fund_stock_position_data = mutual_fund_stock_position_data[q]
        self.processor = processor
        self.window = window

    def get_df(self, universe: Universe, start_date: int = None, inclusive: bool = False) -> pd.DataFrame:
        factor = self.mutual_fund_stock_position_data
        if self.processor == "original":
            factor = np.log(factor)
        elif self.processor == "relative":
            free_mkv = universe.wind_common.get_free_circulation_market_value_na_filled() * 10000
            factor = np.log(factor) - np.log((free_mkv.T / free_mkv.sum(1).values).T)
            factor = universe.align_df_with_trade_dates_with_window_buffer_and_symbols(factor)
        else:
            raise Warning("Warning!!! mutual_fund_stock_mkv_quantile get unexpected parameter 'processor'")

        return factor.ewm(self.window).mean()

