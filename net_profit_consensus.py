import os
from typing import Tuple

import numpy as np
import pandas as pd
from chameqs_common.data.query import Query
from chameqs_common.data.universe import Universe
from chameqs_common.factor.single_factor.factor_definition.single_factor import SingleFactor, SingleFactorsFactory
from chameqs_common.utils.decorator import timeit
from chameqs_red.data.fundamental.fundamental_preprocessor import ConsensusNetProfitProcessor, FundamentalHonestNetprofit
from chameqs_red.data.fundamental.fundamental_data_utils import *
from sklearn import preprocessing


class NetProfitDerivativeIndicatorsFactory(SingleFactorsFactory):
    def __init__(self, universe: Universe, net_profit_type: list[str], indicator: list[str], indicator_mode: list[str]):
        super().__init__()
        self.__prepare_common_data(universe)
        self.parameters_combination = self._gen_parameters_iter_product(
            net_profit_type=net_profit_type, indicator=indicator, indicator_mode=indicator_mode)
        self.current_params = next(self.parameters_combination, False)

    def __prepare_common_data(self, universe: Universe):
        con = ConsensusNetProfitProcessor(universe, field="forecast_np", parallel_processes=5)
        funda = FundamentalHonestNetprofit(universe)
        net_profit, np_period, _ = funda.get_fundamental_honest_netprofit(deal_profit_notice="avg")

        con_data = con.get_consensus_data()
        con_data = con.get_consensus_np_mapped(con_data)
        con_data["con_date"] = [int(i.strftime("%Y%m%d")) for i in con_data["con_date"]]
        con_data["stock_code"] = con_data["stock_code"].astype(int)
        con_np_fy0 = con.get_consensus_data_transform(con_data, field="forecast_np", fy=0)  # .fillna(method="pad")
        con_np_fy1 = con.get_consensus_data_transform(con_data, field="forecast_np", fy=1)  # .fillna(method="pad")
        con_np_fy2 = con.get_consensus_data_transform(con_data, field="forecast_np", fy=2)  # .fillna(method="pad")

        self.universe = universe
        self.np_period = np_period
        self.con_np = (con_np_fy0, con_np_fy1, con_np_fy2)

    def _has_next_factor(self) -> bool:
        return self.current_params

    def _get_next_factor(self) -> SingleFactor:
        factor = NetProfitDerivativeIndicatorsFactor(
            self.universe, self.np_period, self.con_np, **self.current_params)
        self.current_params = next(self.parameters_combination, False)
        return factor


class NetProfitDerivativeIndicatorsFactor(SingleFactor):
    def __init__(self, universe: Universe, np_period: pd.DataFrame,
                 con_np: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
                 net_profit_type: str, indicator: str, indicator_mode: str):
        super().__init__(f"net_profit_{net_profit_type}_{indicator}_{indicator_mode}",
                         [f"NetProfitDerivativeIndicators{indicator_mode}"])
        self.universe = universe
        self.np_period = np_period
        self.con_np = con_np
        self.net_profit_type = net_profit_type
        self.indicator = indicator
        self.indicator_mode = indicator_mode

    def get_df(self, universe: Universe, start_date: int = None, inclusive: bool = False) -> pd.DataFrame:
        # 1) Select net_profit_type
        if self.net_profit_type == "Consensus":
            net_profit = self.calc_con_np_roll()
        else:
            raise Warning("!!! Waring: NetProfitDerivativeIndicatorsFactor get unexpect net_profit_type")
        # 2) Select indicator
        if self.indicator == "NP":
            indicator_value = net_profit
        elif self.indicator == "ROE":
            equity = Query().from_("wind", "asharebalancesheet").\
                on("statement_type").equals("408001000").on("ann_dt").larger_than(universe.window_buffer_from_date). \
                on("s_info_windcode").in_query(universe.wind_common.wind_codes_query). \
                select("actual_ann_dt", "ann_dt", "report_period", "s_info_windcode", "tot_shrhldr_eqy_excl_min_int").df()
            equity = equity[~(equity["actual_ann_dt"] > equity["ann_dt"])]
            equity = fundamental_drop_duplicates(equity, ["s_info_windcode", "ann_dt"], "report_period")
            equity = fundamental_pivot(equity, values="tot_shrhldr_eqy_excl_min_int", columns="s_info_windcode", index="ann_dt")
            equity = self.universe.align_df_with_trade_dates_and_symbols(equity)
            indicator_value = net_profit / equity
        elif self.indicator == "ROA":
            asset = self.universe.get_transform("wind", "ashareeodderivativeindicator", "net_assets_today")
            indicator_value = net_profit / asset
        elif self.indicator == "EP":
            cap = self.universe.wind_common.get_total_market_value_na_filled()
            indicator_value = net_profit / cap
        else:
            raise Warning("!!! Waring: NetProfitDerivativeIndicatorsFactor get unexpect indicator")

        # 3) Select indicator_mode
        if self.indicator_mode == "Value":
            factor = indicator_value
        elif self.indicator_mode == "YOY":
            factor = self.calc_indicator_yoy_chg(indicator_value)
        elif self.indicator_mode == "LP":
            factor = self.calc_indicator_lp_chg(indicator_value)
        elif self.indicator_mode == "SUE":
            factor = self.calc_indicator_sue(indicator_value, self.np_period)

        # exp_data_path = f"D:\\Data\\health_check\\{universe.name}"
        # if not os.path.exists(exp_data_path):
        #     os.mkdir(exp_data_path)
        # exp = self.universe.align_df_with_trade_dates(factor)
        # exp.columns = exp.columns.astype(str)
        # exp.to_parquet(os.path.join(exp_data_path, f"net_profit_{self.net_profit_type}_{self.indicator}_{self.indicator_mode}"))
        # Check non-nan value
        factor[[(~np.isnan(factor)).sum(1) < 100]] = np.nan
        return self.universe.align_df_with_trade_dates(factor)

    def calc_con_np_roll(self):
        con_np_fy0, con_np_fy1, con_np_fy2 = self.con_np
        mon_sig = pd.Series([(i % 10000) // 100 for i in con_np_fy0.index], index=con_np_fy0.index)
        # mon_sig = pd.Series([i.month for i in con_np_fy0.index], index=con_np_fy0.index)
        mon_sig1 = mon_sig.copy(deep=True)
        mon_sig1[mon_sig1 <= 4] = 1
        mon_sig1[mon_sig1 > 4] = 0

        mon_sig2 = mon_sig.copy(deep=True)
        mon_sig2[mon_sig2 <= 4] = 0
        mon_sig2[mon_sig2 > 4] = 1

        np_s1 = (con_np_fy1.T * mon_sig1.values + con_np_fy0.T * mon_sig2.values).T
        np_s2 = (con_np_fy2.T * mon_sig1.values + con_np_fy1.T * mon_sig2.values).T
        con_np_roll = (np_s1.T * (1 - mon_sig.values / 12).T + np_s2.T * (mon_sig.values / 12).T).T

        trade_dates = self.universe.get_trade_dates_with_window_buffer()
        con_np_roll = con_np_roll.reindex(trade_dates)

        return self.universe.align_df_with_symbols(con_np_roll)  # .fillna(method="pad")

    def calc_indicator_last_period(self, indicator, periods=4):
        np_period_val = self.np_period.values
        seasonl_indicator = self.calc_seasonal_indicator(indicator, self.np_period)
        # np_last_year
        indicator_last_year = np.zeros([indicator.shape[0], indicator.shape[1]])
        seasonl_indicator = seasonl_indicator.fillna(0)
        seasonl_indicator_val = seasonl_indicator.values
        for i in range(len(seasonl_indicator.index) - periods):
            this_season = seasonl_indicator.index[i + periods]
            indicator_last_year += seasonl_indicator_val[i, :] * (np_period_val == this_season)
        indicator_last_year[indicator_last_year == 0] = np.nan
        return indicator_last_year

    def calc_seasonal_indicator(self, indicator, np_period):
        seasonl_end = np_period - np_period.shift(-1) < 0
        seasonl_end = seasonl_end.reindex(indicator.index)
        seasonl_end.iloc[-1, :] = True
        indicator = indicator.fillna(method="pad")
        indicator_val = indicator.values
        np_period_val = np_period.values
        seasonl_end_val = seasonl_end.values
        # historical data
        seasonl_indicator = {}
        for s in range(len(self.universe.get_symbols().columns)):
            seasonl_end_mark = seasonl_end_val[:, s]
            seasonl_indicator[s] = pd.Series(indicator_val[seasonl_end_mark, s],
                                             index=np_period_val[seasonl_end_mark, s])
        seasonl_indicator = pd.DataFrame(seasonl_indicator)
        return seasonl_indicator

    def calc_indicator_yoy_chg(self, indicator):
        indicator_yoy = self.calc_indicator_last_period(indicator, 4)

        indicator_yoy_chg = indicator / indicator_yoy
        indicator_yoy_chg[np.isinf(indicator_yoy_chg)] = np.nan
        return indicator_yoy_chg

    def calc_indicator_lp_chg(self, indicator):
        indicator_last_season = self.calc_indicator_last_period(indicator, 1)
        indicator_cagr = indicator / indicator_last_season
        indicator_cagr[np.isinf(indicator_cagr)] = np.nan
        return indicator_cagr

    def calc_indicator_percentile(self, indicator):
        indicator.groupby(pd.PeriodIndex(indicator['Date'], freq="M"))['Value'].mean()

    def calc_indicator_sue(self, factor, np_period):
        seasonal_np = self.calc_seasonal_indicator(factor, np_period)
        delta = ((seasonal_np - seasonal_np.shift(4)).fillna(0).rolling(8).mean())
        sigma = (np.sqrt((((seasonal_np - seasonal_np.shift(4))
                           .fillna(0) - delta) ** 2).rolling(8).sum()) / 7).shift(1)
        delta = delta.shift(1)
        expect_np = (seasonal_np.shift(4) + delta)

        # expand seasonal indicator
        expect_np_expand = np.zeros([factor.shape[0], factor.shape[1]])
        sigma_expand = np.zeros([factor.shape[0], factor.shape[1]])
        expect_np_val = expect_np.fillna(0).values
        sigma_val = sigma.fillna(0).values
        np_period_val = np_period.values

        for i in range(8, len(expect_np.index)):
            this_season = expect_np.index[i]
            expect_np_expand += expect_np_val[i, :] * (np_period_val == this_season)
            sigma_expand += sigma_val[i, :] * (np_period_val == this_season)

        expect_np_expand[expect_np_expand == 0] = np.nan
        sigma_expand[sigma_expand == 0] = np.nan

        sue = (factor - expect_np_expand) / sigma_expand
        return sue
