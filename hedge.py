from datetime import date
from typing import Dict

import numpy as np
import pandas as pd
from hqb.testers.single_factor.groups import SimpleGroupsTester
from hqb.testers.single_factor.tester import Tester
from hqd.decorator import timeit


class HedgeTester(Tester):
    def __init__(self) -> None:
        pass
    
    def _get_hedged_daily_df(self,
        name: str,
        long_daily_return: pd.Series,
        short_daily_return: pd.Series,
        trading_execution_dates: np.ndarray,
        next_trading_execution_dates: np.ndarray) -> pd.DataFrame:
        hedged_daily_compound_net = pd.Series(index=long_daily_return.index, data=np.repeat(1, long_daily_return.size))
        for trading_execution_date, next_trading_execution_date in zip(trading_execution_dates, next_trading_execution_dates):
            long_period_return = long_daily_return[trading_execution_date:next_trading_execution_date][1:]
            short_period_return = short_daily_return[trading_execution_date:next_trading_execution_date][1:]
            # Reset period start net to previous period net on the first day of each period.
            # Long short returns are cumulated within each period and reset at start of each period.
            hedged_daily_compound_net[(hedged_daily_compound_net.index > trading_execution_date) & (hedged_daily_compound_net.index <= next_trading_execution_date)] = \
                hedged_daily_compound_net[trading_execution_date] * (1 + ((long_period_return + 1).cumprod() - (short_period_return + 1).cumprod()))
        hedged_daily_return = np.concatenate(([0], (hedged_daily_compound_net.diff() / hedged_daily_compound_net.shift(periods=1))[1:]))
        hedged_daily_sum_net = hedged_daily_return.cumsum() + 1
        return pd.DataFrame(index=long_daily_return.index, data={
            "{}_compound_net".format(name): hedged_daily_compound_net.values,
            "{}_sum_net".format(name): hedged_daily_sum_net,
            "{}_return".format(name): hedged_daily_return
        })
    
    def _get_hedged_period_df(self,
        name: str,
        long_period_return: pd.Series,
        short_period_return: pd.Series) -> pd.DataFrame:
        hedged_period_return = long_period_return - short_period_return
        return pd.DataFrame(index=long_period_return.index, data={
            "{}_compound_net".format(name): (hedged_period_return + 1).cumprod(),
            "{}_sum_net".format(name): hedged_period_return.cumsum() + 1,
            "{}_return".format(name): hedged_period_return
        })
    
    def _get_hedged_monthly_df(self, monthly_trading_execution_dates, monthly_next_trading_execution_dates, hedged_daily_return: pd.Series) -> pd.DataFrame:
        monthly_return = pd.Series(index=monthly_next_trading_execution_dates[:-1], data=np.repeat(0, monthly_next_trading_execution_dates.size - 1))
        for trading_execution_date, next_trading_execution_date in zip(monthly_trading_execution_dates[:-1], monthly_next_trading_execution_dates[:-1]):
            monthly_return.loc[next_trading_execution_date] = (hedged_daily_return[trading_execution_date:next_trading_execution_date][1:] + 1).prod() - 1
        return pd.DataFrame(index=monthly_return.index, data={
            hedged_daily_return.name: monthly_return
        })


class GroupHedgeTester(HedgeTester):
    class Result(Tester.Result):
        def __init__(self, 
                daily_df: pd.DataFrame, 
                period_df: pd.DataFrame, 
                monthly_df: pd.DataFrame,
                annulized_group_hedge_return_mean: float,
                annulized_group_hedge_return_std: float,
                annulized_group_hedge_sharpe_ratio: float) -> None:
            self.daily_df = daily_df
            self.period_df = period_df
            self.monthly_df = monthly_df
            self.annulized_group_hedge_return_mean = annulized_group_hedge_return_mean
            self.annulized_group_hedge_return_std = annulized_group_hedge_return_std
            self.annulized_group_hedge_sharpe_ratio = annulized_group_hedge_sharpe_ratio
            
    def __init__(self,
        trading_execution_dates: np.ndarray,
        next_trading_execution_dates: np.ndarray,
        monthly_trading_execution_dates: np.ndarray,
        monthly_next_trading_execution_dates: np.ndarray,
        groups_result: SimpleGroupsTester.Result,
        factor_ic_is_positive: bool,
        trading_frequency: str) -> None:
        self.trading_execution_dates = trading_execution_dates
        self.next_trading_execution_dates = next_trading_execution_dates
        self.monthly_trading_execution_dates = monthly_trading_execution_dates
        self.monthly_next_trading_execution_dates = monthly_next_trading_execution_dates
        self.groups_result = groups_result
        self.factor_ic_is_positive = factor_ic_is_positive
        self.factor_first_not_nan_date = groups_result.factor_first_not_nan_date
        self.trading_frequency = trading_frequency

    @timeit
    def run(self) -> Result:
        # Calculate daily and period numbers independently for easier cross check.
        daily_df = self.__get_daily_df()
        period_df = self.__get_period_df()
        monthly_df = self.__get_monthly_df(daily_df)
        return GroupHedgeTester.Result(daily_df, period_df, monthly_df, 
            *self._calculate_annulized_return_stats(period_df.loc[period_df.index >= self.factor_first_not_nan_date, "group_hedge_return"].values, self.trading_frequency))

    def __get_daily_df(self):
        if self.factor_ic_is_positive:
            long_daily_return = self.groups_result.daily_df.loc[:,"group_{}_return".format(self.groups_result.number_of_groups - 1)]
            short_daily_return = self.groups_result.daily_df.loc[:,"group_0_return"]
        else:
            long_daily_return = self.groups_result.daily_df.loc[:,"group_0_return"]
            short_daily_return = self.groups_result.daily_df.loc[:,"group_{}_return".format(self.groups_result.number_of_groups - 1)]
        hedged_daily_df = self._get_hedged_daily_df(
            "group_hedge", long_daily_return, short_daily_return, self.trading_execution_dates, self.next_trading_execution_dates)
        return hedged_daily_df
    
    def __get_period_df(self):
        if self.factor_ic_is_positive:
            long_period_return = self.groups_result.period_df.loc[:,"group_{}_return".format(self.groups_result.number_of_groups - 1)]
            short_period_return = self.groups_result.period_df.loc[:,"group_0_return"]
        else:
            long_period_return = self.groups_result.period_df.loc[:,"group_0_return"]
            short_period_return = self.groups_result.period_df.loc[:,"group_{}_return".format(self.groups_result.number_of_groups - 1)]
        hedged_period_df = self._get_hedged_period_df("group_hedge", long_period_return, short_period_return)
        return hedged_period_df

    def __get_monthly_df(self, daily_df: pd.DataFrame):
        return self._get_hedged_monthly_df(self.monthly_trading_execution_dates, self.monthly_next_trading_execution_dates, daily_df.loc[:,"group_hedge_return"])

class IndexHedgeTester(HedgeTester):
    class Result(Tester.Result):
        def __init__(self, 
                daily_df: pd.DataFrame, 
                period_df: pd.DataFrame,
                monthly_df: pd.DataFrame, 
                annulized_top_hedge_hs300_return_mean: float,
                annulized_top_hedge_hs300_return_std: float,
                annulized_top_hedge_hs300_sharpe_ratio: float,
                annulized_top_hedge_csi500_return_mean: float,
                annulized_top_hedge_csi500_return_std: float,
                annulized_top_hedge_csi500_sharpe_ratio: float) -> None:
            self.daily_df = daily_df
            self.period_df = period_df
            self.monthly_df = monthly_df
            self.annulized_top_hedge_hs300_return_mean = annulized_top_hedge_hs300_return_mean
            self.annulized_top_hedge_hs300_return_std = annulized_top_hedge_hs300_return_std
            self.annulized_top_hedge_hs300_sharpe_ratio = annulized_top_hedge_hs300_sharpe_ratio
            self.annulized_top_hedge_csi500_return_mean = annulized_top_hedge_csi500_return_mean
            self.annulized_top_hedge_csi500_return_std = annulized_top_hedge_csi500_return_std
            self.annulized_top_hedge_csi500_sharpe_ratio = annulized_top_hedge_csi500_sharpe_ratio
            
    def __init__(self,
        trading_execution_dates: pd.DataFrame,
        next_trading_execution_dates: pd.DataFrame,
        monthly_trading_execution_dates: np.ndarray,
        monthly_next_trading_execution_dates: np.ndarray,
        groups_result: SimpleGroupsTester.Result,
        factor_ic_is_positive: bool,
        trading_frequency: str,
        index_daily_returns: Dict[str, pd.Series]) -> None:
        self.trading_execution_dates = trading_execution_dates
        self.next_trading_execution_dates = next_trading_execution_dates
        self.monthly_trading_execution_dates = monthly_trading_execution_dates
        self.monthly_next_trading_execution_dates = monthly_next_trading_execution_dates
        self.groups_result = groups_result
        self.factor_ic_is_positive = factor_ic_is_positive
        self.factor_first_not_nan_date = groups_result.factor_first_not_nan_date
        self.trading_frequency = trading_frequency
        self.index_daily_returns = {}
        for index_name, index_daily_return in index_daily_returns.items():
            filtered_index_daily_return = index_daily_return.copy()
            filtered_index_daily_return[filtered_index_daily_return.index <= self.factor_first_not_nan_date] = 0
            self.index_daily_returns[index_name] = filtered_index_daily_return

    @timeit
    def run(self) -> Result:
        # Calculate daily and period numbers independently for easier cross check.
        daily_df = self.__get_daily_df()
        period_df = self.__get_period_df()
        monthly_df = self.__get_monthly_df(daily_df)
        return IndexHedgeTester.Result(daily_df, period_df, monthly_df,
            *self._calculate_annulized_return_stats(period_df.loc[period_df.index >= self.factor_first_not_nan_date, "top_hedge_HS300_return"].values, self.trading_frequency),
            *self._calculate_annulized_return_stats(period_df.loc[period_df.index >= self.factor_first_not_nan_date, "top_hedge_CSI500_return"].values, self.trading_frequency))

    def __get_daily_df(self):
        if self.factor_ic_is_positive:
            long_daily_return = self.groups_result.daily_df.loc[:,"group_{}_return".format(self.groups_result.number_of_groups - 1)]
        else:
            long_daily_return = self.groups_result.daily_df.loc[:,"group_0_return"]
        daily_df_list = []
        for index_name in self.index_daily_returns.keys():
            index_daily_return = self.index_daily_returns[index_name]
            hedged_daily_df = self._get_hedged_daily_df(
                "top_hedge_{}".format(index_name), long_daily_return, index_daily_return, self.trading_execution_dates, self.next_trading_execution_dates)
            hedged_daily_df.loc[:,"{}_return".format(index_name)] = index_daily_return
            hedged_daily_df.loc[:,"{}_compound_net".format(index_name)] = np.cumprod(index_daily_return + 1)
            hedged_daily_df.loc[:,"{}_sum_net".format(index_name)] = np.cumsum(index_daily_return) + 1
            daily_df_list.append(hedged_daily_df)
        daily_df = pd.concat(daily_df_list, axis=1)
        return daily_df
    
    def __get_period_df(self):
        if self.factor_ic_is_positive:
            long_period_return = self.groups_result.period_df.loc[:,"group_{}_return".format(self.groups_result.number_of_groups - 1)]
        else:
            long_period_return = self.groups_result.period_df.loc[:,"group_0_return"]
        period_df_list = []
        for index_name in self.index_daily_returns.keys():
            index_daily_return = self.index_daily_returns[index_name]
            index_period_return = pd.Series(index=long_period_return.index, data=np.repeat(0, long_period_return.size))
            for i in range(1, long_period_return.size):
                period_returns = index_daily_return[long_period_return.index[i-1]:long_period_return.index[i]][1:]
                index_period_return.loc[long_period_return.index[i]] = (period_returns + 1).prod() - 1
            hedged_period_df = self._get_hedged_period_df("top_hedge_{}".format(index_name), long_period_return, index_period_return)
            period_df_list.append(hedged_period_df)
        period_df = pd.concat(period_df_list, axis=1)
        return period_df
    
    def __get_monthly_df(self, daily_df: pd.DataFrame):
        monthly_df_list = []
        for index_name in self.index_daily_returns.keys():
            monthly_df_list.append(self._get_hedged_monthly_df(self.monthly_trading_execution_dates, self.monthly_next_trading_execution_dates, daily_df.loc[:,"top_hedge_{}_return".format(index_name)]))
        period_df = pd.concat(monthly_df_list, axis=1)
        return period_df
