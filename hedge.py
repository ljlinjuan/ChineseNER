import logging
import numpy as np
import pandas as pd
from chameqs_common.backtest.common.hedge_caculator import get_hedged_nav
from chameqs_common.backtest.portfolio.common.return_store import PortfolioReturnStore
from chameqs_common.backtest.portfolio.single_portfolio.stocks.store.hedge_store import StocksPortfolioHedgeStore
from chameqs_common.backtest.portfolio.single_portfolio.stocks.tester.base import StocksPortfolioTester
from chameqs_common.backtest.common.mode import BacktestMode
from chameqs_common.benchmark.store.benchmark_store import BenchmarkStore
from chameqs_common.config.trading_strategy_config import TradingStrategyConfig
from chameqs_common.data.universe import Universe
from chameqs_common.data.wind import INDEX_CODE_MAP
from chameqs_common.portfolio.common.common import get_factor_calculation_start_date


class StocksPortfolioHedgeTester(StocksPortfolioTester):
    def __init__(self, 
            universe: Universe, 
            trading_strategy_config: TradingStrategyConfig, 
            hedge_cost_rate: str,
            portfolio_return_store: PortfolioReturnStore,
            portfolio_hedge_store: StocksPortfolioHedgeStore,
            benchmark_store: BenchmarkStore,
            hedge_index_names : list[str] = INDEX_CODE_MAP.keys()):
        super().__init__(universe, trading_strategy_config, {
            "Trading Frequency": trading_strategy_config.trading_frequency,
            "Hedge Cost Rate": hedge_cost_rate
            })
        self.hedge_cost_rate = hedge_cost_rate
        self.portfolio_return_store = portfolio_return_store
        self.portfolio_hedge_store = portfolio_hedge_store
        self.benchmark_store = benchmark_store
        self.hedge_index_names = hedge_index_names
        self.index_daily_close = self.__get_indexes_daily_close()

    def __get_indexes_daily_close(self) -> dict[str, pd.Series]:
        indexes_daily_close = {}
        for index_name in self.hedge_index_names:
            if index_name in INDEX_CODE_MAP.keys():
                indexes_daily_close[index_name] = self.universe.wind_common.get_index_daily_close_by_name(index_name)
            else:
                indexes_daily_close[index_name] = self.benchmark_store.get_benchmark_return(index_name)
        return indexes_daily_close

    def run(self, portfolio_meta: dict, backtest_namespace: str, portfolio_weights: pd.DataFrame) -> None:
        portfolio_namespace = portfolio_meta["portfolio_namespace"]
        portfolio_name = portfolio_meta["portfolio_name"]
        if "bench_index_names" in portfolio_meta:
            bench_index_names = portfolio_meta["bench_index_names"]
        else:
            bench_index_names = self.hedge_index_names
        self.__do_calculate(portfolio_namespace, portfolio_name, backtest_namespace, bench_index_names, portfolio_weights, BacktestMode.TOTAL)
        self.__do_calculate(portfolio_namespace, portfolio_name, backtest_namespace, bench_index_names, portfolio_weights, BacktestMode.IN_SAMPLE)
        self.__do_calculate(portfolio_namespace, portfolio_name, backtest_namespace, bench_index_names, portfolio_weights, BacktestMode.OUT_OF_SAMPLE)
    
    def __do_calculate(self, 
            portfolio_namespace: str, 
            portfolio_name: str, 
            backtest_namespace: str, 
            bench_index_names: list[str], 
            portfolio_weights: pd.DataFrame,
            mode: BacktestMode):
        trade_dates, backtest_factor_calculation_dates, trading_execution_dates = \
            self.__get_calculation_context(mode)
        self.portfolio_hedge_store.delete_portfolio(portfolio_namespace, portfolio_name, mode, backtest_namespace)
        portfolio_return = self.portfolio_return_store.get_portfolio_return(portfolio_namespace, portfolio_name, backtest_namespace, mode)
        index_hedge_compound_net = self.__get_index_hedge_compound_nav(
            trade_dates, backtest_factor_calculation_dates, trading_execution_dates,
            bench_index_names, portfolio_return, portfolio_weights)
        index_hedge_return = self.__get_index_hedge_return(index_hedge_compound_net)
        index_hedge_sum_net = self.__get_index_hedge_sum_net(index_hedge_return)
        self.__save_portfolio_hedge(
            portfolio_namespace, portfolio_name, backtest_namespace, portfolio_return.index, 
            index_hedge_return, index_hedge_sum_net, index_hedge_compound_net, mode)

    def __get_calculation_context(self, mode: BacktestMode):
        if mode == BacktestMode.TOTAL:
            trade_dates = self.trade_dates
            backtest_factor_calculation_dates = self.back_test_factor_calculation_dates
            trading_execution_dates = self.trading_execution_dates
        elif mode == BacktestMode.IN_SAMPLE:
            trade_dates = self.in_sample_trade_dates
            backtest_factor_calculation_dates = self.in_sample_back_test_factor_calculation_dates
            trading_execution_dates = self.in_sample_trading_execution_dates
        elif mode == BacktestMode.OUT_OF_SAMPLE:
            trade_dates = self.out_of_sample_trade_dates
            backtest_factor_calculation_dates = self.out_of_sample_back_test_factor_calculation_dates
            trading_execution_dates = self.out_of_sample_trading_execution_dates
        else:
            raise Exception(f"Unknown mode: {mode}")
        return trade_dates, backtest_factor_calculation_dates, trading_execution_dates

    def __get_index_hedge_compound_nav(self, 
            trade_dates: np.ndarray,
            backtest_factor_calculation_dates: np.ndarray,
            trading_execution_dates: np.ndarray, 
            bench_index_names: list[str], 
            portfolio_return: pd.DataFrame, 
            portfolio_weights: pd.DataFrame) -> dict[str, pd.Series]:
        long_only_compound_nets = portfolio_return["compound_net"]
        calculation_start_date = get_factor_calculation_start_date(backtest_factor_calculation_dates, portfolio_weights)
        valid_bench_index_names = self.__get_valid_bench_index_names(bench_index_names, calculation_start_date)
        index_hedge_compound_nav = {}
        for index_name in valid_bench_index_names:    
            index_hedge_compound_nav[index_name] = get_hedged_nav(
                trade_dates, trading_execution_dates, 
                long_only_compound_nets, self.index_daily_close[index_name], self.hedge_cost_rate)
        return index_hedge_compound_nav

    def __get_valid_bench_index_names(self, bench_index_names: list[str], calculation_start_date: int) -> list[str]:
        valid_bench_index_names = []
        for index_name in bench_index_names:
            index_start_date = self.index_daily_close[index_name].index[0]
            if index_start_date <= calculation_start_date:
                valid_bench_index_names.append(index_name)
            else:
                logging.info(f"Index: {index_name} is not valid for single factor hedge backtest, because the index start date {index_start_date} is later than portfolio start date: {calculation_start_date}")
        return valid_bench_index_names

    def __get_index_hedge_return(self, index_hedge_compount_nav: dict[str, pd.Series]) -> dict[str, pd.Series]:
        index_hedge_return = {}
        for index_name in index_hedge_compount_nav:
            index_hedge_return[index_name] = index_hedge_compount_nav[index_name].diff() / index_hedge_compount_nav[index_name].shift(periods=1)
            index_hedge_return[index_name] = index_hedge_return[index_name].fillna(0)
        return index_hedge_return
    
    def __get_index_hedge_sum_net(self, index_hedge_return: dict[str, pd.Series]) -> dict[str, pd.Series]:
        index_sum_nav = {}
        for index_name in index_hedge_return:
            index_sum_nav[index_name] = index_hedge_return[index_name].cumsum() + 1
        return index_sum_nav

    def __save_portfolio_hedge(self, 
            portfolio_namespace: str, portfolio_name: str, backtest_namespace: str, dates: np.ndarray, 
            portfolio_hedge_return: dict[str, pd.Series], portfolio_hedge_sum_net: dict[str, pd.Series], 
            portfolio_hedge_component_net: dict[str, pd.Series], mode: BacktestMode):
        batch = []
        for date in dates:
            document = {
                "portfolio_namespace": portfolio_namespace,
                "portfolio_name": portfolio_name,
                "backtest_namespace": backtest_namespace,
                "date": date
            }
            for index_name in portfolio_hedge_return:
                document[f"{index_name.lower()}_hedge_return"] = portfolio_hedge_return[index_name][date]
                document[f"{index_name.lower()}_hedge_sum_net"] = portfolio_hedge_sum_net[index_name][date]
                document[f"{index_name.lower()}_hedge_compount_net"] = portfolio_hedge_component_net[index_name][date]
            batch.append(document)
        self.portfolio_hedge_store.save_portfolio_hedge(batch, mode)
