from __future__ import annotations

import abc
import multiprocessing as mp
import threading
from collections import deque
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
import pandas as pd
from hqb.pipeline.single_factor.back_test_result_aggregator import SingleFactorBackTestResultAggregator
from hqb.pipeline.single_factor.pipeline_graph import SingleFactorPipelineFactorRepository
from hqb.testers.single_factor.coverage import FactorCoverageTestor
from hqb.testers.single_factor.groups import SimpleGroupsTester
from hqb.testers.single_factor.hedge import GroupHedgeTester, IndexHedgeTester
from hqb.testers.single_factor.icir import ICIRTester
from hqd.decorator import after_init, before_init, cache_return
from hqd.universe import Universe
from hqf.correlation.factors_correlation import FactorsCorrelationCaculator
from hqf.factors.factor import Factor
from hqf.factors.generator import FactorGenerator
from hqf.processors.neutralizer import LinearNeutralizer
from hqf.processors.processor import Processor
from hqf.store.single_factor.factor_correlation_store import SingleFactorsCorrelationStore
from hqs.trading_strategy_config import TradingStrategyConfig

if TYPE_CHECKING:
    from hqb.pipeline.single_factor.back_test_pipeline_identifier import SingleFactorBackTestPipelineABCIdentifier
    from hqb.store.single_factor.factor_test_result_store import SingleFactorTestResultStore
    from hqf.store.single_factor.factor_raw_value_store import SingleFactorRawValueStore
    from hqf.store.single_factor.factor_value_store import SingleFactorValueStore


class SingleFactorBackTestPipelineABC(abc.ABC):
    class FactorTestResult:
        def __init__(self,
                factor: Factor,
                coverage_result: FactorCoverageTestor.Result,
                icir_result: ICIRTester.Result,
                groups_result: SimpleGroupsTester.Result,
                group_hedge_result: GroupHedgeTester.Result,
                index_hedge_result: IndexHedgeTester.Result) -> None:
            self.factor = factor
            self.coverage_result = coverage_result
            self.icir_result = icir_result
            self.groups_result = groups_result
            self.group_hedge_result = group_hedge_result
            self.index_hedge_result = index_hedge_result

    def __init__(self,
        name: str,
        upstream_pipelines: List,
        universe: Universe,
        trading_strategy_config: TradingStrategyConfig,
        price_type: str, weight_type: str, groups_count: int,
        processors: List[Processor], parallel_processes: int,
        factor_raw_value_store: SingleFactorRawValueStore,
        factor_value_store: SingleFactorValueStore,
        factor_test_result_store: SingleFactorTestResultStore,
        factor_correlation_store: SingleFactorsCorrelationStore) -> None:
        self.name = name
        self.upstream_pipelines = upstream_pipelines
        self.universe = universe
        self.trading_strategy_config = trading_strategy_config
        self.monthly_trading_strategy_config = TradingStrategyConfig(universe, trading_frequency="monthly")
        self.price_type = price_type
        self.price_df = self.__get_price_df()
        self.weight_type = weight_type
        self.weight_df = self.__get_weight_df()
        self.groups_count = groups_count
        self.index_names = ["HS300", "CSI500"]
        self.index_daily_returns = self.__get_index_daily_return()
        self.processors = processors
        self.parallel_processes = parallel_processes
        self.factor_raw_value_store = factor_raw_value_store
        self.factor_value_store = factor_value_store
        self.factor_test_result_store = factor_test_result_store
        self.factor_correlation_store = factor_correlation_store
        self.factor_repository = SingleFactorPipelineFactorRepository(self)
        self.factors = deque[Factor]()
        self.processed_factors = []
        # Maximize main process usage. Let it produce factor df in advance before all worker processes return.
        # At the same time, adding semaphore can prevent OOM when the factor list is large.
        # Because the most factor df count in memory at the same time will only be worker processes count * 2
        self.semaphore = threading.Semaphore(parallel_processes * 2)
        self.initialized = False
        
    def __get_price_df(self) -> pd.DataFrame:
        if self.price_type == "adjusted_vwap":
            return self.universe.wind_common.get_adjusted_vwap_na_filled()
        elif self.price_type == "adjusted_open":
            return self.universe.wind_common.get_adjusted_open_price_na_filled()
        elif self.price_type == "adjusted_close":
            return self.universe.wind_common.get_adjusted_close_price_na_filled()
        elif self.price_type == "adjusted_high":
            return self.universe.wind_common.get_adjusted_high_price_na_filled()
        elif self.price_type == "adjusted_low":
            return self.universe.wind_common.get_adjusted_low_price_na_filled()
    
    def __get_weight_df(self) -> pd.DataFrame:
        if self.weight_type == "equal_weight":
            return self.universe.wind_common.get_equal_weight_df()
        elif self.weight_type == "total_market_value":
            return self.universe.wind_common.get_total_market_value_na_filled()
        elif self.weight_type == "free_circulation_market_value":
            return self.universe.wind_common.get_free_circulation_market_value_na_filled()

    def __get_index_daily_return(self):
        index_daily_returns = {}
        for index_name in self.index_names:
            index_daily_return = self.universe.align_df_with_trade_dates(self.universe.wind_common.index_daily_returns[index_name]).iloc[:,0]
            index_daily_returns[index_name] = index_daily_return
        return index_daily_returns

    @before_init
    def add_factor(self, factor: Factor):
        self.factors.append(factor)

    @before_init
    def add_factor_generator(self, factor_generator: FactorGenerator):
        for factor in factor_generator:
            self.factors.append(factor)

    def init(self):
        self.initialized = True

    @after_init
    def run(self, backtest_result_aggregator: SingleFactorBackTestResultAggregator):
        pool = mp.Pool(processes=self.parallel_processes)
        for factor in self.factors:
            self.__add_task_in_pool(factor, pool, backtest_result_aggregator)
        pool.close()
        pool.join()
    
    @cache_return("factors_correlation")
    def get_factors_correlation(self):
        # NOTE. Only call this method on the last pipeline in the graph to save calculation effort.
        if not self.factor_correlation_store.has_correlation_df():
            correlation_df = FactorsCorrelationCaculator(self).get_correlation_df()
            self.factor_correlation_store.save_correlation_df(correlation_df)
            return correlation_df
        else:
            return self.factor_correlation_store.get_correlation_df()
        
    def __add_task_in_pool(self, factor: Factor, pool: mp.Pool, backtest_result_aggregator: SingleFactorBackTestResultAggregator):
        self.semaphore.acquire()
        factor_raw_df, need_to_save_factor_raw_df = self.__get_factor_raw_df(factor)
        if need_to_save_factor_raw_df:
            self.__save_factor_raw_df(factor, factor_raw_df)
        factor_df, neutralizer_parameter_coefficients_df, need_to_save_factor_df = self.__get_factor_df(factor, factor_raw_df)
        if need_to_save_factor_df:
            self.__save_factor_df(factor, factor_df, neutralizer_parameter_coefficients_df)
        if self.factor_test_result_store.has_factor_test_result(factor):
            backtest_result_aggregator.add_factor_results(neutralizer_parameter_coefficients_df, self.factor_test_result_store.get_factor_test_result(factor))
            factor.release_resouces() # NOTE. Release resouce to avoid OOM.
            self.semaphore.release()
            return
        context = self.__create_single_factor_test_context(factor_df, factor_raw_df)
        pool.apply_async(
            SingleFactorBackTestPipelineABC.test_single_factor, 
            args=(context,), 
            callback=SingleFactorBackTestPipelineABC.create_result_callback(self, backtest_result_aggregator, factor, neutralizer_parameter_coefficients_df))

    def __save_factor_raw_df(self,
        factor: Factor,
        factor_raw_df: pd.DataFrame):
        self.factor_raw_value_store.save_factor_raw_df(factor, factor_raw_df)

    def __save_factor_df(self,
        factor: Factor,
        factor_df: pd.DataFrame,
        neutralizer_parameter_coefficients_df: pd.DataFrame):
        self.factor_value_store.save_factor_df(factor, factor_df,neutralizer_parameter_coefficients_df)

    def __get_factor_raw_df(self, factor: Factor) -> Tuple[pd.DataFrame, bool]:
        if self.factor_raw_value_store.has_factor_raw_df(factor):
            return self.factor_raw_value_store.get_factor_raw_df(factor), False
        else:
            return factor.do_init().get_df(), True

    def __get_factor_df(self, factor: Factor, factor_raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
        if self.factor_value_store.has_factor_df(factor):
            return *self.factor_value_store.get_factor_df(factor, return_neutralizer_parameter_coefficients=True), False
        else:
            neutralizer_parameter_coefficients_df = pd.DataFrame()
            processed_df = factor_raw_df.loc[self.trading_strategy_config.factor_calculation_dates].copy()
            for processor in self.processors:
                processed_df, intermediate_status = processor.process(processed_df)
                if isinstance(processor, LinearNeutralizer):
                    neutralizer_parameter_coefficients_df = intermediate_status.parameter_coefficients
            return processed_df, neutralizer_parameter_coefficients_df, True
    
    def __create_single_factor_test_context(self,
        factor_df: pd.DataFrame,
        factor_raw_df: pd.DataFrame) -> dict:
        # Note: Creating context is the key to enable multiprocess parallel tests execution.
        # Because Python multiprocess sharing only allows picklable objects to be passed as parameter to sub-processes.
        # So it means our tester classes should only rely on simple objects as parameter, e.g. built-in types, numpy array and DataFrame.
        # The universe object and other external api related objects are not picklable. So we should avoid referencing to them in testers.
        # We need to follow this rule strickly to prevent from breaking the parallel execution model.
        context = {}
        context["tradable_symbols"] = self.universe.get_tradable_symbols()
        context["trade_dates"] = self.universe.get_trade_dates()
        context["factor_calculation_dates"] = self.trading_strategy_config.factor_calculation_dates.values
        context["trading_execution_dates"] = self.trading_strategy_config.trading_execution_dates.values
        context["next_trading_execution_dates"] = np.concatenate((self.trading_strategy_config.trading_execution_dates[1:].values, self.universe.get_trade_dates()[-1:]))
        context["monthly_trading_execution_dates"] = self.monthly_trading_strategy_config.factor_calculation_dates.values
        context["monthly_next_trading_execution_dates"] = self.monthly_trading_strategy_config.trading_execution_dates.values
        context["factor_df"] = factor_df
        context["factor_raw_df"] = factor_raw_df
        context["price_df"] = self.price_df
        context["weight_df"] = self.weight_df
        context["trading_frequency"] = self.trading_strategy_config.trading_frequency
        context["number_of_groups"] = self.groups_count
        context["index_daily_returns"] = self.index_daily_returns
        return context

    def test_single_factor(context: Dict):
        tradable_symbols = context["tradable_symbols"]
        trade_dates = context["trade_dates"]
        factor_calculation_dates = context["factor_calculation_dates"]
        trading_execution_dates = context["trading_execution_dates"]
        next_trading_execution_dates = context["next_trading_execution_dates"]
        monthly_trading_execution_dates = context["monthly_trading_execution_dates"]
        monthly_next_trading_execution_dates = context["monthly_next_trading_execution_dates"]
        factor_df = context["factor_df"]
        factor_raw_df = context["factor_raw_df"]
        price_df = context["price_df"]
        weight_df = context["weight_df"]
        trading_frequency = context["trading_frequency"]
        number_of_groups = context["number_of_groups"]
        index_daily_returns = context["index_daily_returns"]
        
        coverage_result = FactorCoverageTestor(tradable_symbols, factor_calculation_dates, factor_raw_df).run()
        icir_result = ICIRTester(trade_dates, factor_calculation_dates, trading_execution_dates, factor_df, price_df, trading_frequency).run()
        groups_result = SimpleGroupsTester(
            trade_dates, 
            tradable_symbols.columns.values, 
            factor_calculation_dates,
            trading_execution_dates, 
            next_trading_execution_dates,
            factor_df,
            price_df,
            weight_df,
            number_of_groups,
            icir_result.ic_mean > 0).run()
        group_hedge_result = GroupHedgeTester(
            trading_execution_dates, 
            next_trading_execution_dates, 
            monthly_trading_execution_dates, 
            monthly_next_trading_execution_dates,
            groups_result,
            icir_result.ic_mean > 0,
            trading_frequency).run()
        index_hedge_result = IndexHedgeTester(
            trading_execution_dates, 
            next_trading_execution_dates, 
            monthly_trading_execution_dates, 
            monthly_next_trading_execution_dates,
            groups_result,
            icir_result.ic_mean > 0,
            trading_frequency,
            index_daily_returns).run()
        
        results = {
            "coverage_result": coverage_result,
            "icir_result": icir_result,
            "groups_result": groups_result,
            "group_hedge_result": group_hedge_result,
            "index_hedge_result": index_hedge_result   
        }
        return results

    def create_result_callback(single_factor_backtest_pipeline,
            backtest_result_aggregator: SingleFactorBackTestResultAggregator, 
            factor: Factor, 
            neutralizer_parameter_coefficients_df: pd.DataFrame):
        # Note: Pool callback only takes return value from task function as parameter.
        # But here we need few more external parameter to call result aggregator.
        # So use inclosure here to pass the external paramter.
        def add_test_result(results: dict):
            single_factor_backtest_pipeline.__save_factor_test_result(
                backtest_result_aggregator,
                factor, 
                neutralizer_parameter_coefficients_df, 
                results["coverage_result"], 
                results["icir_result"], 
                results["groups_result"], 
                results["group_hedge_result"], 
                results["index_hedge_result"])
            factor.release_resouces() # NOTE. Release resouce to avoid OOM.
            single_factor_backtest_pipeline.semaphore.release()
        return add_test_result
    
    def __save_factor_test_result(self,
            backtest_result_aggregator: SingleFactorBackTestResultAggregator,
            factor: Factor,
            neutralizer_parameter_coefficients_df: pd.DataFrame,
            coverage_result: FactorCoverageTestor.Result,
            icir_result: ICIRTester.Result,
            groups_result: SimpleGroupsTester.Result,
            group_hedge_result: GroupHedgeTester.Result,
            index_hedge_result: IndexHedgeTester.Result):
        factor_test_result = SingleFactorBackTestPipelineABC.FactorTestResult(
            factor, coverage_result, icir_result, groups_result, group_hedge_result, index_hedge_result)
        self.factor_test_result_store.save_factor_test_result(factor, factor_test_result)
        backtest_result_aggregator.add_factor_results(neutralizer_parameter_coefficients_df, factor_test_result)

    @cache_return("identifier")
    def get_identifier(self) -> SingleFactorBackTestPipelineABCIdentifier:
        from hqb.pipeline.single_factor.back_test_pipeline_identifier import \
            SingleFactorBackTestPipelineABCIdentifier
        return SingleFactorBackTestPipelineABCIdentifier(self)
    
    @cache_return("identifier_dict")
    def get_identifier_dict(self) -> dict:
        return self.get_identifier().to_dict()
