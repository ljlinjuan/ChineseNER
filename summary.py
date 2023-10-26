import pandas as pd
from chameqs_common.backtest.portfolio.common.return_store import PortfolioReturnStore
from chameqs_common.backtest.portfolio.common.summary_store import PortfolioSummaryStore
from chameqs_common.backtest.portfolio.single_portfolio.stocks.store.hedge_store import StocksPortfolioHedgeStore
from chameqs_common.backtest.portfolio.single_portfolio.stocks.tester.base import StocksPortfolioTester
from chameqs_common.backtest.common.mode import BacktestMode
from chameqs_common.config.trading_strategy_config import TradingStrategyConfig
from chameqs_common.data.universe import Universe
from chameqs_common.backtest.common.performance_summary import generate_performance_summary


class StocksPortfolioSummaryTester(StocksPortfolioTester):
    def __init__(self, 
            universe: Universe, 
            trading_strategy_config: TradingStrategyConfig, 
            portfolio_return_store: PortfolioReturnStore,
            portfolio_hedge_store: StocksPortfolioHedgeStore,
            portfolio_summary_store: PortfolioSummaryStore):
        super().__init__(universe, trading_strategy_config, {
            "Trading Frequency": trading_strategy_config.trading_frequency
            })
        self.portfolio_return_store = portfolio_return_store
        self.portfolio_hedge_store = portfolio_hedge_store
        self.portfolio_summary_store = portfolio_summary_store

    def run(self, portfolio_meta: dict, backtest_namespace: str, portfolio_weights: pd.DataFrame) -> None:
        self.__do_calculate(portfolio_meta, backtest_namespace, BacktestMode.TOTAL)
        self.__do_calculate(portfolio_meta, backtest_namespace, BacktestMode.IN_SAMPLE)
        self.__do_calculate(portfolio_meta, backtest_namespace, BacktestMode.OUT_OF_SAMPLE)
    
    def __do_calculate(self, portfolio_meta: dict, backtest_namespace: str, mode: BacktestMode):
        portfolio_namespace = portfolio_meta["portfolio_namespace"]
        portfolio_name = portfolio_meta["portfolio_name"]
        portfolio_id = portfolio_meta["portfolio_id"]
        portfolio_description = portfolio_meta["portfolio_description"]
        self.portfolio_summary_store.delete_portfolio(portfolio_namespace, portfolio_name, mode, backtest_namespace)
        portfolio_return = self.portfolio_return_store.get_portfolio_return(portfolio_namespace, portfolio_name, backtest_namespace, mode)
        portfolio_performance = generate_performance_summary(portfolio_return["return"], [1, 2, 3, 4, 5, 7, 10], skip_zero_return_head=True)
        
        bench_index = None
        if "Bench Index" in portfolio_description:
            bench_index = portfolio_description["Bench Index"]
        if bench_index is not None:
            portfolio_hedge_return = self.portfolio_hedge_store.get_portfolio_hedge(portfolio_namespace, portfolio_name, backtest_namespace, mode)[bench_index.lower()]
            portfolio_hedge_performance = generate_performance_summary(portfolio_hedge_return["return"], [1, 2, 3, 4, 5, 7, 10], skip_zero_return_head=True)
        else:
            portfolio_hedge_performance = None
        self.__save_portfolio_summary(
            portfolio_namespace, portfolio_name, portfolio_id, backtest_namespace,
            portfolio_performance, portfolio_hedge_performance, bench_index, mode)

    def __save_portfolio_summary(self, 
            portfolio_namespace: str, portfolio_name: str, portfolio_id: str, backtest_namespace: str, 
            portfolio_performance: pd.DataFrame, portfolio_hedge_performance: pd.DataFrame,
            bench_index: str, mode: BacktestMode):
        document = {
            "portfolio_namespace": portfolio_namespace,
            "portfolio_name": portfolio_name,
            "portfolio_id": portfolio_id,
            "portfolio_backtest_id": portfolio_id + "_" + backtest_namespace,
            "backtest_namespace": backtest_namespace,
            }
        document["long_only"] = {}
        for row_i in range(0, portfolio_performance.shape[0]):
            for column_i in range(0, portfolio_performance.shape[1]):
                document["long_only"][f"{portfolio_performance.index[row_i]}_{portfolio_performance.columns[column_i]}"] = portfolio_performance.iloc[row_i, column_i]
        if portfolio_hedge_performance is not None:
            document["hedge_" + bench_index.lower()] = {}
            for row_i in range(0, portfolio_hedge_performance.shape[0]):
                for column_i in range(0, portfolio_hedge_performance.shape[1]):
                    document["hedge_" + bench_index.lower()][f"{portfolio_hedge_performance.index[row_i]}_{portfolio_hedge_performance.columns[column_i]}"] = portfolio_hedge_performance.iloc[row_i, column_i]
        self.portfolio_summary_store.save_portfolio_summary(document, mode)
