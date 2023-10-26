import numpy as np
import pandas as pd
from chameqs_common.backtest.common.mode import BacktestMode
from chameqs_common.backtest.portfolio.common.return_store import PortfolioReturnStore
from chameqs_common.backtest.portfolio.single_portfolio.stocks.tester.base import StocksPortfolioTester
from chameqs_common.config.trading_strategy_config import TradingStrategyConfig
from chameqs_common.data.universe import Universe
from chameqs_common.portfolio.common.common import get_factor_calculation_start_date


class StocksPortfolioReturnTester(StocksPortfolioTester):
    def __init__(self, 
            universe: Universe, 
            trading_strategy_config: TradingStrategyConfig, 
            execution_price_type: str,
            valuation_price_type: str, 
            init_balance: float,
            trading_buy_cost_rate: float, 
            trading_sell_cost_rate: float, 
            portfolio_return_store: PortfolioReturnStore):
        super().__init__(universe, trading_strategy_config, {
            "Trading Frequency": trading_strategy_config.trading_frequency,
            "Execution Price Type": execution_price_type,
            "Valuation Price Type": valuation_price_type,
            "Init Balance": init_balance,
            "Trading Buy Cost Rate": trading_buy_cost_rate,
            "Trading Sell Cost Rate": trading_sell_cost_rate
            })
        self.execution_price_type = execution_price_type
        self.valuation_price_type = valuation_price_type
        self.init_balance = init_balance
        self.trading_buy_cost_rate = trading_buy_cost_rate
        self.trading_sell_cost_rate = trading_sell_cost_rate
        self.portfolio_return_store = portfolio_return_store
        self.initialized = False

    def run(self, portfolio_meta: dict, backtest_namespace: str, portfolio_weights: pd.DataFrame):
        portfolio_namespace = portfolio_meta["portfolio_namespace"]
        portfolio_name = portfolio_meta["portfolio_name"]
        self.__do_init()
        self.__do_calculation(portfolio_namespace, portfolio_name, backtest_namespace, portfolio_weights)

    def __do_init(self):
        if not self.initialized:
            self.executon_prices = self.universe.wind_common.get_price_df_by_type(self.execution_price_type)
            self.valuation_prices = self.universe.wind_common.get_price_df_by_type(self.valuation_price_type)
            self.initialized = True

    def __do_calculation(self, 
            portfolio_namespace: str,
            portfolio_name: str, 
            backtest_namespace: str, 
            portfolio_weights: pd.DataFrame):
        self.portfolio_return_store.delete_portfolio(portfolio_namespace, portfolio_name, BacktestMode.TOTAL, backtest_namespace)
        self.portfolio_return_store.delete_portfolio(portfolio_namespace, portfolio_name, BacktestMode.IN_SAMPLE, backtest_namespace)
        self.portfolio_return_store.delete_portfolio(portfolio_namespace, portfolio_name, BacktestMode.OUT_OF_SAMPLE, backtest_namespace)
        
        portfolio_total_returns, portfolio_turnover_ratio, portfolio_cash_balance_ratio = self.__get_portfolio_total_returns(portfolio_weights)
        portfolio_total_sum_net = portfolio_total_returns.cumsum() + 1
        portfolio_total_compound_net = (portfolio_total_returns + 1).cumprod()
        
        portfolio_in_sample_returns = portfolio_total_returns.loc[self.in_sample_trade_dates]
        portfolio_in_sample_sum_net = portfolio_in_sample_returns.cumsum() + 1
        portfolio_in_sample_compound_net = (portfolio_in_sample_returns + 1).cumprod()
        portfolio_in_sample_turnover_ratio = portfolio_turnover_ratio.loc[self.trading_execution_dates[(self.trading_execution_dates >= self.in_sample_trade_dates[0]) & (self.trading_execution_dates <= self.in_sample_trade_dates[-1])]]
        portfolio_in_sample_cash_balance_ratio = portfolio_cash_balance_ratio.loc[self.in_sample_trade_dates]

        portfolio_out_of_sample_returns = portfolio_total_returns.loc[self.out_of_sample_trade_dates]
        portfolio_out_of_sample_sum_net = portfolio_out_of_sample_returns.cumsum() + 1
        portfolio_out_of_sample_compound_net = (portfolio_out_of_sample_returns + 1).cumprod()
        portfolio_out_of_sample_turnover_ratio = portfolio_turnover_ratio.loc[self.trading_execution_dates[(self.trading_execution_dates >= self.out_of_sample_trade_dates[0]) & (self.trading_execution_dates <= self.out_of_sample_trade_dates[-1])]]
        portfolio_out_of_sample_cash_balance_ratio = portfolio_cash_balance_ratio.loc[self.out_of_sample_trade_dates]
        
        self.__save_portfolio_returns(
            portfolio_namespace, portfolio_name, backtest_namespace, portfolio_total_returns.index, 
            portfolio_total_returns, portfolio_total_sum_net, portfolio_total_compound_net, 
            portfolio_turnover_ratio, portfolio_cash_balance_ratio, BacktestMode.TOTAL)
        self.__save_portfolio_returns(
            portfolio_namespace, portfolio_name, backtest_namespace, portfolio_in_sample_returns.index, 
            portfolio_in_sample_returns, portfolio_in_sample_sum_net, portfolio_in_sample_compound_net, 
            portfolio_in_sample_turnover_ratio, portfolio_in_sample_cash_balance_ratio, BacktestMode.IN_SAMPLE)
        self.__save_portfolio_returns(
            portfolio_namespace, portfolio_name, backtest_namespace, portfolio_out_of_sample_returns.index, 
            portfolio_out_of_sample_returns, portfolio_out_of_sample_sum_net, portfolio_out_of_sample_compound_net, 
            portfolio_out_of_sample_turnover_ratio, portfolio_out_of_sample_cash_balance_ratio, BacktestMode.OUT_OF_SAMPLE)

    def __get_portfolio_total_returns(self, portfolio_weights: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        trade_dates = self.trade_dates
        back_test_factor_calculation_dates = self.back_test_factor_calculation_dates
        trading_execution_dates = self.trading_execution_dates
        calculation_start_date = get_factor_calculation_start_date(back_test_factor_calculation_dates, portfolio_weights)
        portfolio_weights_sum = portfolio_weights.sum(axis=1)
        # NOTE. Handle float rounding problem.
        if np.any(portfolio_weights_sum > 1.0001):
            raise Exception("Portfolio weights sum is larger than 1.0")
        
        filtered_trade_dates = trade_dates[trade_dates > calculation_start_date]
        portfolio_cash_balance = pd.Series(index=trade_dates, data=np.nan)
        portfolio_shares = pd.DataFrame(index=trade_dates, columns=portfolio_weights.columns, data=0.0)
        portfolio_total_balance = pd.Series(index=trade_dates, data=np.nan)
        portfolio_turnover_ratio = pd.Series(index=trading_execution_dates, data=np.nan)
        previous_date = trade_dates[trade_dates <= calculation_start_date][-1]
        portfolio_cash_balance.loc[previous_date] = self.init_balance
        portfolio_total_balance.loc[previous_date] = self.init_balance
        
        for date in filtered_trade_dates:
            if date in trading_execution_dates:
                previous_shares_balance = np.nansum(portfolio_shares.loc[previous_date] * self.executon_prices.loc[date])
                execution_balance = portfolio_cash_balance.loc[previous_date] + previous_shares_balance
                if execution_balance < 0:
                    portfolio_shares.loc[date] = 0
                    portfolio_cash_balance.loc[date] = 0
                else:
                    # NOTE. Allow not full capital positions.
                    target_shares = np.floor(execution_balance * portfolio_weights.loc[previous_date] / self.executon_prices.loc[date])
                    target_shares_balance = np.nansum(target_shares * self.executon_prices.loc[date])
                    portfolio_shares.loc[date] = target_shares
                    delta_shares = portfolio_shares.loc[date] - portfolio_shares.loc[previous_date]
                    delta_buy_shares = delta_shares[delta_shares > 0]
                    delta_sell_shares = delta_shares[delta_shares < 0]
                    delta_buy_value = np.nansum(np.abs(delta_buy_shares) * self.executon_prices.loc[date][delta_buy_shares.index])
                    delta_sell_value = np.nansum(np.abs(delta_sell_shares) * self.executon_prices.loc[date][delta_sell_shares.index])
                    delta_value = delta_buy_value + delta_sell_value
                    trading_buy_cost = delta_buy_value * self.trading_buy_cost_rate
                    trading_sell_cost = delta_sell_value * self.trading_sell_cost_rate
                    portfolio_cash_balance.loc[date] = portfolio_cash_balance.loc[previous_date] - (target_shares_balance - previous_shares_balance) - trading_buy_cost - trading_sell_cost
                    portfolio_turnover_ratio.loc[date] = delta_value / execution_balance
            else:
                portfolio_cash_balance.loc[date] = portfolio_cash_balance.loc[previous_date]
                portfolio_shares.loc[date] = portfolio_shares.loc[previous_date]
            portfolio_total_balance.loc[date] = portfolio_cash_balance.loc[date] + np.nansum(portfolio_shares.loc[date] * self.valuation_prices.loc[date])
            previous_date = date
        portfolio_return = portfolio_total_balance.diff() / portfolio_total_balance.shift(periods=1)
        portfolio_return = portfolio_return.fillna(0)
        portfolio_cash_balance_ratio = portfolio_cash_balance / portfolio_total_balance
        return portfolio_return, portfolio_turnover_ratio, portfolio_cash_balance_ratio
    
    def __save_portfolio_returns(self, 
            portfolio_namespace: str,
            portfolio_name: str, 
            backtest_namespace: str, 
            dates: np.ndarray, 
            portfolio_returns: pd.Series, 
            portfolio_sum_net: pd.Series, 
            portfolio_compound_net: pd.Series,
            portfolio_turnover_ratio: pd.Series,
            portfolio_cash_balance_ratio: pd.Series,
            mode: BacktestMode) -> None:
        return_batch = []
        for date in dates:
            return_batch.append({
                "portfolio_namespace": portfolio_namespace,
                "portfolio_name": portfolio_name,
                "backtest_namespace": backtest_namespace,
                "date": date,
                "return": portfolio_returns[date],
                "sum_net": portfolio_sum_net[date],
                "compound_net": portfolio_compound_net[date],
                "cash_balance_ratio": portfolio_cash_balance_ratio[date]
            })
        turnover_batch = []
        for date in portfolio_turnover_ratio.index:
            turnover_batch.append({
                "portfolio_namespace": portfolio_namespace,
                "portfolio_name": portfolio_name,
                "backtest_namespace": backtest_namespace,
                "date": date,
                "turnover": portfolio_turnover_ratio[date]
            })
        self.portfolio_return_store.save_portfolio_return(return_batch, turnover_batch, mode)