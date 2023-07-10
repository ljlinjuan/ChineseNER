import pandas as pd
import numpy as np
import logging
from sklearn import preprocessing
from sklearn.base import TransformerMixin

from chameqs_common.config.trading_strategy_config import TradingStrategyConfig
from chameqs_common.data.universe import Universe
from chameqs_common.factor.single_factor.store.single_factor_source import SingleFactorSource
from chameqs_common.backtest.common.mode import BacktestMode
from chameqs_common.utils.decorator import ignore_warning
from chameqs_common.utils.date_util import int_to_date, date_to_int
from chameqs_common.utils.calculator import calc_rowwise_nan_correlation, calc_rowwise_nan_rank_correlation
from chameqs_red.factor.return_attribution.barra_pure_factor_portfolio import PureFactor
from chameqs_common.factor.single_factor.processor.filler import IndustryAndMarketMeanFiller
from chameqs_red.factor.combined_factor.models.lgbm_model.funcs import *
from chameqs_red.data.intraday.intraday_specific_period_price import load_stock_intraday_specific_period_vwap
from chameqs_red.factor.combined_factor.expect_return.excess_return_defination import ExcessReturnDefination


class CombinorCommonOper():
    def __init__(self, universe: Universe, trading_strategy_config: TradingStrategyConfig):
        self.pure_factor_portfolio_constructor = PureFactor(universe, trading_strategy_config)

        self.universe = universe
        self.symbols = self.universe.get_symbols().columns.values
        self.factor_calculation_dates = trading_strategy_config.factor_calculation_dates
        self.trading_execution_dates = trading_strategy_config.trading_execution_dates
        self.is_last_date_factor_calculation_date = trading_strategy_config.is_last_date_factor_calculation_date
        self.icir_min_rolling_window = 5
        self.icir_rolling_window = 12

    def load_all_single_factors(self, all_single_factors: dict[str, pd.DataFrame], factor_source: SingleFactorSource):
        for meta in factor_source.get_filtered_factor_meta_list():
            factor_namespace = meta["factor_namespace"]
            factor_name = meta["factor"]["name"]
            factor_type = meta["factor_type"]
            top_category = meta["factor"]["category"][0]
            logging.info(
                f"Start combine [factor names space: {factor_namespace}, factor name: {factor_name}, factor type: {factor_type}] into [top category: {top_category}]")
            factor_value_on_calculation_dates = factor_source.factor_value_store.get_factor_intermediate_status(
                factor_namespace, factor_name, factor_type, IndustryAndMarketMeanFiller,
                dates=self.factor_calculation_dates.tolist(),
                kwargs={"target_symbols": self.symbols, "all_symbols_sorted_by_list_date": self.universe.wind_common.all_symbols_sorted_by_list_date})
            all_single_factors[f"{factor_name}_{factor_type}"] = factor_value_on_calculation_dates

    def load_all_single_factors_by_category(self, all_single_factors: dict[str, pd.DataFrame], factor_source: SingleFactorSource, target_categories: list):
        for meta in factor_source.get_filtered_factor_meta_list():
            top_category = meta["factor"]["category"][0]
            if top_category not in target_categories:
                continue
            factor_namespace = meta["factor_namespace"]
            factor_name = meta["factor"]["name"]
            factor_type = meta["factor_type"]
            logging.info(
                f"Start combine [factor names space: {factor_namespace}, factor name: {factor_name}, factor type: {factor_type}] into [top category: {top_category}]")
            factor_value_on_calculation_dates = factor_source.factor_value_store.get_factor_intermediate_status(
                factor_namespace, factor_name, factor_type, IndustryAndMarketMeanFiller,
                dates=self.factor_calculation_dates.tolist(),
                kwargs={"target_symbols": self.symbols, "all_symbols_sorted_by_list_date": self.universe.wind_common.all_symbols_sorted_by_list_date})
            all_single_factors[f"{factor_name}_{factor_type}"] = factor_value_on_calculation_dates

    def calc_single_factors_rolling_icir(self, all_single_factors: dict[str, pd.DataFrame], return_df):
        all_factors_ic = {}
        for name, single_factor in all_single_factors.items():
            print(name)
            ic_this_factor = self._calculate_rolling_icir_matched_factor_calculation_dates(single_factor, return_df)
            all_factors_ic[name] = ic_this_factor
        return pd.DataFrame(all_factors_ic)

    def calc_single_factors_rolling_ic(self, all_single_factors: dict[str, pd.DataFrame], return_df):
        all_factors_ic = {}
        for name, single_factor in all_single_factors.items():
            print(name)
            ic_this_factor = self._calculate_rolling_ic_matched_factor_calculation_dates(single_factor, return_df)
            all_factors_ic[name] = ic_this_factor
        return pd.DataFrame(all_factors_ic)

    def load_layer1_combined_factors_by_icir(self, top_category_factors: dict[str, pd.DataFrame], factor_source: SingleFactorSource):
        for meta in factor_source.get_filtered_factor_meta_list():
            factor_namespace = meta["factor_namespace"]
            factor_name = meta["factor"]["name"]
            factor_type = meta["factor_type"]
            top_category = f'{meta["factor"]["category"][0]}_{factor_type}'
            logging.info(
                f"Start combine [factor names space: {factor_namespace}, factor name: {factor_name}, factor type: {factor_type}] into [top category: {top_category}]")
            if top_category not in top_category_factors:
                top_category_factors[top_category] = pd.DataFrame(0, index=self.factor_calculation_dates,
                                                                  columns=self.symbols)
            top_category_factor_df = top_category_factors[top_category]

            factor_value_on_calculation_dates = factor_source.factor_value_store.get_factor_intermediate_status(
                factor_namespace, factor_name, factor_type, IndustryAndMarketMeanFiller,
                dates=self.factor_calculation_dates.tolist(),
                kwargs={"target_symbols": self.symbols,
                        "all_symbols_sorted_by_list_date": self.universe.wind_common.all_symbols_sorted_by_list_date})

            rolling_icir_matched_calculation_dates = self._get_rolling_icir_matched_factor_calculation_dates(
                factor_source, factor_namespace, factor_name, factor_type)
            weighted_factor_value = factor_value_on_calculation_dates.multiply(rolling_icir_matched_calculation_dates,
                                                                               axis=0)
            top_category_factor_df[(np.isnan(top_category_factor_df) & ~np.isnan(weighted_factor_value))] = 0
            top_category_factor_df += weighted_factor_value
            logging.info(
                f"Done combine [factor names space: {factor_namespace}, factor name: {factor_name}, factor type: {factor_type}] into [top category: {top_category}]")

    def load_layer2_combined_factors_by_pure_icir(self, layer2_top_category_factors: dict[str, pd.DataFrame], top_category_factors: dict[str, pd.DataFrame], factors_aggregate_config: pd.DataFrame):
        trade_dates_len = len(self.factor_calculation_dates)

        for k , v in factors_aggregate_config.groupby("category"):
            combined_factor = pd.DataFrame(np.nan, index=self.factor_calculation_dates, columns=self.symbols)
            factor_names = list(v["factor_name"])
            factor_dfs = {i: top_category_factors[i] for i in factor_names}
            if len(factor_names) == 1:
                pure_factor_port_return = \
                self.pure_factor_portfolio_constructor.construct_pure_factor(factor_dfs[factor_names[0]], factor_names[0])[factor_names]
            else:
                pure_factor_port_return = self.pure_factor_portfolio_constructor.construct_list_of_pure_factors(factor_dfs, factor_names)[factor_names]
            pure_factor_port_sharpe = (pure_factor_port_return.rolling(self.icir_rolling_window, min_periods=self.icir_min_rolling_window).mean()\
                                      /pure_factor_port_return.rolling(self.icir_rolling_window, min_periods=self.icir_min_rolling_window).std()**2).shift(1).fillna(0)
            pure_factor_port_sharpe_selection = (pure_factor_port_return.rolling(trade_dates_len, min_periods=self.icir_min_rolling_window).mean()\
                                      /pure_factor_port_return.rolling(trade_dates_len, min_periods=self.icir_min_rolling_window).std()).shift(1).fillna(0)
            pure_factor_port_sharpe[(pure_factor_port_sharpe_selection.T != pure_factor_port_sharpe_selection.max(1).values).T] = 0
            pure_factor_port_sharpe[pure_factor_port_sharpe < 0] = 0
            for f in factor_names:
                logging.info(f"Start combine [top category: {f}] into expected return")
                top_category_factor = top_category_factors[f]
                weighted_factor_value = top_category_factor.multiply(pure_factor_port_sharpe[f], axis=0)
                combined_factor[(np.isnan(combined_factor) & ~np.isnan(weighted_factor_value))] = 0
                combined_factor += weighted_factor_value
                logging.info(f"Done combine [top category: {f}] into expected return")
            layer2_top_category_factors[k] = combined_factor

    def load_barra_common_factors(self, factor_names, do_transform=False):
        common_factors = {}
        if "industry" in factor_names:
            industry_symbols = self.universe.get_citics_highest_level_industry_symbols().fillna(method="ffill")
            common_factors["industry"] = industry_symbols.fillna(0).astype(int).loc[self.factor_calculation_dates]
            factor_names.remove("industry")

        for factor_name in factor_names:
            barra_df = self.universe.get_transform("barra", "am_marketdata_stock_barra_rsk",
                                                   factor_name).fillna(method="pad")
            common_factors[factor_name] = barra_df.loc[self.factor_calculation_dates]
            if do_transform:
                common_factors[factor_name] = self._transform_factor_value(barra_df.loc[self.factor_calculation_dates], preprocessing.RobustScaler())
        return common_factors

    def load_adj_intraday_period_vwap(self, specific_period_start, specific_period_end, time_bucket="30 minutes", start_date="2017-01-01", end_date="2022-06-23",
                                      ):
        vwap_930_1000 = load_stock_intraday_specific_period_vwap(time_bucket=time_bucket,
                                                                 start_date=start_date, end_date=end_date,
                                                                 specific_period_start=specific_period_start,
                                                                 specific_period_end=specific_period_end)
        period_vwap = vwap_930_1000.pivot(index="time", columns="symbol", values="period_vwap")
        period_vwap.index = [date_to_int(i.date()) for i in period_vwap.index]
        period_vwap = self.universe.align_df_with_trade_dates_and_symbols(period_vwap)
        adj_factor = self.universe.get_transform("wind", "ashareeodprices", "s_dq_adjfactor")
        adj_period_vwap = (period_vwap * adj_factor).fillna(method="pad")
        return adj_period_vwap

    def get_trading_days_in_range(self, start_date: int, end_date: int):
        return self.factor_calculation_dates[(self.factor_calculation_dates >= start_date) & (self.factor_calculation_dates <= end_date)]

    def get_stock_return(self, lag: int, price_type="adjusted_vwap", specific_period_start=None, specific_period_end=None):
        if price_type == "adjusted_intraday_vwap":
            px = self.load_adj_intraday_period_vwap(specific_period_start, specific_period_end).shift(-1) # !!!! Generate signal before start_time and do trades ASAP
        else:
            px = self.universe.wind_common.get_price_df_by_type(price_type).reindex(self.factor_calculation_dates)
        stock_return = px.shift(-lag)/px - 1
        return stock_return

    def get_stock_debarra_excess_return(self, lag: int, mode="industry") -> pd.DataFrame:
        erd = ExcessReturnDefination(self.universe)
        excess_return = erd.get_stk_excess_ret_debarra(mode)
        excess_return_nav = excess_return.fillna(0).add(1).cumprod()
        period_excess_nav = excess_return_nav.reindex(self.factor_calculation_dates)
        return (period_excess_nav.shift(-lag).fillna(method="pad") / period_excess_nav).shift(lag) -1

    def _transform_top_category_factors_value(self, top_category_factors: dict[str, pd.DataFrame], processor=preprocessing.RobustScaler()):
        for top_category in top_category_factors:
            logging.info(f"Start transform [top category: {top_category}] factor value")
            top_category_factors[top_category] = self._transform_factor_value(top_category_factors[top_category],
                                                                               processor) #preprocessing.QuantileTransformer(n_quantiles=5000, output_distribution="normal")
            logging.info(f"Done transform [top category: {top_category}] factor value")

    @ignore_warning
    def _transform_factor_value(self, factor_df: pd.DataFrame, transformer: TransformerMixin) -> pd.DataFrame:
        result_df = pd.DataFrame(np.nan, index=factor_df.index, columns=factor_df.columns)
        for date, factor_value in factor_df.iterrows():
            not_nan_indexes = np.where(~np.isnan(factor_value))[0]
            if not_nan_indexes.size > 0:
                result_df.loc[date].iloc[not_nan_indexes] = transformer.fit_transform(
                    factor_value.iloc[not_nan_indexes].values.reshape(not_nan_indexes.size, 1)).reshape(
                    not_nan_indexes.size)
        return result_df

    def _get_rolling_icir_matched_factor_calculation_dates(self, factor_source: SingleFactorSource, factor_namespace: str, factor_name: str,
                                                           factor_type: str) -> pd.Series:
        factor_ic = factor_source.factor_ic_store.get_factor_ic(factor_namespace, factor_name, factor_type,
                                                                factor_source.backtest_namespace, BacktestMode.TOTAL)
        factor_ic = self.universe.align_df_with_trade_dates(factor_ic)
        factor_ic_on_trading_execution_dates = factor_ic.loc[self.trading_execution_dates]
        factor_rolling_icir: pd.Series = factor_ic_on_trading_execution_dates.rolling(self.icir_rolling_window,
                                                                                      min_periods=self.icir_min_rolling_window).mean() / \
                                         factor_ic_on_trading_execution_dates.rolling(self.icir_rolling_window,
                                                                                      min_periods=self.icir_min_rolling_window).std()
        factor_rolling_icir_shift_to_match_factor_calculation_dates = pd.Series(index=self.factor_calculation_dates,
                                                                                data=np.nan)
        if self.is_last_date_factor_calculation_date:
            factor_rolling_icir_shift_to_match_factor_calculation_dates.iloc[1:] = factor_rolling_icir
        else:
            factor_rolling_icir_shift_to_match_factor_calculation_dates.iloc[:] = factor_rolling_icir.shift(periods=1)
        return factor_rolling_icir_shift_to_match_factor_calculation_dates

    def _calculate_rolling_icir_matched_factor_calculation_dates(self, factor_on_calculation_dates: pd.DataFrame, factor_dates_return) -> pd.Series:
        factor_dates_ic = pd.Series(index=self.factor_calculation_dates, data=calc_rowwise_nan_correlation(factor_dates_return, factor_on_calculation_dates))
        factor_dates_rolling_icir: pd.Series = factor_dates_ic.rolling(self.icir_rolling_window, min_periods=self.icir_min_rolling_window).mean()/ \
            factor_dates_ic.rolling(self.icir_rolling_window, min_periods=self.icir_min_rolling_window).std()
        return factor_dates_rolling_icir

    def _calculate_linear_combine_weight_by_icir(self, factor_on_calculation_dates: pd.DataFrame, factor_dates_return) -> pd.Series:
        factor_dates_ic = pd.Series(index=self.factor_calculation_dates, data=calc_rowwise_nan_correlation(factor_dates_return, factor_on_calculation_dates))
        factor_dates_rolling_icir0: pd.Series = factor_dates_ic.rolling(self.icir_rolling_window, min_periods=self.icir_min_rolling_window).mean()/ \
            factor_dates_ic.rolling(self.icir_rolling_window, min_periods=self.icir_min_rolling_window).std()
        factor_dates_rolling_icir1: pd.Series = factor_dates_ic.rolling(int(self.icir_rolling_window/2), min_periods=self.icir_min_rolling_window).mean()/ \
            factor_dates_ic.rolling(int(self.icir_rolling_window/2), min_periods=self.icir_min_rolling_window).std()**2
        return factor_dates_rolling_icir0+factor_dates_rolling_icir1

    def _calculate_rolling_ic_matched_factor_calculation_dates(self, factor_on_calculation_dates: pd.DataFrame, factor_dates_return) -> pd.Series:
        factor_dates_ic = pd.Series(index=self.factor_calculation_dates, data=calc_rowwise_nan_correlation(factor_dates_return, factor_on_calculation_dates))
        return factor_dates_ic

    @staticmethod
    def cut_dataframe(df: pd.DataFrame, bins: int =10):
        result = np.full(df.shape, np.nan)
        for i in range(df.shape[0]):
            x = df.iloc[i, :]
            not_nan = ~np.isnan(x)
            if not_nan.sum() > 0:
                result[i, not_nan] = pd.cut(x[not_nan] , bins, labels=False)
        return pd.DataFrame(result, index=df.index, columns=df.columns)
