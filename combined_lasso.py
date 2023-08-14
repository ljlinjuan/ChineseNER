import abc
import logging

import numpy as np
import pandas as pd
import os
from chameqs_red import asgl
from chameqs_common.backtest.common.mode import BacktestMode
from chameqs_common.config.trading_strategy_config import TradingStrategyConfig
from chameqs_common.data.universe import Universe
from chameqs_common.factor.combined_factor.factor_definition.combined_factor import CombinedFactor
from chameqs_common.factor.single_factor.processor.filler import IndustryAndMarketMeanFiller
from chameqs_common.factor.single_factor.store.single_factor_source import SingleFactorSource
from chameqs_red.factor.return_attribution.barra_pure_factor_portfolio import PureFactor
from chameqs_common.utils.calculator import calc_rowwise_nan_correlation, calc_rowwise_nan_rank_correlation
from sklearn import preprocessing
from sklearn.base import TransformerMixin
from chameqs_red.factor.combined_factor.combined_1000.common_oper import CombinorCommonOper


class TopCategoryCombinedFactorWithIC(CombinedFactor):
    def __init__(self,
                 name: str, universe: Universe, trading_strategy_config: TradingStrategyConfig, price_type: str,
                 factor_sources: list[SingleFactorSource], icir_rolling_window: int,
                 icir_min_rolling_window: int) -> None:
        super().__init__(
            name, universe, trading_strategy_config,
            {"Price Type": price_type, "ICIR Rolling Window": icir_rolling_window,
             "ICIR Min Rolling Window": icir_min_rolling_window})
        self.common_oper = CombinorCommonOper(universe, trading_strategy_config)
        self.price_df = self.universe.wind_common.get_price_df_by_type(price_type)
        self.factor_sources = factor_sources
        self.icir_rolling_window = icir_rolling_window
        self.icir_min_rolling_window = icir_min_rolling_window
        self.common_oper = CombinorCommonOper(universe, trading_strategy_config)
        self.pure_factor_portfolio_constructor = PureFactor(universe, trading_strategy_config)
        self.barra_style_factors = np.array([]) # "beta", "leverage", "resvol", "sizenl"
        self.lag = 5
        self.stock_return = self.common_oper.get_stock_return(lag=self.lag, price_type="adjusted_vwap")

    def gen_stock_period_returns(self, period: int):
        price = self.universe.wind_common.get_price_df_by_type("adjusted_vwap")
        factor_dates_price = price.loc[self.factor_calculation_dates]
        return (factor_dates_price.shift(-period) / factor_dates_price) -1

    def get_df(self, last_date=None) -> pd.DataFrame:
        top_category_factors = {}

        top_category_factors_icir: dict[str, pd.DataFrame] = {}
        for factor_source in self.factor_sources:
            factors_stats_filterd = self.filt_single_factor_stats(factor_source, 0.5)
            self.__get_top_category_factor_value_on_factor_calcualtion_dates(top_category_factors_icir, factor_source, factors_stats_filterd)
        for k, v in top_category_factors_icir.items():
            top_category_factors[f"{k}_icir"] = v
        self.__transform_top_category_factors_value(top_category_factors)
        common_factors = self.common_oper.load_barra_common_factors(self.barra_style_factors)
        for each_f, each_fv in common_factors.items():
            common_factor_this = each_fv
        #     common_factor_this = self.common_oper._transform_factor_value(each_fv, preprocessing.QuantileTransformer(n_quantiles=5000, output_distribution="normal"))
        #     if each_f in ["liquidty", "resvol", "sizenl"]:
        #         common_factor_this *= -1
            top_category_factors[each_f] = common_factor_this
        combined_factor = self.gen_expected_return_top_category(top_category_factors)
        return combined_factor

    def filt_single_factor_stats(self, factor_source: SingleFactorSource, qntl:float):
        factors_stats = {}
        for meta in factor_source.get_filtered_factor_meta_list():
            factor_namespace = meta["factor_namespace"]
            factor_name = meta["factor"]["name"]
            factor_type = meta["factor_type"]
            top_category = meta["factor"]["category"][0]
            logging.info(
                f"Start [factor names space: {factor_namespace}, factor name: {factor_name}, factor type: {factor_type}] into [top category: {top_category}]")
            if top_category not in factors_stats:
                factors_stats[top_category] = {}
            factor_value_on_calculation_dates = factor_source.factor_value_store.get_factor_intermediate_status(
                factor_namespace, factor_name, factor_type, IndustryAndMarketMeanFiller,
                dates=self.factor_calculation_dates.tolist(),
                kwargs={"target_symbols": self.symbols,
                        "all_symbols_sorted_by_list_date": self.universe.wind_common.all_symbols_sorted_by_list_date})
            # rolling_icir_matched_calculation_dates = self.common_oper._calculate_rolling_icir_matched_factor_calculation_dates(factor_value_on_calculation_dates, self.stock_return)
            rolling_icir_matched_calculation_dates = self.common_oper._calculate_linear_combine_weight_by_icir(factor_value_on_calculation_dates, self.stock_return)

            factors_stats[top_category][factor_name] = rolling_icir_matched_calculation_dates
        factors_stats_df = {}
        for cate in factors_stats.keys():
            factors_stats_this_cate = pd.DataFrame(factors_stats[cate])
            icir_threshold = factors_stats_this_cate.abs().quantile(axis=1, q=qntl)
            factors_stats_this_cate = factors_stats_this_cate.mask(~(factors_stats_this_cate.abs().T > icir_threshold).T)
            factors_stats_df[cate] = factors_stats_this_cate.fillna(0).shift(self.lag)
        return factors_stats_df

    def filt_single_factor_group_return(self, factor_source: SingleFactorSource, qntl:float):
        factors_stats = {}
        for meta in factor_source.get_filtered_factor_meta_list():
            factor_namespace = meta["factor_namespace"]
            factor_name = meta["factor"]["name"]
            factor_type = meta["factor_type"]
            top_category = meta["factor"]["category"][0]
            logging.info(
                f"Start [factor names space: {factor_namespace}, factor name: {factor_name}, factor type: {factor_type}] into [top category: {top_category}]")
            if top_category not in factors_stats:
                factors_stats[top_category] = {}
            hedged_return_ir = self._get_hedged_return(factor_source, factor_namespace, factor_name, factor_type)
            factors_stats[top_category][factor_name] = hedged_return_ir
        factors_stats_df = {}
        for cate in factors_stats.keys():
            factors_stats_this_cate = pd.DataFrame(factors_stats[cate])
            icir_threshold = factors_stats_this_cate.abs().quantile(axis=1, q=qntl)
            factors_stats_this_cate = factors_stats_this_cate.mask(~(factors_stats_this_cate.abs().T > icir_threshold).T)
            factors_stats_df[cate] = factors_stats_this_cate.fillna(0)
        return factors_stats_df

    def get_layer2_top_category_factors(self, layer2_top_category_factors: dict[str, pd.DataFrame], top_category_factors: dict[str, pd.DataFrame]):
        trade_dates_len = len(self.trading_strategy_config.factor_calculation_dates)

        for k, v in self.factors_aggregate_config.groupby("category"):
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

    def __get_top_category_factor_value_on_factor_calcualtion_dates(self,
                                                                    top_category_factors: dict[str, pd.DataFrame],
                                                                    factor_source: SingleFactorSource,
                                                                    factor_stats: dict[str, pd.DataFrame]):
        for meta in factor_source.get_filtered_factor_meta_list():
            factor_namespace = meta["factor_namespace"]
            factor_name = meta["factor"]["name"]
            factor_type = meta["factor_type"]
            top_category = meta["factor"]["category"][0]
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
            # rolling_icir_matched_calculation_dates = self._get_rolling_icir_matched_factor_calculation_dates(
            #     factor_source, factor_namespace, factor_name, factor_type)
            rolling_icir_matched_calculation_dates = factor_stats[top_category][factor_name]
            weighted_factor_value = factor_value_on_calculation_dates.multiply(rolling_icir_matched_calculation_dates,
                                                                               axis=0)
            top_category_factor_df[(np.isnan(top_category_factor_df) & ~np.isnan(weighted_factor_value))] = 0
            top_category_factor_df += weighted_factor_value
            self._log_composite_factor(factor_namespace, factor_name, factor_type, factor_source.backtest_namespace)
            logging.info(
                f"Done combine [factor names space: {factor_namespace}, factor name: {factor_name}, factor type: {factor_type}] into [top category: {top_category}]")

    def _get_rolling_icir_matched_factor_calculation_dates(self,
            factor_source: SingleFactorSource, factor_namespace: str, factor_name: str, factor_type: str) -> pd.Series:
        factor_ic = factor_source.factor_ic_store.get_factor_rank_ic(factor_namespace, factor_name, factor_type, factor_source.backtest_namespace, BacktestMode.TOTAL)
        factor_ic = self.universe.align_df_with_trade_dates(factor_ic)
        factor_ic_on_trading_execution_dates = factor_ic.loc[self.trading_execution_dates]
        factor_rolling_icir: pd.Series = factor_ic_on_trading_execution_dates.rolling(self.icir_rolling_window, min_periods=self.icir_min_rolling_window).mean()/ \
            factor_ic_on_trading_execution_dates.rolling(self.icir_rolling_window, min_periods=self.icir_min_rolling_window).std()
        factor_rolling_icir_shift_to_match_factor_calculation_dates = pd.Series(index=self.factor_calculation_dates, data=np.nan)
        if self.is_last_date_factor_calculation_date:
            factor_rolling_icir_shift_to_match_factor_calculation_dates.iloc[1:] = factor_rolling_icir
        else:
            factor_rolling_icir_shift_to_match_factor_calculation_dates.iloc[:] = factor_rolling_icir.shift(periods=1)
        return factor_rolling_icir_shift_to_match_factor_calculation_dates

    def _get_hedged_return(self,
                           factor_source: SingleFactorSource, factor_namespace: str, factor_name: str, factor_type: str) -> pd.Series:
        factor_hedge_return = factor_source.factor_hedge_return_store.get_hedge_return(factor_namespace, factor_name, factor_type, factor_source.backtest_namespace, BacktestMode.TOTAL)
        try:
            factor_hedge_return = factor_hedge_return["top_hedge_csi1000_return"].add(1).cumprod().loc[self.trading_execution_dates].pct_change().replace(0, np.nan)
        except:
            factor_hedge_return = pd.Series(index=self.factor_calculation_dates, data=np.nan)
        factor_hedge_ir = factor_hedge_return.rolling(self.icir_rolling_window, min_periods=self.icir_min_rolling_window).mean()/\
                          factor_hedge_return.rolling(self.icir_rolling_window, min_periods=self.icir_min_rolling_window).std()

        factor_rolling_hedged_ir_shift_to_match_factor_calculation_dates = pd.Series(index=self.factor_calculation_dates, data=np.nan)
        if self.is_last_date_factor_calculation_date:
            factor_rolling_hedged_ir_shift_to_match_factor_calculation_dates.iloc[1:] = factor_hedge_ir
        else:
            factor_rolling_hedged_ir_shift_to_match_factor_calculation_dates.iloc[:] = factor_hedge_ir.shift(periods=1)
        return factor_rolling_hedged_ir_shift_to_match_factor_calculation_dates

    def gen_expected_return_top_category(self, top_category_factors: dict[str, pd.DataFrame]):
        stock_period_return = self.gen_stock_period_returns(5).values
        # stock_period_return = (self._transform_factor_value(stock_period_return,
        #                                                     preprocessing.RobustScaler())).values
        # top_category_factors = {k: self._transform_factor_value(v, preprocessing.RobustScaler()) for k,v in top_category_factors.items()}
        tradable = self.universe.get_tradable_symbols().reindex(self.factor_calculation_dates).values
        # common_factors = self.get_common_factors()
        # for each_f, each_fv in common_factors.items():
        #     top_category_factors[each_f] = each_fv

        top_categories = list(top_category_factors.keys())
        factor_beta = pd.DataFrame(0.0, index=self.factor_calculation_dates, columns=top_categories)
        factor_beta_val = factor_beta.values
        lasso_prediction_error = pd.Series(np.nan, index=self.factor_calculation_dates)

        for i in range(50, stock_period_return.shape[0]):
            logging.info(f"{i} / {stock_period_return.shape[0]}")
            j = i + 1
            pred_period = 3
            sample_period = 5
            # Define Features
            df = pd.DataFrame(
                {each_category: each_category_value.values[j-sample_period-pred_period:j-pred_period, :].flatten() for each_category, each_category_value in
                 top_category_factors.items()})
            use_factors = ((~np.isnan(df)).sum() > 0)
            df = df[use_factors[use_factors].index]
            df["Tradable"] = pd.Series(tradable[j-sample_period-pred_period:j-pred_period, :].flatten())
            df["StockRet"] = pd.Series(stock_period_return[j-sample_period-pred_period+1:j-pred_period+1, :].flatten())
            df = df[[i is True for i in df["Tradable"]]].dropna()

            train_size = int(len(df) * 0.5)
            validate_size = int(len(df) * 0.2)
            lambda1 = 10.0 ** np.arange(-5, -3, 1)

            tvt_alasso = asgl.TVT(model='lm', penalization='alasso', intercept=True, lambda1=lambda1, parallel=False,
                                  weight_technique='lasso', error_type='CORR', random_state=1, tol=1e-4,
                                  lambda1_weights=1e-3, tau=0.5,
                                  train_size=train_size, validate_size=validate_size)
            alasso_result = tvt_alasso.train_validate_test(x=df[use_factors[use_factors].index].values,
                                                           y=df["StockRet"].values)
            factor_beta_val[i, use_factors] = alasso_result['optimal_betas'][1:]  # Remove intercept
            lasso_prediction_error.iloc[i] = alasso_result['test_error']

        combined_factor = np.full((self.factor_calculation_dates.size, self.symbols.size), np.nan)
        for each_category in list(use_factors.index):
            weighted_factor_value = (
                        top_category_factors[each_category].T * (factor_beta[each_category].values)).T
            combined_factor[(np.isnan(combined_factor) & ~np.isnan(weighted_factor_value))] = 0
            combined_factor += weighted_factor_value
        return combined_factor


    def __transform_top_category_factors_value(self, top_category_factors: dict[str, pd.DataFrame]):
        for top_category in top_category_factors:
            logging.info(f"Start transform [top category: {top_category}] factor value")
            top_category_factors[top_category] = self.__transform_factor_value(top_category_factors[top_category],
                                                                               preprocessing.RobustScaler())
            logging.info(f"Done transform [top category: {top_category}] factor value")

    def __transform_factor_value(self, factor_df: pd.DataFrame, transformer: TransformerMixin) -> pd.DataFrame:
        result_df = pd.DataFrame(np.nan, index=factor_df.index, columns=factor_df.columns)
        for date, factor_value in factor_df.iterrows():
            not_nan_indexes = np.where(~np.isnan(factor_value))[0]
            if not_nan_indexes.size > 0:
                result_df.loc[date].iloc[not_nan_indexes] = transformer.fit_transform(
                    factor_value.iloc[not_nan_indexes].values.reshape(not_nan_indexes.size, 1)).reshape(
                    not_nan_indexes.size)
        return result_df

    def __combine_top_category_factors_as_expected_return(self, top_category_factors: dict[str, pd.DataFrame]):
        combined_factor = pd.DataFrame(np.nan, index=self.factor_calculation_dates, columns=self.symbols)
        weight = {}
        for top_category in top_category_factors:
            logging.info(f"Start combine [top category: {top_category}] into expected return")
            top_category_factor = top_category_factors[top_category]
            top_category_rolling_icir = self.common_oper._calculate_linear_combine_weight_by_icir(top_category_factor, self.stock_return).shift(self.lag)
            if top_category in list(self.barra_style_factors) + ["NetProfitDerivativeIndicatorsSUE_icir", "MoneyFlowLSDifference_icir"]:
                top_category_rolling_icir *= 3

            top_category_rolling_icir[top_category_rolling_icir < 0] = 0
            top_category_rolling_icir[top_category_rolling_icir < (top_category_rolling_icir.ewm(12).mean() - 3*top_category_rolling_icir.ewm(12).std())] = (top_category_rolling_icir.ewm(12).mean() - 3*top_category_rolling_icir.ewm(12).std())
            top_category_rolling_icir[top_category_rolling_icir > (top_category_rolling_icir.ewm(12).mean() + 3*top_category_rolling_icir.ewm(12).std())] = (top_category_rolling_icir.ewm(12).mean() + 3*top_category_rolling_icir.ewm(12).std())

            weighted_factor_value = top_category_factor.multiply(top_category_rolling_icir, axis=0)
            combined_factor[(np.isnan(combined_factor) & ~np.isnan(weighted_factor_value))] = 0
            combined_factor += weighted_factor_value
            weight[top_category] = top_category_rolling_icir
            logging.info(f"Done combine [top category: {top_category}] into expected return")
        return combined_factor

    def _get_factor_calculation_dates_to_previous_trading_execution_dates_series(self) -> pd.Series:
        if self.trading_strategy_config.is_last_date_factor_calculation_date:
            return pd.Series(index=self.factor_calculation_dates[1:], data=self.trading_execution_dates)
        else:
            return pd.Series(index=self.factor_calculation_dates[1:], data=self.trading_execution_dates[:-1])

    def _get_trading_execution_dates_to_previous_factor_calculation_dates_series(self):
        if self.trading_strategy_config.is_last_date_factor_calculation_date:
            return pd.Series(index=self.trading_execution_dates[1:], data=self.factor_calculation_dates[:-2])
        else:
            return pd.Series(index=self.trading_execution_dates[1:], data=self.factor_calculation_dates[:-1])