import pandas as pd
import numpy as np
import logging
import os
from sklearn import preprocessing

from chameqs_red import asgl
from chameqs_common.config.trading_strategy_config import TradingStrategyConfig
from chameqs_common.data.universe import Universe
from chameqs_common.factor.single_factor.store.single_factor_source import SingleFactorSource
from chameqs_common.backtest.common.mode import BacktestMode
from chameqs_common.utils.calculator import calc_rowwise_nan_correlation, calc_rowwise_nan_rank_correlation
from chameqs_red.factor.return_attribution.barra_pure_factor_portfolio import PureFactor
from chameqs_red.factor.combined_factor.expect_return.category_combined_factor import TopCategoryCombinedFactorWithIC
from chameqs_common.factor.single_factor.processor.filler import IndustryAndMarketMeanFiller
from chameqs_red.factor.combined_factor.expect_return import factor_categories_config
from chameqs_red.factor.combined_factor.expect_return.excess_return_defination import ExcessReturnDefination


class ExpectReturnGenerator(TopCategoryCombinedFactorWithIC):
    def __init__(self,
                 name: str, universe: Universe, trading_strategy_config: TradingStrategyConfig, price_type: str,
                 factor_sources: list[SingleFactorSource], icir_rolling_window: int,
                 icir_min_rolling_window: int) -> None:
        super().__init__(name, universe, trading_strategy_config, price_type, factor_sources, icir_rolling_window,
                         icir_min_rolling_window)
        self.factor_dates_price = self.price_df.loc[self.factor_calculation_dates]
        self.factor_dates_return = self.factor_dates_price.diff(axis=0) / self.factor_dates_price.shift(axis=0, periods=1)
        self.barra_style_factors = np.array(["beta", "btop", "earnyild", "growth", "leverage", "liquidty", "momentum", "resvol", "sizefactor", "sizenl"])
        self.pure_factor_portfolio_constructor = PureFactor(universe, trading_strategy_config)
        # self.factors_aggregate_config = pd.DataFrame(factor_categories_config.MAPPING)
        self.factors_aggregate_config = self.read_factors_aggregate_config()

    def read_factors_aggregate_config(self):
        path = "D:\Data"
        conf = pd.read_excel(os.path.join(path, "factor_categories.xlsx"), sheet_name="part_202206", index_col=0)
        return conf

    def get_df(self) -> pd.DataFrame:
        top_category_factors: dict[str, pd.DataFrame] = {}
        # for factor_source in self.factor_sources:
        #     self.save_single_factor_value(top_category_factors, factor_source)
        #     self._get_top_category_factor_value_on_factor_calcualtion_dates(top_category_factors, factor_source)

        path = r"D:\Data\CombinedFactor\by_realized_ic_update"
        for f in self.factors_aggregate_config["factor_name"]:
            df = pd.read_parquet(os.path.join(path, f))
            df.columns = df.columns.astype(int)
            top_category_factors[f] = self.universe.align_df_with_trade_dates_and_symbols(df)

        # for k, v in top_category_factors.items():
        #     v.columns = v.columns.astype(str)
        #     v.to_parquet(os.path.join(path, k))

        self.intersection_of_factor_categories_and_aggregate_config(list(top_category_factors.keys()))
        layer2_top_category_factors: dict[str, pd.DataFrame] = {}
        self.get_layer2_top_category_factors(layer2_top_category_factors, top_category_factors)
        combined_factor = self.gen_expected_return_top_category(layer2_top_category_factors)
        return combined_factor

    def intersection_of_factor_categories_and_aggregate_config(self, input_factor_categories: list[str]):
        cof = self.factors_aggregate_config.copy(deep=True)
        self.factors_aggregate_config = cof[[i in input_factor_categories for i in cof["factor_name"]]]
        for f in input_factor_categories:
            if f not in self.factors_aggregate_config["factor_name"].to_list():
                print(f"Warning!! ExpectReturnGenerator: cannot find factor {f} in configuration.")

    def save_single_factor_value(self, top_category_factors: dict[str, pd.DataFrame],
                                                                   factor_source: SingleFactorSource):
        path = r"D:\Data\CombinedFactor\by_realized_ic_update"
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
            factor_filled_value = factor_source.factor_value_store.get_factor_intermediate_status(
                factor_namespace, factor_name, factor_type, IndustryAndMarketMeanFiller,
                kwargs={"target_symbols": self.symbols, "all_symbols_sorted_by_list_date": self.universe.wind_common.all_symbols_sorted_by_list_date})


    def get_layer2_top_category_factors(self, layer2_top_category_factors: dict[str, pd.DataFrame], top_category_factors: dict[str, pd.DataFrame]):
        trade_dates_len = len(self.trading_strategy_config.factor_calculation_dates)

        for k , v in self.factors_aggregate_config.groupby("category"):
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

    def get_common_factors(self):
        common_factors = {}
        include_industries = True
        industry_names = self.universe.get_citics_highest_level_industries().loc[:,
                         "industry_name"].values if include_industries else []
        industry_symbols = self.universe.get_citics_highest_level_industry_symbols().fillna(
            method="backfill") if include_industries else None

        for i in range(0, self.barra_style_factors.size):
            barra_df = self.universe.get_transform("barra", "am_marketdata_stock_barra_rsk",
                                                   self.barra_style_factors[i]).fillna(method="pad")
            common_factors[self.barra_style_factors[i]] = barra_df.loc[self.factor_calculation_dates]
            common_factors[self.barra_style_factors[i]] = self._transform_factor_value(barra_df.loc[self.factor_calculation_dates], preprocessing.RobustScaler())

        for j in range(0, industry_names.size):
            common_factors[industry_names[j]] = (industry_symbols == j).loc[self.factor_calculation_dates]
        return common_factors

    def gen_stock_period_returns(self, period: int):
        return (self.factor_dates_price.shift(-period).fillna(method="pad") / self.factor_dates_price).shift(1) -1

    def gen_debarra_excess_return(self, period: int):
        erd = ExcessReturnDefination(self.universe)
        excess_return = erd.get_stk_excess_ret_debarra()
        excess_return_nav = excess_return.fillna(0).add(1).cumprod()
        period_excess_nav = excess_return_nav.reindex(self.factor_calculation_dates)
        return (period_excess_nav.shift(-period).fillna(method="pad") / period_excess_nav).shift(1) -1

    def gen_expected_return_top_category(self, top_category_factors: dict[str, pd.DataFrame]):
        alpha_factors = list(top_category_factors.keys())
        # mkv = np.log(self.universe.wind_common.get_free_circulation_market_value_na_filled().reindex(self.factor_calculation_dates))
        # stock_period_return = self.factor_dates_return
        stock_period_return = self.gen_stock_period_returns(5)
        stock_period_return = (self._transform_factor_value(stock_period_return,
                                                            preprocessing.RobustScaler())).values
        top_category_factors = {k: self._transform_factor_value(v, preprocessing.RobustScaler()) for k,v in top_category_factors.items()}
        tradable = self.universe.get_tradable_symbols().reindex(self.factor_calculation_dates).values

        common_factors = self.get_common_factors()
        for each_f, each_fv in common_factors.items():
            top_category_factors[each_f] = each_fv

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

        # factor_beta_smooth = (factor_beta.rolling(self.icir_rolling_window).mean() / factor_beta.rolling(self.icir_rolling_window).std()).fillna(0)
        # factor_beta_smooth[np.isinf(factor_beta_smooth)] = 0
        # factor_beta_smooth[factor_beta_smooth < 0] = 0
        combined_factor = np.full((self.factor_calculation_dates.size, self.symbols.size), np.nan)
        for each_category in list(use_factors.index):
            weighted_factor_value = (
                        top_category_factors[each_category].T * (factor_beta[each_category].values)).T
            combined_factor[(np.isnan(combined_factor) & ~np.isnan(weighted_factor_value))] = 0
            combined_factor += weighted_factor_value
        return combined_factor
        # return self._transform_factor_value(pd.DataFrame(combined_factor), preprocessing.QuantileTransformer())

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

    def _calculate_rolling_icir_matched_factor_calculation_dates(self, factor_on_calculation_dates: pd.DataFrame) -> pd.Series:
        # NOTE. When combine factors, we have to avoid the forward-looking problem.
        # Let's assume we have:
        #   factor calculation      t+0, t+4, t+8, t+12
        #   trading execution dates t+1, t+5, t+9, t+13
        # IC is calculated using formula: return between(t+5, t+1) correlation with factor value from t+0
        # When combining factors on t+8, we can only use the ic on t+5.
        ic_matched_factor_calculation_dates = pd.Series(data=np.nan, index=self.factor_calculation_dates)
        end_price = self.price_df.loc[self.trading_execution_dates]
        start_price = end_price.shift(periods=1)
        period_returns = (end_price - start_price) / start_price
        # NOTE. The first trading execution date has no ic. Exclude it for the convenience of calculation.
        trading_execution_dates_for_calculation = self.trading_execution_dates[1:]
        trading_execution_dates_to_previous_factor_calculation_dates_series = self._get_trading_execution_dates_to_previous_factor_calculation_dates_series()
        factor_calculation_dates_to_previous_trading_execution_dates_series = self._get_factor_calculation_dates_to_previous_trading_execution_dates_series()
        ic_on_trading_execution_dates = pd.Series(
            index=trading_execution_dates_for_calculation,
            data=calc_rowwise_nan_rank_correlation(
                period_returns.loc[trading_execution_dates_for_calculation],
                factor_on_calculation_dates.loc[trading_execution_dates_to_previous_factor_calculation_dates_series.loc[trading_execution_dates_for_calculation]].values))
        # NOTE. The first two factor calculation dates has no previous trading execution dates with ic. Exclude it for the convenience of calculation.
        factor_calculation_dates_for_calculation = self.factor_calculation_dates[2:]
        ic_matched_factor_calculation_dates.loc[factor_calculation_dates_for_calculation] = ic_on_trading_execution_dates.loc[factor_calculation_dates_to_previous_trading_execution_dates_series.loc[factor_calculation_dates_for_calculation]].values
        # NOTE. Return index as factor calculaton dates, data as ic from previous trading execution dates.
        rolling_icir_matched_factor_calculation_dates = (ic_matched_factor_calculation_dates.rolling(self.icir_rolling_window, min_periods=self.icir_min_rolling_window).mean()/ \
                ic_matched_factor_calculation_dates.rolling(self.icir_rolling_window, min_periods=self.icir_min_rolling_window).std())
        return rolling_icir_matched_factor_calculation_dates

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