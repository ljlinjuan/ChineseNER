import abc
import logging

import numpy as np
import pandas as pd
from chameqs_common.utils.decorator import ignore_warning
from chameqs_common.config.trading_strategy_config import TradingStrategyConfig
from chameqs_common.data.universe import Universe
from chameqs_common.factor.combined_factor.factor_definition.combined_factor import CombinedFactor
from chameqs_common.factor.single_factor.processor.filler import IndustryAndMarketMeanFiller
from chameqs_common.factor.single_factor.store.single_factor_source import SingleFactorSource
from sklearn import preprocessing
from sklearn.base import TransformerMixin


class TopCategoryCombinedFactorWithIC(CombinedFactor):
    def __init__(self,
                 name: str, universe: Universe, trading_strategy_config: TradingStrategyConfig, price_type: str,
                 factor_sources: list[SingleFactorSource], icir_rolling_window: int, icir_min_rolling_window: int) -> None:
        super().__init__(name, universe, trading_strategy_config, {"Price Type": price_type, "ICIR Rolling Window": icir_rolling_window, "ICIR Min Rolling Window": icir_min_rolling_window})
        self.price_df = self.universe.wind_common.get_price_df_by_type(price_type)
        self.factor_sources = factor_sources
        self.icir_rolling_window = icir_rolling_window
        self.icir_min_rolling_window = icir_min_rolling_window

    def _get_top_category_factor_value_on_factor_calcualtion_dates(self,
                                                                   top_category_factors: dict[str, pd.DataFrame],
                                                                   factor_source: SingleFactorSource):
        for meta in factor_source.get_filtered_factor_meta_list():
            factor_namespace = meta["factor_namespace"]
            factor_name = meta["factor"]["name"]
            factor_type = meta["factor_type"]
            top_category = meta["factor"]["category"][0]

            logging.info(
                f"Start combine [factor names space: {factor_namespace}, factor name: {factor_name}, factor type: {factor_type}] into [top category: {top_category}]")
            if top_category not in top_category_factors:
                top_category_factors[top_category] = pd.DataFrame(0, index=self.factor_calculation_dates, columns=self.symbols)
            top_category_factor_df = top_category_factors[top_category]
            factor_value_on_calculation_dates = factor_source.factor_value_store.get_factor_intermediate_status(
                factor_namespace, factor_name, factor_type, IndustryAndMarketMeanFiller, 
                dates=self.factor_calculation_dates.tolist(), 
                kwargs={"target_symbols": self.symbols, "all_symbols_sorted_by_list_date": self.universe.wind_common.all_symbols_sorted_by_list_date})
            rolling_icir_matched_calculation_dates = self._get_rolling_icir_matched_factor_calculation_dates(
                factor_source, factor_namespace, factor_name, factor_type)
            weighted_factor_value = factor_value_on_calculation_dates.multiply(rolling_icir_matched_calculation_dates, axis=0)
            top_category_factor_df[(np.isnan(top_category_factor_df) & ~np.isnan(weighted_factor_value))] = 0
            top_category_factor_df += weighted_factor_value
            self._log_composite_factor(factor_namespace, factor_name, factor_type, factor_source.backtest_namespace)
            logging.info(f"Done combine [factor names space: {factor_namespace}, factor name: {factor_name}, factor type: {factor_type}] into [top category: {top_category}]")

    @abc.abstractmethod
    def _get_rolling_icir_matched_factor_calculation_dates(self,
                                                           factor_source: SingleFactorSource, factor_namespace: str,
                                                           factor_name: str, factor_type: str) -> pd.Series:
        pass

    def _transform_top_category_factors_value(self, top_category_factors: dict[str, pd.DataFrame]):
        for top_category in top_category_factors:
            logging.info(f"Start transform [top category: {top_category}] factor value")
            top_category_factors[top_category] = self._transform_factor_value(top_category_factors[top_category],
                                                                              preprocessing.PowerTransformer())  # RobustScaler()
            logging.info(f"Done transform [top category: {top_category}] factor value")

    def _quantile_transform_top_category_factors_value(self, top_category_factors: dict[str, pd.DataFrame]):
        for top_category in top_category_factors:
            logging.info(f"Start transform [top category: {top_category}] factor value")
            top_category_factors[top_category] = self._quantile_transform_factor_value(
                top_category_factors[top_category])
            logging.info(f"Done transform [top category: {top_category}] factor value")

    @ignore_warning
    def _quantile_transform_factor_value(self, factor_df: pd.DataFrame, ) -> pd.DataFrame:
        result_df = pd.DataFrame(np.nan, index=factor_df.index, columns=factor_df.columns)
        for date, factor_value in factor_df.iterrows():
            not_nan_indexes = np.where(~np.isnan(factor_value))[0]
            if not_nan_indexes.size > 0:
                result_df.loc[date].iloc[not_nan_indexes] = preprocessing.QuantileTransformer(n_quantiles=5000,
                                                                                              output_distribution="normal").fit_transform(
                    factor_value.iloc[not_nan_indexes].values.reshape(not_nan_indexes.size, 1)).reshape(
                    not_nan_indexes.size)
        return result_df

    def _transform_factor_value(self, factor_df: pd.DataFrame, transformer: TransformerMixin) -> pd.DataFrame:
        result_df = pd.DataFrame(np.nan, index=factor_df.index, columns=factor_df.columns)
        for date, factor_value in factor_df.iterrows():
            not_nan_indexes = np.where(~np.isnan(factor_value))[0]
            if not_nan_indexes.size > 0:
                result_df.loc[date].iloc[not_nan_indexes] = transformer.fit_transform(
                    factor_value.iloc[not_nan_indexes].values.reshape(not_nan_indexes.size, 1)).reshape(
                    not_nan_indexes.size)
        return result_df

    def _combine_top_category_factors_as_expected_return(self, top_category_factors: dict[str, pd.DataFrame]):
        combined_factor = pd.DataFrame(np.nan, index=self.factor_calculation_dates, columns=self.symbols)
        for top_category in top_category_factors:
            logging.info(f"Start combine [top category: {top_category}] into expected return")
            top_category_factor = top_category_factors[top_category]
            top_category_rolling_icir = self._calculate_rolling_icir_matched_factor_calculation_dates(
                top_category_factor)
            weighted_factor_value = top_category_factor.multiply(top_category_rolling_icir, axis=0)
            combined_factor[(np.isnan(combined_factor) & ~np.isnan(weighted_factor_value))] = 0
            combined_factor += weighted_factor_value
            logging.info(f"Done combine [top category: {top_category}] into expected return")
        return combined_factor

    @abc.abstractmethod
    def _calculate_rolling_icir_matched_factor_calculation_dates(self,
                                                                 factor_on_calculation_dates: pd.DataFrame) -> pd.Series:
        pass