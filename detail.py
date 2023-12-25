
import json
import logging
from flask import current_app

import numpy as np
import pandas as pd
from chameqs_common.backtest.common.mode import BacktestMode
from chameqs_common.backtest.factor.common.store.groups_store import FactorGroupsReturnStore
from chameqs_common.backtest.factor.common.store.hedge_store import FactorHedgeReturnStore
from chameqs_common.backtest.factor.common.store.ic_store import FactorICStore
from chameqs_common.backtest.factor.common.store.meta_store import FactorBacktestMetaStore
from chameqs_common.backtest.factor.common.store.summary_store import FactorSummaryStore
from chameqs_common.data.universe import Universe
from chameqs_common.factor.common.store.factor_value_store import FactorValueStore
from chameqs_common.factor.single_factor.processor.coverage import FactorCoverageChecker
from chameqs_common.factor.single_factor.processor.neutralizer import LinearNeutralizer
from chameqs_common.utils.array_util import (deserialize_array,
                                             int_array_to_yyyyMMdd_str_list)
from chameqs_common.utils.decorator import timeit
from chameqs_view.factors.common.detail import FactorDetail


class SingleFactorDetail(FactorDetail):
    def __init__(self, 
            universe: Universe,
            single_factor_value_store: FactorValueStore, 
            factor_backtest_meta_store: FactorBacktestMetaStore,
            factor_ic_store: FactorICStore, 
            factor_groups_return_store: FactorGroupsReturnStore, 
            factor_hedge_return_store: FactorHedgeReturnStore, 
            factor_summary_store: FactorSummaryStore) -> None:
        super().__init__(universe, factor_backtest_meta_store, factor_ic_store, factor_groups_return_store, factor_hedge_return_store, factor_summary_store)
        self.single_factor_value_store = single_factor_value_store
        
    @timeit(logging.INFO)
    def get_detail(self, factor_namespace: str, factor_name: str, factor_type: str, backtest_namespace: str, mode: BacktestMode) -> dict:
        factor_meta = self.single_factor_value_store.get_factor_meta(factor_namespace, factor_name, factor_type)
        groups_return = self.factor_groups_return_store.get_groups_and_beta_return(factor_namespace, factor_name, factor_type, backtest_namespace, mode)
        hedge_return = self.factor_hedge_return_store.get_hedge_return(factor_namespace, factor_name, factor_type, backtest_namespace, mode)
        factor_summary = self.factor_summary_store.get_summary(factor_namespace, factor_name, factor_type, backtest_namespace, mode)
        dates = deserialize_array(factor_summary["dates"])
        dates_str_list = int_array_to_yyyyMMdd_str_list(dates)
        factor_first_not_nan_date, coverage_chart_config = self._generate_coverage_chart_config(dates, factor_namespace, factor_name, factor_type)
        groups_return_check = (groups_return != 0.0).any(axis=1)
        indexes_return, _, _ = self.universe.wind_common.get_indexes_compound_and_sum_net_for_dates(dates, groups_return_check[groups_return_check == True].index[0])
        valid_hedge_index_names = self._get_valid_hedge_index_names(factor_summary)
        content = {
            "view_type": "view1",
            "factor_namespace": factor_namespace,
            "backtest_namespace": backtest_namespace,
            "factor_name": factor_name,
            "factor_type": factor_type,
            "group_split_strategy": self._get_group_split_strategy_name(factor_namespace, factor_name, factor_type, backtest_namespace),
            "factor_doc_href": self._get_factor_doc_href(factor_meta),
            "factor_correlation_href": self._get_factor_correlation_href(factor_namespace, factor_name, factor_type, mode),
            "valid_hedge_index_names": valid_hedge_index_names,
            "factor_first_not_nan_date": factor_first_not_nan_date,
            "ic_mean": factor_summary["ic_mean"],
            "rank_ic_mean": factor_summary["rank_ic_mean"],
            "icir": factor_summary["icir"],
            "annulized_long_short_return": factor_summary["performance"]["hedge_return"]["long_short_group_return"]["total"]["annual_return"],
            "annulized_long_short_return_std": factor_summary["performance"]["hedge_return"]["long_short_group_return"]["total"]["vol"],
            "annulized_long_short_sharpe_ratio": factor_summary["performance"]["hedge_return"]["long_short_group_return"]["total"]["sharpe_ratio"],
            "coverage_chart_config": coverage_chart_config,
            "groups_daily_compound_net_chart_config": self._generate_groups_daily_net_chart_config(
                groups_return, indexes_return, dates_str_list, "groups_daily_compound_net_chart", "compound"),
            "groups_daily_sum_net_chart_config": self._generate_groups_daily_net_chart_config(
                groups_return, indexes_return, dates_str_list, "groups_daily_sum_net_chart", "sum"),
            "long_short_group_daily_compound_net_and_mdd_chart_config": self._generate_daily_net_and_mdd_chart_config(
                hedge_return["long_short_group_return"], "long_short_group_daily_compound_net_and_mdd_chart", "compound"),
            "long_short_group_daily_sum_net_and_mdd_chart_config": self._generate_daily_net_and_mdd_chart_config(
                hedge_return["long_short_group_return"], "long_short_group_daily_sum_net_and_mdd_chart", "sum"),
            "long_short_group_monthly_return_heatmap_chart_config": self._generate_monthly_return_heatmap_chart_config(hedge_return["long_short_group_return"], mode),
            "linear_neutralization_average_coefficients_chart_config": self._generate_linear_neutralization_average_coefficients_chart_config(dates, factor_namespace, factor_name, factor_type),
            "groups_annual_return_bar_chart_config": self._generate_groups_annual_return_bar_chart_config(groups_return),
            "groups_monthly_average_return_bar_chart_config": self._generate_groups_monthly_average_return_bar_chart_config(groups_return, mode),
            "ic_bar_chart_config": self._generate_ic_bar_chart_config(factor_namespace, factor_name, factor_type, backtest_namespace, mode, "ic_bar_chart"),
            "rank_ic_bar_chart_config": self._generate_rank_ic_bar_chart_config(factor_namespace, factor_name, factor_type, backtest_namespace, mode, "rank_ic_bar_chart"),
        }
        content["index_hedges"] = {}
        for index_name in valid_hedge_index_names:
            content["index_hedges"][f"annulized_top_hedge_{index_name.lower()}_return_mean"] = factor_summary["performance"]["hedge_return"][f"top_hedge_{index_name.lower()}_return"]["total"]["annual_return"]
            content["index_hedges"][f"annulized_top_hedge_{index_name.lower()}_return_std"] = factor_summary["performance"]["hedge_return"][f"top_hedge_{index_name.lower()}_return"]["total"]["vol"]
            content["index_hedges"][f"annulized_top_hedge_{index_name.lower()}_sharpe_ratio"] = factor_summary["performance"]["hedge_return"][f"top_hedge_{index_name.lower()}_return"]["total"]["sharpe_ratio"]
            content["index_hedges"][f"top_group_hedge_{index_name.lower()}_index_daily_compound_net_and_mdd_chart_config"] = \
                self._generate_daily_net_and_mdd_chart_config(
                    hedge_return[f"top_hedge_{index_name.lower()}_return"], f"top_group_hedge_{index_name.lower()}_index_daily_compound_net_and_mdd_chart", "compound")
            content["index_hedges"][f"top_group_hedge_{index_name.lower()}_index_daily_sum_net_and_mdd_chart_config"] = \
                self._generate_daily_net_and_mdd_chart_config(
                    hedge_return[f"top_hedge_{index_name.lower()}_return"], f"top_group_hedge_{index_name.lower()}_index_daily_sum_net_and_mdd_chart", "sum")
        return content

    def _get_factor_correlation_href(self, factor_namespace: str, factor_name: str, factor_type: str, mode: BacktestMode) -> str:
        return f"/single_factors/{mode.name.lower()}/single_factor_correlation/{self.universe.name}/{factor_namespace}/{factor_name}/{factor_type}"

    def _generate_coverage_chart_config(self, dates: np.ndarray, factor_namespace: str, factor_name: str, factor_type: str):
        factor_coverage_ratios: pd.Series = self.single_factor_value_store.get_factor_intermediate_status(factor_namespace, factor_name, factor_type, FactorCoverageChecker)
        factor_coverage_ratios = factor_coverage_ratios.loc[dates]
        if factor_coverage_ratios.empty:
            return "", ""
        chart_config = current_app.jinja_env.get_template("echarts/coverage_bar_chart_template.js").render(
            chart_id = "coverage_chart",
            labels = json.dumps(int_array_to_yyyyMMdd_str_list(factor_coverage_ratios.index.values)),
            data = json.dumps(factor_coverage_ratios.values.tolist()))
        factor_first_not_nan_date = ""
        for date in factor_coverage_ratios.index:
            if factor_coverage_ratios[date] > 0:
                factor_first_not_nan_date = date
                break
        return factor_first_not_nan_date, chart_config

    def _generate_linear_neutralization_average_coefficients_chart_config(self, dates: np.ndarray, factor_namespace: str, factor_name: str, factor_type: str):
        linear_neutralization_coefficients: pd.DataFrame = self.single_factor_value_store.get_factor_intermediate_status(factor_namespace, factor_name, factor_type, LinearNeutralizer)
        if linear_neutralization_coefficients is None:
            return ""
        linear_neutralization_coefficients = linear_neutralization_coefficients.loc[dates]
        if linear_neutralization_coefficients.empty:
            return ""
        chart_config = current_app.jinja_env.get_template("echarts/average_coefficients_bar_chart_template.js").render(
            chart_id = "linear_neutralization_average_coefficients_chart",
            labels = json.dumps(linear_neutralization_coefficients.columns.values.tolist()),
            data = json.dumps(np.nanmean(linear_neutralization_coefficients.values, axis=0).tolist()))
        return chart_config