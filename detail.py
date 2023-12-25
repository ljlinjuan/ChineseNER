
import json
from flask import current_app

import numpy as np
import pandas as pd
from chameqs_common.backtest.common.mode import BacktestMode
from chameqs_common.backtest.factor.common.store.groups_store import FactorGroupsReturnStore
from chameqs_common.backtest.factor.common.store.hedge_store import FactorHedgeReturnStore
from chameqs_common.backtest.factor.common.store.ic_store import FactorICStore
from chameqs_common.backtest.factor.common.store.meta_store import FactorBacktestMetaStore
from chameqs_common.backtest.factor.common.store.summary_store import FactorSummaryStore
from chameqs_common.config.trading_strategy_config import TradingStrategyConfig
from chameqs_common.data.universe import Universe
from chameqs_common.data.wind import INDEX_CODE_MAP
from chameqs_common.utils.array_util import (int_array_to_MM_number_list,
                                             int_array_to_yyyy_number_list,
                                             int_array_to_yyyyMMdd_str_list)


class FactorDetail:
    def __init__(self, 
            universe: Universe,
            factor_backtest_meta_store: FactorBacktestMetaStore,
            factor_ic_store: FactorICStore, 
            factor_groups_return_store: FactorGroupsReturnStore, 
            factor_hedge_return_store: FactorHedgeReturnStore, 
            factor_summary_store: FactorSummaryStore) -> None:
        self.universe = universe
        self.factor_backtest_meta_store = factor_backtest_meta_store
        self.factor_ic_store = factor_ic_store
        self.factor_groups_return_store = factor_groups_return_store
        self.factor_hedge_return_store = factor_hedge_return_store
        self.factor_summary_store = factor_summary_store
    
    def _get_group_split_strategy_name(self, factor_namespace: str, factor_name: str, factor_type: str, backtest_namespace: str) -> str:
        backtest_meta_list = self.factor_backtest_meta_store.get_backtest_meta_list(factor_namespace, factor_name, factor_type, backtest_namespace)
        for backtest_meta in backtest_meta_list:
            if backtest_meta["tester_class"] == "chameqs_common.backtest.factor.common.tester.groups.FactorGroupsReturnTester":
                return backtest_meta["description"]["Group Split Strategy"]

    def _get_valid_hedge_index_names(self, factor_summary: dict) -> list[str]:
        valid_hedge_index_names = []
        for index_name in INDEX_CODE_MAP:
            if f"top_hedge_{index_name.lower()}_return" in factor_summary["performance"]["hedge_return"]:
                valid_hedge_index_names.append(index_name)
        return valid_hedge_index_names

    def _get_factor_doc_href(self, factor_meta: dict) -> str:
        lib_name = factor_meta["factor"]["module"].split(".")[0]
        lib_version = factor_meta["lib_version"]
        factor_module = factor_meta["factor"]["module"]
        factor_class = factor_meta["factor"]["class"]
        return f"/docs/{lib_name}-{lib_version}-cp39-cp39-linux_x86_64.whl/{factor_module.replace('.', '/')}.html#{factor_class}"

    def _generate_groups_daily_net_chart_config(self, 
            groups_return: pd.DataFrame, indexes_return: dict[str, np.ndarray],
            date_list: list[str], chart_id: str, type: str):
        datasets = []
        groups_count = groups_return.columns.size - 1
        for i in range(groups_count):
            datasets.append({
                "label": "'Group {}'".format(i), 
                "data": json.dumps(groups_return["group_{}".format(i)].tolist())
                })
        datasets.append({"label": "'Beta'", "data": json.dumps(groups_return["beta"].tolist())})
        for index_name in indexes_return:
            datasets.append({"label": "'{}'".format(index_name), "data": json.dumps(indexes_return[index_name].tolist())})
        chart_config = current_app.jinja_env.get_template("echarts/nav_line_chart_template.js").render(
            chart_id = chart_id,
            nav_type = type,
            labels = json.dumps(date_list),
            datasets = datasets)
        return chart_config
    
    def _generate_daily_net_and_mdd_chart_config(self, returns: pd.Series, chart_id: str, nav_type: str):
        dates_str_list = int_array_to_yyyyMMdd_str_list(returns.index)
        chart_config = current_app.jinja_env.get_template("echarts/nav_and_mdd_mixed_chart_template.js").render(
            chart_id = chart_id,
            nav_type = nav_type,
            x_data = json.dumps(dates_str_list),
            returns = returns.to_list()
        )
        return chart_config
    
    def _generate_monthly_return_heatmap_chart_config(self, returns: pd.Series, mode: BacktestMode):
        nets = (returns + 1).cumprod()
        monthly_trading_execution_dates = self._get_monthly_trading_execution_dates(mode)
        monthly_trading_execution_dates_indexes = np.where(np.in1d(returns.index, monthly_trading_execution_dates))[0]
        monthly_trading_execution_dates_indexes = monthly_trading_execution_dates_indexes[monthly_trading_execution_dates_indexes <= nets.size]
        monthly_returns = np.diff(nets.iloc[monthly_trading_execution_dates_indexes]) / nets.iloc[monthly_trading_execution_dates_indexes][:-1]
        month_array = int_array_to_MM_number_list(monthly_trading_execution_dates)
        year_array = int_array_to_yyyy_number_list(monthly_trading_execution_dates)
        month_unique_array = np.unique(month_array[1:]).tolist() # asc order
        year_unique_array = np.flip(np.unique(year_array[1:])).tolist() # desc order
        return_in_percent_array = monthly_returns * 100
        # Use same range for min and max to make the negative and positive number have different colors on chart.
        range_limit = np.max([np.abs(min(return_in_percent_array)), np.abs(max(return_in_percent_array))])
        return_year_index = list(map(lambda y: year_unique_array.index(y), year_array[1:]))
        return_month_index = []
        for month_i in range(1, len(month_array)):
            previous_month = month_array[month_i - 1]
            month = month_array[month_i]
            if previous_month == month:
                index = month_unique_array.index(month)
            else:
                index = month_unique_array.index(month) - 1
            if index < 0:
                return_year_index[month_i - 1] = return_year_index[month_i - 1] + 1
                return_month_index.append(len(month_unique_array) - 1)
            else:
                return_month_index.append(index)
        chart_config = current_app.jinja_env.get_template("echarts/return_heatmap_chart_template.js").render(
            chart_id = "long_short_group_monthly_return_heatmap_chart",
            label = "'Long Short'",
            x_data = json.dumps(month_unique_array),
            y_data = json.dumps(year_unique_array),
            v_data = pd.DataFrame({
                "x": return_month_index,
                "y": return_year_index,
                "v": return_in_percent_array}).to_json(orient="values", double_precision=2),
            min_v = range_limit * -1,
            max_v = range_limit)
        return chart_config

    def _generate_groups_annual_return_bar_chart_config(self, groups_return: pd.DataFrame):
        dates = groups_return.index.values
        yyyy_list = int_array_to_yyyy_number_list(dates)
        yyyy_diff = np.diff(yyyy_list)
        year_unique_array = np.unique(yyyy_list).tolist()
        year_start_index = np.insert(np.where(yyyy_diff > 0)[0], 0, 0)
        year_end_index = np.append(np.where(yyyy_diff > 0)[0] - 1, dates.size - 1)
        legends = []
        datasets = []
        groups_count = groups_return.columns.size - 1
        groups_compound_net = (groups_return + 1).cumprod()
        for i in range(groups_count):
            name = "Group {}".format(i)
            legends.append(name)
            group_compound_net = groups_compound_net[f"group_{i}"].values
            group_annual_return_in_percent = ((group_compound_net[year_end_index] - group_compound_net[year_start_index]) / group_compound_net[year_start_index]) * 100
            datasets.append({"name": "'{}'".format(name), "data": json.dumps(group_annual_return_in_percent.tolist())})
        legends.append("Beta")
        beta_compound_net = groups_compound_net["beta"].values
        beta_annual_return_in_percent = ((beta_compound_net[year_end_index] - beta_compound_net[year_start_index]) / beta_compound_net[year_start_index]) * 100
        datasets.append({"name": "'Beta'", "data": json.dumps(beta_annual_return_in_percent.tolist())})
        chart_config = current_app.jinja_env.get_template("echarts/groups_return_bar_chart_template.js").render(
            chart_id = "groups_annual_return_bar_chart",
            labels = json.dumps(year_unique_array),
            legends = json.dumps(legends),
            datasets = datasets)
        return chart_config

    def _generate_groups_monthly_average_return_bar_chart_config(self, groups_return: pd.DataFrame, mode: BacktestMode):
        dates = groups_return.index.values
        month_array = np.array(int_array_to_MM_number_list(dates))
        month_unique_array = np.unique(month_array[1:]).tolist() # asc order
        monthly_trading_execution_dates = self._get_monthly_trading_execution_dates(mode)
        monthly_trading_execution_dates_indexes = np.where(np.in1d(dates, monthly_trading_execution_dates))[0]
        legends = []
        datasets = []
        groups_count = groups_return.columns.size - 1
        groups_compound_net = (groups_return + 1).cumprod()
        for i in range(groups_count):
            name = "Group {}".format(i)
            legends.append(name)
            group_compound_net = groups_compound_net[f"group_{i}"].values
            monthly_returns = np.diff(group_compound_net[monthly_trading_execution_dates_indexes]) / group_compound_net[monthly_trading_execution_dates_indexes][:-1]
            monthly_returns_series = pd.Series(index=month_array[monthly_trading_execution_dates_indexes[:-1]], data=monthly_returns)
            monthly_average_returns_in_percent = monthly_returns_series.groupby(monthly_returns_series.index).mean() * 100
            group_data = []
            for month in month_unique_array:
                if month in monthly_average_returns_in_percent.index:
                    group_data.append(monthly_average_returns_in_percent[month])
                else:
                    group_data.append(None)
            datasets.append({"name": "'{}'".format(name), "data": json.dumps(group_data)})
        beta_compound_net = groups_compound_net["beta"].values
        beta_monthly_returns = np.diff(beta_compound_net[monthly_trading_execution_dates_indexes]) / beta_compound_net[monthly_trading_execution_dates_indexes][:-1]
        beta_monthly_returns_series = pd.Series(index=month_array[monthly_trading_execution_dates_indexes[:-1]], data=beta_monthly_returns)
        beta_monthly_average_returns_in_percent = beta_monthly_returns_series.groupby(beta_monthly_returns_series.index).mean().values * 100
        legends.append("Beta")
        datasets.append({"name": "'Beta'", "data": json.dumps(beta_monthly_average_returns_in_percent.tolist())})
        chart_config = current_app.jinja_env.get_template("echarts/groups_return_bar_chart_template.js").render(
            chart_id = "groups_monthly_average_return_bar_chart",
            labels = json.dumps(month_unique_array),
            legends = json.dumps(legends),
            datasets = datasets)
        return chart_config

    def _get_monthly_trading_execution_dates(self, mode: BacktestMode):
        monthly_trading_strategy_config = TradingStrategyConfig(self.universe, "monthly")
        if mode == BacktestMode.TOTAL:
            return monthly_trading_strategy_config.monthly_trading_execution_dates
        elif mode == BacktestMode.IN_SAMPLE:
            return monthly_trading_strategy_config.in_sample_monthly_trading_execution_dates
        elif mode == BacktestMode.OUT_OF_SAMPLE:
            return monthly_trading_strategy_config.out_of_sample_monthly_trading_execution_dates
        else:
            raise Exception(f"Unknow mode: {mode}")

    def _generate_ic_bar_chart_config(self, factor_namespace: str, factor_name: str, factor_type: str, backtest_namespace: str, mode: BacktestMode, chart_id: str):
        ic = self.factor_ic_store.get_factor_ic(factor_namespace, factor_name, factor_type, backtest_namespace, mode)
        chart_config = current_app.jinja_env.get_template("echarts/1d_bar_chart_template.js").render(
            chart_id = chart_id,
            labels = json.dumps(int_array_to_yyyyMMdd_str_list(ic.index)),
            data = json.dumps(ic.tolist()))
        return chart_config
    
    def _generate_rank_ic_bar_chart_config(self, factor_namespace: str, factor_name: str, factor_type: str, backtest_namespace: str, mode: BacktestMode, chart_id: str):
        rank_ic = self.factor_ic_store.get_factor_rank_ic(factor_namespace, factor_name, factor_type, backtest_namespace, mode)
        chart_config = current_app.jinja_env.get_template("echarts/1d_bar_chart_template.js").render(
            chart_id = chart_id,
            labels = json.dumps(int_array_to_yyyyMMdd_str_list(rank_ic.index)),
            data = json.dumps(rank_ic.tolist()))
        return chart_config