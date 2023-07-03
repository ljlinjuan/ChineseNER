
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from hqb.testers.single_factor.coverage import FactorCoverageTestor
from hqb.testers.single_factor.groups import SimpleGroupsTester
from hqb.testers.single_factor.hedge import GroupHedgeTester, IndexHedgeTester
from hqb.testers.single_factor.icir import ICIRTester
from hqd.decorator import timeit
from hqu.sanitize_filename import sanitize
from jinja2 import Environment, PackageLoader, select_autoescape


class SingleFactorBackTestResultHTMLGenerator:
    def __init__(self,
        name: str,
        neutralizer_parameter_coefficients: pd.DataFrame,
        coverage_result: FactorCoverageTestor.Result,
        icir_result: ICIRTester.Result,
        groups_result: SimpleGroupsTester.Result,
        group_hedge_result: GroupHedgeTester.Result,
        index_hedge_result: IndexHedgeTester.Result,
        local_cache_folder = os.path.join(tempfile.gettempdir(), "hqb_results", "single_factor", "direct")) -> None:
        self.name = name
        self.neutralizer_parameter_coefficients = neutralizer_parameter_coefficients
        self.coverage_result = coverage_result
        self.icir_result = icir_result
        self.groups_result = groups_result
        self.group_hedge_result = group_hedge_result
        self.index_hedge_result =index_hedge_result
        self.env = Environment(loader=PackageLoader('.'.join(__name__.split('.')[:-1])), autoescape=select_autoescape())
        self.template = self.env.get_template("single_factor_backtest_result.html")
        self.local_cache_folder = local_cache_folder
        Path(self.local_cache_folder).mkdir(parents=True, exist_ok=True)
    
    @timeit
    def run(self) -> str:
        html_file_name = sanitize("{}.html".format(self.name))
        html_file_path = os.path.join(self.local_cache_folder, html_file_name)
        if os.path.exists(html_file_path):
            os.remove(html_file_path)
        with open(html_file_path, "w") as html_file:
            html_file.write(self.__render())
        return html_file_path, html_file_name
    
    def __render(self) -> str:
        return self.template.render(
            title = self.name,
            factor_first_not_nan_date = self.groups_result.factor_first_not_nan_date,
            ic_mean = self.icir_result.ic_mean,
            rank_ic_mean = self.icir_result.rank_ic_mean,
            icir = self.icir_result.icir,
            annulized_long_short_return_mean = self.group_hedge_result.annulized_group_hedge_return_mean,
            annulized_long_short_return_std = self.group_hedge_result.annulized_group_hedge_return_std,
            annulized_long_short_sharpe_ratio = self.group_hedge_result.annulized_group_hedge_sharpe_ratio,
            annulized_top_hedge_hs300_return_mean = self.index_hedge_result.annulized_top_hedge_hs300_return_mean,
            annulized_top_hedge_hs300_return_std = self.index_hedge_result.annulized_top_hedge_hs300_return_std,
            annulized_top_hedge_hs300_sharpe_ratio = self.index_hedge_result.annulized_top_hedge_hs300_sharpe_ratio,
            annulized_top_hedge_csi500_return_mean = self.index_hedge_result.annulized_top_hedge_csi500_return_mean,
            annulized_top_hedge_csi500_return_std = self.index_hedge_result.annulized_top_hedge_csi500_return_std,
            annulized_top_hedge_csi500_sharpe_ratio = self.index_hedge_result.annulized_top_hedge_csi500_sharpe_ratio,
            coverage_chart_config = self.__generate_coverage_chart_config(),
            groups_daily_compound_net_chart_config = self.__generate_groups_daily_net_chart_config(
                "group_{}_compound_net", "beta_tradable_compound_net", "HS300_compound_net", "CSI500_compound_net", "groups_daily_compound_net_chart"),
            groups_daily_sum_net_chart_config = self.__generate_groups_daily_net_chart_config(
                "group_{}_sum_net", "beta_tradable_sum_net", "HS300_sum_net", "CSI500_sum_net", "groups_daily_sum_net_chart"),
            long_short_group_daily_compound_net_and_mdd_chart_config = self.__generate_long_short_group_daily_net_and_mdd_chart_config(
                self.group_hedge_result.daily_df, "group_hedge_compound_net", "long_short_group_daily_compound_net_and_mdd_chart"),
            long_short_group_daily_sum_net_and_mdd_chart_config = self.__generate_long_short_group_daily_net_and_mdd_chart_config(
                self.group_hedge_result.daily_df, "group_hedge_sum_net", "long_short_group_daily_sum_net_and_mdd_chart"),
            top_group_hedge_hs300_index_daily_compound_net_and_mdd_chart_config = self.__generate_long_short_group_daily_net_and_mdd_chart_config(
                self.index_hedge_result.daily_df, "top_hedge_HS300_compound_net", "top_group_hedge_hs300_index_daily_compound_net_and_mdd_chart"),
            top_group_hedge_hs300_index_daily_sum_net_and_mdd_chart_config = self.__generate_long_short_group_daily_net_and_mdd_chart_config(
                self.index_hedge_result.daily_df, "top_hedge_HS300_sum_net", "top_group_hedge_hs300_index_daily_sum_net_and_mdd_chart"),
            top_group_hedge_csi500_index_daily_compound_net_and_mdd_chart_config = self.__generate_long_short_group_daily_net_and_mdd_chart_config(
                self.index_hedge_result.daily_df, "top_hedge_CSI500_compound_net", "top_group_hedge_csi500_index_daily_compound_net_and_mdd_chart"),
            top_group_hedge_csi500_index_daily_sum_net_and_mdd_chart_config = self.__generate_long_short_group_daily_net_and_mdd_chart_config(
                self.index_hedge_result.daily_df, "top_hedge_CSI500_sum_net", "top_group_hedge_csi500_index_daily_sum_net_and_mdd_chart"),
            long_short_group_monthly_return_heatmap_chart_config = self.__generate_long_short_group_monthly_return_heatmap_chart_config(),
            linear_neutralization_average_coefficients_chart_config = self.__generate_linear_neutralization_average_coefficients_chart_config(),
            groups_annual_return_bar_chart_config = self.__generate_groups_annual_return_bar_chart_config(),
            groups_monthly_average_return_bar_chart_config = self.__generate_groups_monthly_average_return_bar_chart_config(),
            ic_daily_moving_bar_chart_config = self.__generate_ic_daily_moving_bar_chart_config(
                "ic_daily_moving_bar_chart", "ic"),
            rank_ic_daily_moving_bar_chart_config = self.__generate_ic_daily_moving_bar_chart_config(
                "rank_ic_daily_moving_bar_chart", "rank_ic"),
            ic_in_days_bucket_bar_chart_config = self.__generate_ic_from_factor_calculation_days_bucket_bar_chart_config(
                "ic_in_days_bucket_bar_chart", "ic"),
            rank_ic_in_days_bucket_bar_chart_config = self.__generate_ic_from_factor_calculation_days_bucket_bar_chart_config(
                "rank_ic_in_days_bucket_bar_chart", "rank_ic"))

    def __generate_coverage_chart_config(self):
        chart_config = self.env.get_template("coverage_bar_chart_template.js").render(
            chart_id = "coverage_chart",
            labels = json.dumps(date_array_to_yyyyMMdd_str_array(self.coverage_result.coverage_df.index.values)),
            data = json.dumps(self.coverage_result.coverage_df.loc[:,"ratio"].values.tolist()))
        return chart_config
    
    def __generate_groups_daily_net_chart_config(self,
        group_net_column_name: str,
        beta_net_column_name: str,
        hs300_net_column_name: str,
        csi500_net_column_name: str,
        chart_id: str):
        datasets = []
        for i in range(self.groups_result.number_of_groups):
            datasets.append({"label": "'Group {}'".format(i), "data": json.dumps(self.groups_result.daily_df.loc[:,group_net_column_name.format(i)].values.tolist())})
        datasets.append({"label": "'Beta (tradable)'", "data": json.dumps(self.groups_result.daily_df.loc[:,beta_net_column_name].values.tolist())})
        datasets.append({"label": "'HS300'", "data": json.dumps(self.index_hedge_result.daily_df.loc[:,hs300_net_column_name].values.tolist())})
        datasets.append({"label": "'CSI500'", "data": json.dumps(self.index_hedge_result.daily_df.loc[:,csi500_net_column_name].values.tolist())})
        chart_config = self.env.get_template("net_line_chart_template.js").render(
            chart_id = chart_id,
            labels = json.dumps(date_array_to_yyyyMMdd_str_array(self.groups_result.daily_df.index.values.tolist())),
            datasets = datasets)
        return chart_config
    
    def __generate_long_short_group_daily_net_and_mdd_chart_config(self, daily_df: pd.DataFrame, net_column_name: str, chart_id: str):
        date_array = date_array_to_yyyyMMdd_str_array(daily_df.index)
        nets = daily_df.loc[:,net_column_name].values
        roll_max = pd.Series(nets).cummax().values
        mdd = ((nets / roll_max) - 1 ) * 100
        chart_config = self.env.get_template("net_and_mdd_mixed_chart_template.js").render(
            chart_id = chart_id,
            x_data = json.dumps(date_array),
            y_data_1 = json.dumps(nets.tolist()),
            y_data_2 = json.dumps(mdd.tolist()))
        return chart_config
    
    def __generate_long_short_group_monthly_return_heatmap_chart_config(self):
        returns = self.group_hedge_result.monthly_df.loc[:,"group_hedge_return"]
        month_array = date_array_to_MM_number_array(returns.index.values)
        year_array = date_array_to_yyyy_number_array(returns.index.values)
        return_in_percent_array = returns.values * 100
        month_unique_array = np.unique(month_array).tolist() # asc order
        year_unique_array = np.flip(np.unique(year_array)).tolist() # desc order
        chart_config = self.env.get_template("return_heatmap_chart_template.js").render(
            chart_id = "long_short_group_monthly_return_heatmap_chart",
            label = "'Long Short'",
            x_data = json.dumps(month_unique_array),
            y_data = json.dumps(year_unique_array),
            v_data = pd.DataFrame({
                "x": map(lambda x: month_unique_array.index(x), month_array),
                "y": map(lambda y: year_unique_array.index(y), year_array),
                "v": return_in_percent_array}).to_json(orient="values", double_precision=2),
            min_v = min(return_in_percent_array),
            max_v = max(return_in_percent_array))
        return chart_config
    
    def __generate_linear_neutralization_average_coefficients_chart_config(self):
        if self.neutralizer_parameter_coefficients.empty:
            return ""
        chart_config = self.env.get_template("average_coefficients_bar_chart_template.js").render(
            chart_id = "linear_neutralization_average_coefficients_chart",
            labels = json.dumps(self.neutralizer_parameter_coefficients.columns.values.tolist()),
            data = json.dumps(np.nanmean(self.neutralizer_parameter_coefficients.values, axis=0).tolist()))
        return chart_config

    def __generate_groups_annual_return_bar_chart_config(self):
        year_unique_array = np.unique(date_array_to_yyyy_number_array(self.groups_result.period_df.index.values)).tolist()
        legends = []
        datasets = []
        for i in range(self.groups_result.number_of_groups):
            name = "Group {}".format(i)
            legends.append(name)
            group_monthly_return = self.groups_result.period_df.loc[:,"group_{}_return".format(i)]
            group_annual_return_in_percent = group_monthly_return.groupby(lambda x: x.year).sum().values * 100
            datasets.append({"name": "'{}'".format(name), "data": json.dumps(group_annual_return_in_percent.tolist())})
        legends.append("Beta (tradable)")
        datasets.append({"name": "'Beta (tradable)'", "data": json.dumps((self.groups_result.period_df.loc[:,"beta_tradable_return"].groupby(lambda x: x.year).sum().values * 100).tolist())})
        chart_config = self.env.get_template("groups_return_bar_chart_template.js").render(
            chart_id = "groups_annual_return_bar_chart",
            labels = json.dumps(year_unique_array),
            legends = json.dumps(legends),
            datasets = datasets)
        return chart_config
    
    def __generate_groups_monthly_average_return_bar_chart_config(self):
        month_unique_array = np.unique(date_array_to_MM_number_array(self.groups_result.daily_df.index.values)).tolist()
        legends = []
        datasets = []
        for i in range(self.groups_result.number_of_groups):
            name = "Group {}".format(i)
            legends.append(name)
            group_monthly_return = self.groups_result.daily_df.loc[:,"group_{}_return".format(i)]
            group_monthly_average_return_in_percent = group_monthly_return.groupby(lambda x: x.month).mean().values * 100 * 22
            datasets.append({"name": "'{}'".format(name), "data": json.dumps(group_monthly_average_return_in_percent.tolist())})
        legends.append("Beta (tradable)")
        datasets.append({"name": "'Beta (tradable)'", "data": json.dumps((self.groups_result.daily_df.loc[:,"beta_tradable_return"].groupby(lambda x: x.month).mean().values * 100 * 22).tolist())})
        chart_config = self.env.get_template("groups_return_bar_chart_template.js").render(
            chart_id = "groups_monthly_average_return_bar_chart",
            labels = json.dumps(month_unique_array),
            legends = json.dumps(legends),
            datasets = datasets)
        return chart_config
    
    def __generate_ic_daily_moving_bar_chart_config(self, chart_id, ic_column_name):
        chart_config = self.env.get_template("ic_bar_chart_template.js").render(
            chart_id = chart_id,
            labels = json.dumps(date_array_to_yyyyMMdd_str_array(self.icir_result.ic_daily_df.index.values)),
            data = json.dumps(self.icir_result.ic_daily_df.loc[:,ic_column_name].values.tolist()))
        return chart_config
    
    def __generate_ic_from_factor_calculation_days_bucket_bar_chart_config(self, chart_id, ic_column_name):
        df = self.icir_result.ic_daily_df.groupby("days_gap_to_factor_calculation_date").mean()
        chart_config = self.env.get_template("ic_bar_chart_template.js").render(
            chart_id = chart_id,
            labels = json.dumps(df.index.values.tolist()),
            data = json.dumps(df.loc[:,ic_column_name].values.tolist()))
        return chart_config

def date_array_to_yyyyMMdd_str_array(array):
    return list(map(lambda x: "'{}'".format(x.strftime('%Y-%m-%d')), array))

def date_array_to_yyyy_number_array(array):
    return list(map(lambda x: x.year, array))

def date_array_to_MM_number_array(array):
    return list(map(lambda x: x.month, array))
