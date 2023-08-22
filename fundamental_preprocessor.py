# for type hint without circular import error
from __future__ import annotations

import abc
import datetime
import logging
import multiprocessing as mp

import numpy as np
import pandas as pd
from chameqs_common.utils.date_util import date_to_int
from chameqs_common.data.query import Query, get_clickhouse_client
from chameqs_common.data.universe import Universe
from chameqs_red_local.data.fundamental.fundamental_data_utils import *


class FudamentalCommon(abc.ABC):

    def _load_ashareincome_field(self, field="net_profit_excl_min_int_inc") -> pd.DataFrame:
        """
        Read and process ashareincome data
        """
        df_ashareincome = Query().from_("wind", "ashareincome")\
            .on("statement_type").equals("408001000")\
            .on("s_info_windcode").in_query(self.universe.wind_common.wind_codes_query) \
            .on("ann_dt").range(self.universe.window_buffer_from_date, self.universe.to_date) \
            .select("actual_ann_dt", "ann_dt", "report_period", "s_info_windcode", field).df()
        df_ashareincome = df_ashareincome[~(df_ashareincome["actual_ann_dt"] > df_ashareincome["ann_dt"])]
        df_ashareincome = self.__fundamental_data_pre_process(df_ashareincome,
                                                              season_mapping_col="report_period",
                                                              drop_dup_col_sort=["s_info_windcode", "ann_dt"],
                                                              drop_dup_col_keep_last="report_period"
                                                              )
        return df_ashareincome

    def _load_profit_express_field(self, field="net_profit_excl_min_int_inc") -> pd.DataFrame:
        """
        Read and process profit_express data
        """
        df_profit_express = Query().from_("wind", "ashareprofitexpress") \
            .on("s_info_windcode").in_query(self.universe.wind_common.wind_codes_query)\
            .on("ann_dt").larger_than(self.universe.window_buffer_from_date)\
            .select("actual_ann_dt", "ann_dt", "report_period", "s_info_windcode", field).df()
        df_profit_express = df_profit_express[~(df_profit_express["actual_ann_dt"] > df_profit_express["ann_dt"])]
        df_profit_express = self.__fundamental_data_pre_process(df_profit_express,
                                                                season_mapping_col="report_period",
                                                                drop_dup_col_sort=["s_info_windcode", "ann_dt"],
                                                                drop_dup_col_keep_last="report_period"
                                                                )
        return df_profit_express

    def _load_profit_notice(self, deal_profit_notice="avg") -> pd.DataFrame:
        """
        Read and process profit_express data
        """
        df_profit_notice = Query().from_("wind", "ashareprofitnotice")\
            .on("s_info_windcode").in_query(self.universe.wind_common.wind_codes_query) \
            .on("s_profitnotice_period").larger_than(self.universe.from_date)\
            .select("s_profitnotice_date", "s_profitnotice_period", "s_info_windcode",
                    "s_profitnotice_netprofitmin",
                    "s_profitnotice_netprofitmax",
                    "s_profitnotice_changemin",
                    "s_profitnotice_changemax",
                    "s_profitnotice_net_parent_firm")\
            .df()
        df_profit_notice = self.__fundamental_data_pre_process(df_profit_notice,
                                                               season_mapping_col="s_profitnotice_period",
                                                               drop_dup_col_sort=["s_info_windcode", "s_profitnotice_date"],
                                                               drop_dup_col_keep_last="s_profitnotice_period"
                                                               )
        # Calculate netprofit value from change% if only the change% published. Be careful about the np<0 case.
        df_profit_notice["calc_min"] = df_profit_notice["s_profitnotice_net_parent_firm"] \
                                       * ( 1 + df_profit_notice["s_profitnotice_changemin"]
                                           *np.sign(df_profit_notice["s_profitnotice_net_parent_firm"].fillna(0)) / 100)
        df_profit_notice["calc_max"] = df_profit_notice["s_profitnotice_net_parent_firm"] \
                                       * ( 1 + df_profit_notice["s_profitnotice_changemax"]
                                           *np.sign(df_profit_notice["s_profitnotice_net_parent_firm"].fillna(0)) / 100)
        df_profit_notice["s_profitnotice_netprofitmin"] = df_profit_notice["s_profitnotice_netprofitmin"].fillna(
            df_profit_notice["calc_min"])
        df_profit_notice["s_profitnotice_netprofitmax"] = df_profit_notice["s_profitnotice_netprofitmax"].fillna(
            df_profit_notice["calc_max"])
        df_profit_notice_syn = {"s_info_windcode": df_profit_notice["s_info_windcode"],
                                "ann_dt": df_profit_notice["s_profitnotice_date"],
                                "report_period": df_profit_notice["s_profitnotice_period"],
                                }
        # If only min/max noticed, fill na max/min. TBD here...
        df_profit_notice["s_profitnotice_netprofitmin"] = df_profit_notice["s_profitnotice_netprofitmin"].fillna(
            df_profit_notice["s_profitnotice_netprofitmax"])
        df_profit_notice["s_profitnotice_netprofitmax"] = df_profit_notice["s_profitnotice_netprofitmax"].fillna(
            df_profit_notice["s_profitnotice_netprofitmin"])

        if deal_profit_notice == "avg":
            df_profit_notice_syn["net_profit_excl_min_int_inc"] = (df_profit_notice["s_profitnotice_netprofitmin"]
                                                                   + df_profit_notice["s_profitnotice_netprofitmax"])/2
        elif deal_profit_notice == "min":
            df_profit_notice_syn["net_profit_excl_min_int_inc"] = df_profit_notice["s_profitnotice_netprofitmin"]
        elif deal_profit_notice == "max":
            df_profit_notice_syn["net_profit_excl_min_int_inc"] = df_profit_notice["s_profitnotice_netprofitmax"]
        df_profit_notice_syn["net_profit_excl_min_int_inc"] *= 10000
        return pd.DataFrame(df_profit_notice_syn)

    def __fundamental_data_pre_process(self, df: pd.DataFrame,
                                       season_mapping_col: str,
                                       drop_dup_col_sort: list,
                                       drop_dup_col_keep_last: str) -> pd.DataFrame:
        season_map = [1231, 1231, 331, 331, 331, 630, 630, 630, 930, 930, 930, 1231]
        def season_mapping(date: datetime.date):
            year = date.year
            month = date.month
            season = season_map[month - 1]
            if month in [1, 2]:
                year -= 1
            return year * 10000 + season # Number calculation should be faster than string concatnation.
        df[season_mapping_col] = df[season_mapping_col].apply(season_mapping)
        df = fundamental_drop_duplicates(df, col_sort=drop_dup_col_sort, col_keep_last=drop_dup_col_keep_last)
        return df

    def _pivot_and_align(self, df: pd.DataFrame, values: str, columns: str, index: str, fillna=True) -> pd.DataFrame:
        df = fundamental_pivot(df, values, columns, index, fillna)
        df = self.universe.align_df_with_trade_dates_with_window_buffer_and_symbols(df)
        return df


class FundamentalHonestNetprofit(FudamentalCommon):
    def __init__(self, universe: Universe) -> None:
        self.universe = universe

    def get_fundamental_honest_netprofit_net_profit_excl_min_int_inc_df(self, deal_profit_notice='avg') -> pd.DataFrame:
        # 1) net_profit_excl_min_int_inc from income sheet -- data point
        np_ashareincome = self._load_ashareincome_field(field="net_profit_excl_min_int_inc")
        # 2) net_profit_excl_min_int_inc from profit express -- data point
        np_profit_express = self._load_profit_express_field(field="net_profit_excl_min_int_inc")
        # 3) net_profit_excl_min_int_inc from profit notice -- data range
        np_profit_notice = self._load_profit_notice(deal_profit_notice=deal_profit_notice)
        np_ashareincome["tag"] = "i"
        np_profit_express["tag"] = "e"
        np_profit_notice["tag"] = "p"
        np_combined = pd.concat([np_ashareincome, np_profit_express, np_profit_notice])
        return np_combined

    def get_fundamental_honest_netprofit(self, deal_profit_notice='avg', np_combined=None) -> tuple:
        if np_combined is None:
            np_combined = self.get_fundamental_honest_netprofit_net_profit_excl_min_int_inc_df(deal_profit_notice)
        np_combined = np_combined.dropna(subset=["net_profit_excl_min_int_inc"])
        np_combined = fundamental_drop_duplicates(np_combined,
                                                                ["s_info_windcode", "ann_dt"], "report_period")
        #只取最新期限的预告/快报数据，不考虑过期修正
        flag = True
        while flag:
            revised_signal = (np_combined["s_info_windcode"] == np_combined.shift(1)["s_info_windcode"]) \
                              & (np_combined["report_period"] < np_combined.shift(1)["report_period"])
            np_combined = np_combined[~revised_signal]
            flag = (revised_signal.sum() > 0)

        np_combined_np = self._pivot_and_align(np_combined, values="net_profit_excl_min_int_inc",
                                                       columns="s_info_windcode", index="ann_dt")
        np_combined_period = self._pivot_and_align(np_combined, values="report_period",
                                                       columns="s_info_windcode", index="ann_dt")
        np_combined_tag = self._pivot_and_align(np_combined, values="tag",
                                                       columns="s_info_windcode", index="ann_dt")
        return np_combined_np, np_combined_period, np_combined_tag

    def get_ashareincome_netprofit_excl_min_int_inc(self) -> tuple:
        net_profit = self._load_ashareincome_field(field="net_profit_excl_min_int_inc")
        flag = True
        while flag:
            revised_signal = (net_profit["s_info_windcode"] == net_profit.shift(1)["s_info_windcode"]) \
                              & (net_profit["report_period"] < net_profit.shift(1)["report_period"])
            net_profit = net_profit[~revised_signal]
            flag = (revised_signal.sum() > 0)
        np_combined_np = self._pivot_and_align(net_profit, values="net_profit_excl_min_int_inc",
                                                       columns="s_info_windcode", index="ann_dt")
        np_combined_period = self._pivot_and_align(net_profit, values="report_period",
                                                       columns="s_info_windcode", index="ann_dt")
        return np_combined_np, np_combined_period


class FundamentalHonestOperFields(FudamentalCommon):
    def __init__(self, universe: Universe, field: str) -> None:
        self.universe = universe
        self.field = field
        self.funda_np = FundamentalHonestNetprofit(universe)

    def get_fundamental_honest_oper_df(self, deal_profit_notice='avg') -> pd.DataFrame:
        # 1) oper field from income sheet -- data point
        df_ashareincome = self._load_ashareincome_field(field=self.field)
        # 2) oper field from profit express -- data point
        df_profit_express = self._load_profit_express_field(field=self.field)
        df_ashareincome["tag"] = "i"
        df_profit_express["tag"] = "e"
        df_combined = pd.concat([df_ashareincome, df_profit_express])

        # 3) oper field calc from np of profit notice * np margin -- data range
        np_combined = self.funda_np.get_fundamental_honest_netprofit_net_profit_excl_min_int_inc_df(deal_profit_notice=deal_profit_notice)
        df_combined = df_combined.merge(np_combined, how="right")
        df_combined["margin"] = df_combined["net_profit_excl_min_int_inc"] / df_combined[self.field]

        df_combined = df_combined.sort_values(["s_info_windcode", "ann_dt", "report_period"])
        df_combined["margin"] = df_combined.groupby(["s_info_windcode"]).fillna(method="pad")["margin"]

        df_combined[self.field] = df_combined[self.field].fillna(df_combined["net_profit_excl_min_int_inc"]/df_combined["margin"])

        return df_combined

    def get_fundamental_honest_netprofit(self, deal_profit_notice='avg', np_combined=None) -> tuple:
        if np_combined is None:
            np_combined = self.get_fundamental_honest_oper_df(deal_profit_notice)
        np_combined = np_combined.dropna(subset=["net_profit_excl_min_int_inc"])
        np_combined = fundamental_drop_duplicates(np_combined,
                                                                ["s_info_windcode", "ann_dt"], "report_period")
        #只取最新期限的预告/快报数据，不考虑过期修正
        flag = True
        while flag:
            revised_signal = (np_combined["s_info_windcode"] == np_combined.shift(1)["s_info_windcode"]) \
                              & (np_combined["report_period"] < np_combined.shift(1)["report_period"])
            np_combined = np_combined[~revised_signal]
            flag = (revised_signal.sum() > 0)

        np_combined_np = self._pivot_and_align(np_combined, values="net_profit_excl_min_int_inc",
                                                       columns="s_info_windcode", index="ann_dt")
        np_combined_period = self._pivot_and_align(np_combined, values="report_period",
                                                       columns="s_info_windcode", index="ann_dt")
        np_combined_tag = self._pivot_and_align(np_combined, values="tag",
                                                       columns="s_info_windcode", index="ann_dt")
        return np_combined_np, np_combined_period, np_combined_tag


class ConsensusCommon(FudamentalCommon):
    def __init__(self, universe: Universe, func_class: str, parallel_processes=1, consensus_data_start_window=90) -> None:
        self.universe = universe
        self.func_class = func_class
        self.parallel_processes = parallel_processes
        self.consensus_data_start_window = consensus_data_start_window #3300 #
        self.field = None
        self._consensus_data = {}
        self._consensus_data["Field"] = None

    @abc.abstractmethod
    def get_consensus_data(self):
        pass

    def get_consensus_data_transform(self, con_data=None, field=None, fy=0) -> pd.DataFrame:
        if con_data is None:
            con_data = self.get_consensus_data()
            con_data = self._consensus_map_forcast_year(con_data)
            field = self.field
        con_data_this_fy = con_data[con_data["con_year_index"] == fy]
        con_data_this_fy = con_data_this_fy.pivot(values=field, index="con_date", columns="stock_code")
        return self.universe.align_df_with_trade_dates_with_window_buffer_and_symbols(con_data_this_fy)

    def get_consensus_np_mapped(self, con_np):
        con_np = self._consensus_map_forcast_year(con_np)
        return con_np

    def _calc_consensus_data_by_rpts_effective_in_3m(self, np_combined, rpts):
        rpts = self._consensus_append_announcements(con=rpts, ann=np_combined, field=self.field, ann_mask=None)
        rpts = rpts.dropna(subset=[self.field])

        self._consensus_data = {}
        self._consensus_data["Field"] = self.field
        self._consensus_data["Data"] = []
        pool = mp.Pool(processes=self.parallel_processes)
        for date in self.universe.get_trade_dates_with_window_buffer()[self.consensus_data_start_window:]:
            date = datetime.date(date//10000, (date%10000)//100, (date%10000)%100)
            logging.info(f"{self.func_class}._calc_consensus_data_by_rpts_effective_in_3m run to {date}")
            rpts_here = rpts[(rpts["updatetime"] <= date) & (rpts["updatetime"] > date-datetime.timedelta(days=90))]
            context = {}
            context["df"] = rpts_here
            context["date"] = date
            context["field"] = self.field
            context["rpts_range"] = "3m"
            pool.apply_async(eval(f"{self.func_class}.consensus_calc_mean_each_upddate"),
                             args=(context,),
                             callback=self.create_consensus_data_callback()
                             )
        pool.close()
        pool.join()

    def _add_consensus_data_by_rpts_in_6m(self, rpts):
        consensus_data = pd.DataFrame(self._consensus_data["Data"])
        pool = mp.Pool(processes=self.parallel_processes)
        for dt in self.universe.get_trade_dates_with_window_buffer()[self.consensus_data_start_window:]:
            dt = datetime.date(dt // 10000, (dt % 10000) // 100, (dt % 10000) % 100)
            logging.info(f"{self.func_class}._add_consensus_data_by_rpts_in_6m run to {dt}")
            consensus_data_this_day = consensus_data[consensus_data["updatetime"] == dt]
            stocks_have_con_data = set(consensus_data_this_day["stock_code"])
            rpts_here = rpts[(rpts["updatetime"] <= dt-datetime.timedelta(days=90)) & (rpts["updatetime"] > dt-datetime.timedelta(days=180))]
            rpts_here = rpts_here[~rpts_here["stock_code"].isin(stocks_have_con_data)]

            context = {}
            context["df"] = rpts_here
            context["date"] = dt-datetime.timedelta(days=90)
            context["field"] = self.field
            context["rpts_range"] = "6m"
            pool.apply_async(eval(f"{self.func_class}.consensus_calc_mean_each_upddate"),
                             args=(context,),
                             callback=self.create_consensus_data_callback()
                             )
        pool.close()
        pool.join()

    @staticmethod
    def consensus_generate_fy0(upd_date: datetime.date) -> int:
        if upd_date.month < 5:
            return upd_date.year -1
        else:
            return upd_date.year

    @staticmethod
    def consensus_monthly_mean(rpts:pd.DataFrame, upd_date: datetime.date, field:str) -> pd.DataFrame:
        df_here_3m = rpts[(rpts["updatetime"] <= upd_date - datetime.timedelta(days=60))]
        df_here_2m = rpts[(rpts["updatetime"] <= upd_date - datetime.timedelta(days=30)) & (
                rpts["updatetime"] > upd_date - datetime.timedelta(days=60))]
        df_here_1m = rpts[(rpts["updatetime"] > upd_date - datetime.timedelta(days=30))]

        con_df_3m = df_here_3m.groupby(["stock_code", "report_year"]).mean()[field]
        con_df_2m = df_here_2m.groupby(["stock_code", "report_year"]).mean()[field]
        con_df_1m = df_here_1m.groupby(["stock_code", "report_year"]).mean()[field]

        con_df = pd.DataFrame({"3m": con_df_3m,
                               "2m": con_df_2m,
                               "1m": con_df_1m})
        return con_df

    def create_consensus_data_callback(self):
        def _consensus_data_callback(con_data_here):
            self._consensus_data["Data"].extend(con_data_here)
        return _consensus_data_callback

    def _consensus_map_forcast_year(self, con_data: pd.DataFrame) -> pd.DataFrame:
        con_year_index = Query().from_("suntime", "con_year_index").select("con_date", "con_year", "con_year_index").df()
        con_data = con_data.rename(columns={"updatetime": "con_date", "report_year": "con_year"})
        con_data = con_data.merge(con_year_index, how="left")
        return con_data

    def _load_consensus_data(self, field: str, dropna=True, unit=None) -> pd.DataFrame:
        query_str = f"select stock_code, organ_id, \
                             toDate(create_date, 'Asia/Shanghai') as create_date, \
                             toDate(updatetime, 'Asia/Shanghai') as updatetime, report_year, {field} " \
                    "from suntime.rpt_forecast_stk " \
                    "where report_quarter=4 and (reliability>=5 or not report_type=21) " \
                    f"and toYYYYMMDD(create_date, 'Asia/Shanghai') >= '{self.universe.window_buffer_from_date}'" \
                    "order by stock_code asc, create_date asc, report_year asc"

        df, col = get_clickhouse_client("suntime").execute(query_str, with_column_types=True)
        df = pd.DataFrame(df, columns=[i[0] for i in col])
        df.loc[df["create_date"] < datetime.date(2017, 12, 31), "updatetime"] = df[df["create_date"] < datetime.date(2017, 12, 31)]["create_date"]
        if dropna:
            df = df.dropna(subset=[field])
        if unit is not None:
            df[field] *= unit
        return df

    def _consensus_append_announcements(self, con: pd.DataFrame, ann: pd.DataFrame, field: str, ann_mask=None) -> pd.DataFrame:
        ann["report_year"] = ann["report_period"] // 10000
        ann["report_quarter"] = ann["report_period"] % 10000
        ann = ann.drop("report_period", axis=1)
        ann["create_date"] = ann["ann_dt"]
        ann = ann.rename(columns={"ann_dt": "updatetime",
                                  "s_info_windcode": "stock_code",
                                  "ann_data": field,
                                  "tag": "organ_id"})
        con["report_quarter"] = 1231
        if ann_mask is not None:
            ann[field] = ann_mask
        df = pd.concat([con, ann])
        return df


class ConsensusNetProfitProcessor(ConsensusCommon):

    def __init__(self, universe: Universe, field="forecast_np", unit=10000, parallel_processes=1) -> None:
        super().__init__(universe=universe, func_class="ConsensusNetProfitProcessor", parallel_processes=parallel_processes)
        self.field = field
        self.unit = unit
        self.funda = FundamentalHonestNetprofit(universe)

    def get_consensus_data(self):
        if self._consensus_data["Field"] != self.field:
            np_combined = self.funda.get_fundamental_honest_netprofit_net_profit_excl_min_int_inc_df(
                deal_profit_notice="min")
            np_combined = np_combined.rename(columns={"net_profit_excl_min_int_inc": "ann_data"})
            rpts = self._load_consensus_data(field=self.field, unit=self.unit)
            self._calc_consensus_data_by_rpts_effective_in_3m(np_combined=np_combined, rpts=rpts)
            # 若3个月内没有有效报告，则采用6个月内的报告
            self._add_consensus_data_by_rpts_in_6m(rpts)
        return pd.DataFrame(self._consensus_data["Data"])

    @staticmethod
    def consensus_calc_mean_each_upddate(context):
        df = context["df"]
        date = context["date"]
        field = context["field"]
        rpts_range = context["rpts_range"]
        df = fundamental_drop_duplicates(df, ["stock_code", "report_year", "organ_id"], "create_date")
        if rpts_range == "3m":
            df = ConsensusNetProfitProcessor.consensus_select_reports_later_than_company_announcements(df, date)
        con_df = ConsensusNetProfitProcessor.consensus_monthly_mean(rpts=df, upd_date=date, field=field)
        con_df = con_df.T.fillna(method="bfill").fillna(method="ffill").T
        con_data_here = (con_df["3m"] * 1 + con_df["2m"] * 2 + con_df["1m"] * 4) / (1 + 2 + 4)
        con_data_here = con_data_here.reset_index().rename(columns={0: field})
        if rpts_range == "3m":
            con_data_here["updatetime"] = date
        elif rpts_range == "6m":
            con_data_here["updatetime"] = date + datetime.timedelta(days=90)
        return list(con_data_here.T.to_dict().values())

    @staticmethod
    def consensus_select_reports_later_than_company_announcements(rpts: pd.DataFrame, upd_date=None) -> pd.DataFrame:
        rpts = rpts.copy(deep=True)
        if upd_date is None:
            upd_date = rpts["updatetime"].max()
        fy0 = ConsensusNetProfitProcessor.consensus_generate_fy0(upd_date)
        rpts = rpts[rpts["report_year"] >= fy0]
        rpts["organ_signal"] = rpts["organ_id"].apply(lambda x: 1.0 if x in ["i", "e", "p"] else 0)
        rpts = rpts.sort_values(["stock_code", "updatetime", "organ_signal"], ascending=[True, True, False])
        rpts["ann_signal"] = rpts["organ_id"].apply(lambda x: 1.0 if x in ["i", "e", "p"] else np.nan)
        rpts["ann_signal"] = rpts.groupby("stock_code")["ann_signal"].fillna(method="ffill").fillna(0).values

        #未发过公告的情况
        ann_signal = rpts.groupby("stock_code")["ann_signal"].max()
        rpts_without_ann_stock_code = ann_signal[ann_signal==0].index
        rpts_without_ann = rpts[rpts["stock_code"].isin(rpts_without_ann_stock_code)]
        #发过公告的情况
        rpts_with_ann = rpts[~rpts["stock_code"].isin(rpts_without_ann_stock_code)].copy(deep=True)
        rpts_with_ann = rpts_with_ann[rpts_with_ann["ann_signal"] != 0]

        #如果发过多份公告，多份公告都保留，而剔除多份公告之间的预测报告
        rpts_with_ann.loc[:, "ann_signal"] = rpts_with_ann.groupby("stock_code")["organ_signal"].transform(pd.Series.cumsum)
        idx_after_latest_ann = rpts_with_ann.groupby(["stock_code"])["ann_signal"].transform(max) == rpts_with_ann["ann_signal"]
        rpts_with_ann_before_latest_ann = rpts_with_ann[~idx_after_latest_ann]
        rpts_with_ann_before_latest_ann = rpts_with_ann_before_latest_ann[rpts_with_ann_before_latest_ann["organ_signal"]==1]
        rpts_with_ann = pd.concat([rpts_with_ann_before_latest_ann, rpts_with_ann[idx_after_latest_ann]])
        #如果有fy0的公告数据，则采用公告，剔除fy0预测报告数据
        rpts_with_ann_fyo_announced_stock_code = rpts_with_ann[((rpts_with_ann["organ_signal"]==1)&(rpts_with_ann["report_year"]==fy0))
                                                               &(rpts_with_ann["report_quarter"]==1231)]["stock_code"]
        if len(rpts_with_ann_fyo_announced_stock_code) < len(set(rpts_with_ann["stock_code"])):
            rpts_with_ann_fyo_announced = rpts_with_ann[rpts_with_ann["stock_code"].isin(rpts_with_ann_fyo_announced_stock_code)]
            rpts_with_ann_fyo_announced = rpts_with_ann_fyo_announced[~((rpts_with_ann_fyo_announced["report_year"]==fy0)&(rpts_with_ann_fyo_announced["organ_signal"]!=1))]
            rpts_without_ann_fyo_announced = rpts_with_ann[~rpts_with_ann["stock_code"].isin(rpts_with_ann_fyo_announced_stock_code)]
            rpts_with_ann = pd.concat([rpts_with_ann_fyo_announced, rpts_without_ann_fyo_announced])
        else:
            rpts_with_ann = rpts_with_ann[~((rpts_with_ann["report_year"]==fy0)&(rpts_with_ann["organ_signal"]!=1))]
        #仅保留年度数据
        rpts_with_ann = rpts_with_ann[rpts_with_ann["report_quarter"] == 1231]
        rpts = pd.concat([rpts_with_ann, rpts_without_ann])
        return rpts


class ConsensusOperFieldProcessor(ConsensusCommon):

    def __init__(self, universe: Universe, field="forecast_or", unit=10000, parallel_processes=1) -> None:
        super().__init__(universe=universe, func_class="ConsensusOperFieldProcessor", parallel_processes=parallel_processes)
        self.field = field
        self.unit = unit
        self.funda = FundamentalHonestOperFields(universe=universe, field=self.map_suntime_field_to_announcements(self.field))

    def map_suntime_field_to_announcements(self, suntime_field: str) -> str:
        mapping = {"forecast_or": "oper_rev",
                   "forecast_op": "oper_profit",
                   "forecast_tp": "tot_profit"}
        return mapping[suntime_field]

    def get_consensus_data(self):
        if self._consensus_data["Field"] != self.field:
            df_combined = self.funda.get_fundamental_honest_oper_df(deal_profit_notice="min")
            df_combined = df_combined.rename(columns={self.map_suntime_field_to_announcements(self.field): "ann_data"})
            rpts = self._load_consensus_data(field=self.field, unit=self.unit)
            self._calc_consensus_data_by_rpts_effective_in_3m(np_combined=df_combined, rpts=rpts)
            # 若3个月内没有有效报告，则采用6个月内的报告
            self._add_consensus_data_by_rpts_in_6m(rpts)
        return pd.DataFrame(self._consensus_data["Data"])

    @staticmethod
    def consensus_calc_mean_each_upddate(context):
        df = context["df"]
        date = context["date"]
        field = context["field"]
        rpts_range = context["rpts_range"]
        df = fundamental_drop_duplicates(df, ["stock_code", "report_year", "organ_id"], "create_date")
        if rpts_range == "3m":
            df = ConsensusNetProfitProcessor.consensus_select_reports_later_than_company_announcements(df, date)
        con_df = ConsensusNetProfitProcessor.consensus_monthly_mean(rpts=df, upd_date=date, field=field)
        con_df = con_df.T.fillna(method="bfill").fillna(method="ffill").T
        con_data_here = (con_df["3m"] * 1 + con_df["2m"] * 2 + con_df["1m"] * 4) / (1 + 2 + 4)
        con_data_here = con_data_here.reset_index().rename(columns={0: field})
        if rpts_range == "3m":
            con_data_here["updatetime"] = date
        elif rpts_range == "6m":
            con_data_here["updatetime"] = date + datetime.timedelta(days=90)
        return list(con_data_here.T.to_dict().values())

    @staticmethod
    def consensus_select_reports_later_than_company_announcements(rpts: pd.DataFrame, upd_date=None) -> pd.DataFrame:
        rpts = rpts.copy(deep=True)
        if upd_date is None:
            upd_date = rpts["updatetime"].max()
        fy0 = ConsensusNetProfitProcessor.consensus_generate_fy0(upd_date)
        rpts = rpts[rpts["report_year"] >= fy0]
        rpts["organ_signal"] = rpts["organ_id"].apply(lambda x: 1.0 if x in ["i", "e", "p"] else 0)
        rpts = rpts.sort_values(["stock_code", "updatetime", "organ_signal"], ascending=[True, True, False])
        rpts["ann_signal"] = rpts["organ_id"].apply(lambda x: 1.0 if x in ["i", "e", "p"] else np.nan)
        rpts["ann_signal"] = rpts.groupby("stock_code")["ann_signal"].fillna(method="ffill").fillna(0).values

        #未发过公告的情况
        ann_signal = rpts.groupby("stock_code")["ann_signal"].max()
        rpts_without_ann_stock_code = ann_signal[ann_signal==0].index
        rpts_without_ann = rpts[rpts["stock_code"].isin(rpts_without_ann_stock_code)]
        #发过公告的情况
        rpts_with_ann = rpts[~rpts["stock_code"].isin(rpts_without_ann_stock_code)].copy(deep=True)
        rpts_with_ann = rpts_with_ann[rpts_with_ann["ann_signal"] != 0]

        #如果发过多份公告，多份公告都保留，而剔除多份公告之间的预测报告
        rpts_with_ann.loc[:, "ann_signal"] = rpts_with_ann.groupby("stock_code")["organ_signal"].transform(pd.Series.cumsum)
        idx_after_latest_ann = rpts_with_ann.groupby(["stock_code"])["ann_signal"].transform(max) == rpts_with_ann["ann_signal"]
        rpts_with_ann_before_latest_ann = rpts_with_ann[~idx_after_latest_ann]
        rpts_with_ann_before_latest_ann = rpts_with_ann_before_latest_ann[rpts_with_ann_before_latest_ann["organ_signal"]==1]
        rpts_with_ann = pd.concat([rpts_with_ann_before_latest_ann, rpts_with_ann[idx_after_latest_ann]])
        #如果有fy0的公告数据，则采用公告，剔除fy0预测报告数据
        rpts_with_ann_fyo_announced_stock_code = rpts_with_ann[((rpts_with_ann["organ_signal"]==1)&(rpts_with_ann["report_year"]==fy0))
                                                               &(rpts_with_ann["report_quarter"]==1231)]["stock_code"]
        if len(rpts_with_ann_fyo_announced_stock_code) < len(set(rpts_with_ann["stock_code"])):
            rpts_with_ann_fyo_announced = rpts_with_ann[rpts_with_ann["stock_code"].isin(rpts_with_ann_fyo_announced_stock_code)]
            rpts_with_ann_fyo_announced = rpts_with_ann_fyo_announced[~((rpts_with_ann_fyo_announced["report_year"]==fy0)&(rpts_with_ann_fyo_announced["organ_signal"]!=1))]
            rpts_without_ann_fyo_announced = rpts_with_ann[~rpts_with_ann["stock_code"].isin(rpts_with_ann_fyo_announced_stock_code)]
            rpts_with_ann = pd.concat([rpts_with_ann_fyo_announced, rpts_without_ann_fyo_announced])
        else:
            rpts_with_ann = rpts_with_ann[~((rpts_with_ann["report_year"]==fy0)&(rpts_with_ann["organ_signal"]!=1))]
        #仅保留年度数据
        rpts_with_ann = rpts_with_ann[rpts_with_ann["report_quarter"] == 1231]
        rpts = pd.concat([rpts_with_ann, rpts_without_ann])
        return rpts


class ConsensusNonNetProfitProcessor(ConsensusCommon):

    def __init__(self, universe: Universe, field: str, unit=None, parallel_processes=1) -> None:
        super().__init__(universe=universe, func_class="ConsensusNonNetProfitProcessor", parallel_processes=parallel_processes)
        self.field = field
        self.unit = unit

    def get_consensus_data(self):
        if self._consensus_data["Field"] != self.field:
            rpts = self._load_consensus_data(field=self.field, unit=self.unit)
            self._calc_consensus_np_by_rpts_effective_in_3m(rpts)
            # 若3个月内没有有效报告，则采用6个月内的报告
            self._add_consensus_np_by_rpts_in_6m(rpts)
        return pd.DataFrame(self._consensus_data["Data"])

    @staticmethod
    def consensus_calc_mean_each_upddate(context):
        df = context["df"]
        date = context["date"]
        field = context["field"]
        rpts_range = context["rpts_range"]
        df = fundamental_drop_duplicates(df, ["stock_code", "report_year", "organ_id"], "create_date")
        if rpts_range == "3m":
            # df = ConsensusNetProfitProcessor.consensus_select_reports_later_than_company_announcements(df, date)
            df = df[~df["organ_id"].isin(["i", "p", "e"])]
        con_df = ConsensusNetProfitProcessor.consensus_monthly_mean(rpts=df, upd_date=date, field=field)
        con_df = con_df.T.fillna(method="bfill").fillna(method="ffill").T
        con_data_here = (con_df["3m"] * 1 + con_df["2m"] * 2 + con_df["1m"] * 4) / (1 + 2 + 4)
        con_data_here = con_data_here.reset_index().rename(columns={0: field})
        if rpts_range == "3m":
            con_data_here["updatetime"] = date
        elif rpts_range == "6m":
            con_data_here["updatetime"] = date + datetime.timedelta(days=90)
        return list(con_data_here.T.to_dict().values())

    @staticmethod
    def consensus_select_reports_later_than_company_announcements(rpts: pd.DataFrame, upd_date=None) -> pd.DataFrame:
        rpts = rpts.copy(deep=True)
        if upd_date is None:
            upd_date = rpts["updatetime"].max()
        fy0 = ConsensusNetProfitProcessor.consensus_generate_fy0(upd_date)
        rpts = rpts[rpts["report_year"] >= fy0]
        rpts["organ_signal"] = rpts["organ_id"].apply(lambda x: 1.0 if x in ["i", "e", "p"] else 0)
        rpts = rpts.sort_values(["stock_code", "updatetime", "organ_signal"], ascending=[True, True, False])
        rpts["ann_signal"] = rpts.groupby("stock_code")["organ_signal"].transform(pd.Series.cumsum)
        idx_after_latest_ann = rpts.groupby(["stock_code"])["ann_signal"].transform(max) == rpts["ann_signal"]
        rpts = rpts[idx_after_latest_ann]

        rpts["ann_signal"] = rpts["organ_id"].apply(lambda x: 1.0 if x in ["i", "e", "p"] else np.nan)
        rpts["ann_signal"] = rpts.groupby("stock_code")["ann_signal"].fillna(method="ffill").fillna(0).values
        rpts = rpts[rpts["organ_signal"] == 0]
        return rpts
