import pandas as pd
import datetime
from chameqs_common.utils.decorator import timeit
from chameqs_common.data.query import Query
from chameqs_common.data.universe import Universe
from chameqs_common.utils.date_util import int_to_date


@timeit()
def get_mutual_fund_stock_position_median(universe:Universe, mutual_fund_stock_position):
    mutual_fund_stock_position = mutual_fund_stock_position.reset_index()
    mutual_fund_stock_position_median = {}
    for date in universe.get_trade_dates_with_window_buffer():
        print(date)
        mutual_fund_stock_position_as_of_date = mutual_fund_stock_position[(mutual_fund_stock_position["ann_date"] <= int_to_date(date))
                                                                           &(mutual_fund_stock_position["ann_date"] >= int_to_date(date)-datetime.timedelta(days=120))]
        if len(mutual_fund_stock_position_as_of_date) == 0:
            continue
        mutual_fund_stock_position_as_of_date_latest_idx = mutual_fund_stock_position_as_of_date.groupby("level_0")["ann_date"].transform(max) == mutual_fund_stock_position_as_of_date["ann_date"]
        mutual_fund_stock_position_as_of_date_latest = mutual_fund_stock_position_as_of_date[mutual_fund_stock_position_as_of_date_latest_idx]
        mutual_fund_stock_position_median[date] = mutual_fund_stock_position_as_of_date_latest[["s_info_stockwindcode", 0]].groupby("s_info_stockwindcode").median()
    return pd.concat(mutual_fund_stock_position_median)[0].unstack()

@timeit()
def get_mutual_fund_stock_position_quantile(universe:Universe, mutual_fund_stock_position, q):
    mutual_fund_stock_position = mutual_fund_stock_position.reset_index()
    mutual_fund_stock_position[0] = mutual_fund_stock_position[0].astype(float)
    mutual_fund_stock_position_median = {}
    for date in universe.get_trade_dates_with_window_buffer():
        print(date)
        mutual_fund_stock_position_as_of_date = mutual_fund_stock_position[(mutual_fund_stock_position["ann_date"] <= int_to_date(date))
                                                                           &(mutual_fund_stock_position["ann_date"] >= int_to_date(date)-datetime.timedelta(days=120))]
        if len(mutual_fund_stock_position_as_of_date) == 0:
            continue
        mutual_fund_stock_position_as_of_date_latest_idx = mutual_fund_stock_position_as_of_date.groupby("level_0")["ann_date"].transform(max) == mutual_fund_stock_position_as_of_date["ann_date"]
        mutual_fund_stock_position_as_of_date_latest = mutual_fund_stock_position_as_of_date[mutual_fund_stock_position_as_of_date_latest_idx]
        mutual_fund_stock_position_median[date] = mutual_fund_stock_position_as_of_date_latest[["s_info_stockwindcode", 0]].groupby("s_info_stockwindcode").quantile(q)
    return pd.concat(mutual_fund_stock_position_median)[0].unstack()

@timeit()
def get_mutual_fund_stock_position_count(universe:Universe, mutual_fund_stock_position, field="hodings_to_free_shares"):
    mutual_fund_stock_position[field] = mutual_fund_stock_position[field].astype(float)
    mutual_fund_stock_position_median = {}
    for date in universe.get_trade_dates_with_window_buffer():
        print(date)
        mutual_fund_stock_position_as_of_date = mutual_fund_stock_position[(mutual_fund_stock_position["ann_date"] <= int_to_date(date))
                                                                           &(mutual_fund_stock_position["ann_date"] >= int_to_date(date)-datetime.timedelta(days=120))]
        if len(mutual_fund_stock_position_as_of_date) == 0:
            continue
        mutual_fund_stock_position_as_of_date_latest_idx = mutual_fund_stock_position_as_of_date.groupby("s_info_windcode")["ann_date"].transform(max) == mutual_fund_stock_position_as_of_date["ann_date"]
        mutual_fund_stock_position_as_of_date_latest = mutual_fund_stock_position_as_of_date[mutual_fund_stock_position_as_of_date_latest_idx]
        mutual_fund_stock_position_median[date] = mutual_fund_stock_position_as_of_date_latest[["s_info_stockwindcode", field]].groupby("s_info_stockwindcode").count()
    return pd.concat(mutual_fund_stock_position_median)[field].unstack()

@timeit()
def get_mutual_fund_stock_position_mkv(universe:Universe, mutual_fund_stock_position, field="hodings_to_free_shares"):
    mutual_fund_stock_position[field] = mutual_fund_stock_position[field].astype(float)
    mutual_fund_stock_position_median = {}
    for date in universe.get_trade_dates_with_window_buffer():
        print(date)
        mutual_fund_stock_position_as_of_date = mutual_fund_stock_position[(mutual_fund_stock_position["ann_date"] <= pd.to_datetime(str(date)))
                                                                           &(mutual_fund_stock_position["ann_date"] >= pd.to_datetime(str(date))-datetime.timedelta(days=120))]
        if len(mutual_fund_stock_position_as_of_date) == 0:
            continue
        mutual_fund_stock_position_as_of_date_latest_idx = mutual_fund_stock_position_as_of_date.groupby("s_info_windcode")["ann_date"].transform(max) == mutual_fund_stock_position_as_of_date["ann_date"]
        mutual_fund_stock_position_as_of_date_latest = mutual_fund_stock_position_as_of_date[mutual_fund_stock_position_as_of_date_latest_idx]
        mutual_fund_stock_position_median[date] = mutual_fund_stock_position_as_of_date_latest[["s_info_stockwindcode", field]].groupby("s_info_stockwindcode").sum()
    return pd.concat(mutual_fund_stock_position_median)[field].unstack()

@timeit()
def get_mutual_fund_stock_position_mkv_quantile(universe:Universe, mutual_fund_stock_position, q, field="hodings_to_free_shares"):
    mutual_fund_stock_position[field] = mutual_fund_stock_position[field].astype(float)
    mutual_fund_stock_position_median = {}
    for date in universe.get_trade_dates_with_window_buffer():
        print(date)
        mutual_fund_stock_position_as_of_date = mutual_fund_stock_position[(mutual_fund_stock_position["ann_date"] <= pd.to_datetime(str(date)))
                                                                           &(mutual_fund_stock_position["ann_date"] >= pd.to_datetime(str(date))-datetime.timedelta(days=120))]
        if len(mutual_fund_stock_position_as_of_date) == 0:
            continue
        mutual_fund_stock_position_as_of_date_latest_idx = mutual_fund_stock_position_as_of_date.groupby("s_info_windcode")["ann_date"].transform(max) == mutual_fund_stock_position_as_of_date["ann_date"]
        mutual_fund_stock_position_as_of_date_latest = mutual_fund_stock_position_as_of_date[mutual_fund_stock_position_as_of_date_latest_idx]
        mutual_fund_stock_position_as_of_date_latest.assign(stkvaluetonav=mutual_fund_stock_position_as_of_date_latest[
                                                                           "stkvaluetonav"] / mutual_fund_stock_position_as_of_date_latest.groupby(
            ["s_info_windcode"]).stkvaluetonav.transform(sum))
        mutual_fund_stock_position_as_of_date_quan_idx = mutual_fund_stock_position_as_of_date_latest.groupby("s_info_windcode")["stkvaluetonav"].transform(
            lambda x: x.quantile(1-q)) <= mutual_fund_stock_position_as_of_date_latest["stkvaluetonav"]
        mutual_fund_stock_position_as_of_date_quan = mutual_fund_stock_position_as_of_date_latest[mutual_fund_stock_position_as_of_date_quan_idx]
        mutual_fund_stock_position_median[date] = mutual_fund_stock_position_as_of_date_quan[["s_info_stockwindcode", field]].groupby("s_info_stockwindcode").sum()
    return pd.concat(mutual_fund_stock_position_median)[field].unstack()

def get_top10_stock_portfolio_weight(df):
    first_ann_idx = df.groupby(["s_info_windcode", "f_prt_enddate"])["ann_date"].transform(min)==df["ann_date"]
    first_ann_port = df[first_ann_idx]
    return first_ann_port

def get_one_fund_top10_stock_portfolio_weight(first_ann_port, fund_id):
    first_ann_port_this_fund = first_ann_port[first_ann_port["s_info_windcode"] == fund_id].drop_duplicates(["ann_date", "s_info_stockwindcode"])# !!!!Shouldn't have duplicates
    first_ann_port_this_fund = first_ann_port_this_fund.pivot(index="ann_date", columns="s_info_stockwindcode", values="values")
    return first_ann_port_this_fund

def get_total_stock_portfolio_weight(df):
    df_semi_annual_rpts = df[df["f_prt_enddate"].apply(lambda x: x.month % 6 == 0)]
    tot_weight = df_semi_annual_rpts.groupby(["f_prt_enddate", "s_info_windcode", "s_info_stockwindcode"]).sum()["values"]
    tot_weight_ana_date = df_semi_annual_rpts.groupby(["f_prt_enddate", "s_info_windcode"]).max()["ann_date"]
    return tot_weight.unstack(), tot_weight_ana_date.unstack()

def get_one_fund_total_stock_portfolio_weight(tot_weight, tot_weight_ana_date, fund_id):
    tot_weight_this_fund = tot_weight.loc[:, fund_id, :]
    tot_weight_ana_date_this_fund = tot_weight_ana_date.loc[:, fund_id]
    tot_weight_this_fund.index = tot_weight_ana_date_this_fund[tot_weight_this_fund.index]
    return tot_weight_this_fund

def get_total_stock_portfolio_weight_composite(tot_weight, tot_weight_ana_date, first_ann_port):
    all_fund_ids = set(first_ann_port["s_info_windcode"]).intersection(set(tot_weight.index.unique(1)))
    total_stock_portfolio_weight_composite = {}
    for fund_id in all_fund_ids:
        total_stock_portfolio_weight_this_fund = get_one_fund_total_stock_portfolio_weight(tot_weight, tot_weight_ana_date, fund_id)
        first_ann_port_this_fund = get_one_fund_top10_stock_portfolio_weight(first_ann_port, fund_id)
        if len(first_ann_port_this_fund) <= 4:
            continue
        print(fund_id)
        total_stock_portfolio_weight_composite_this_fund = get_one_fund_total_stock_portfolio_weight_composite(first_ann_port_this_fund,
                                                                                                                    total_stock_portfolio_weight_this_fund)
        total_stock_portfolio_weight_composite[fund_id] = total_stock_portfolio_weight_composite_this_fund
    return pd.concat(total_stock_portfolio_weight_composite)

def get_one_fund_total_stock_portfolio_weight_composite(top10_stock_portfolio_weight, total_stock_portfolio_weight):
    all_symbols = set(top10_stock_portfolio_weight.columns.to_list() + total_stock_portfolio_weight.columns.to_list())
    top10_stock_portfolio_weight = top10_stock_portfolio_weight.T.reindex(all_symbols).T
    total_stock_portfolio_weight = total_stock_portfolio_weight.T.reindex(all_symbols).T
    total_stock_portfolio_weight_composite = pd.DataFrame(index=top10_stock_portfolio_weight.index, columns=top10_stock_portfolio_weight.columns)
    total_stock_portfolio_weight_composite_val = total_stock_portfolio_weight_composite.values
    total_stock_portfolio_weight_ann_dates = total_stock_portfolio_weight.index
    for i, date in enumerate(top10_stock_portfolio_weight.index):
        if i <= 4:
            continue
        top10_stock_portfolio_weight_this_period = top10_stock_portfolio_weight.loc[date]
        total_stock_portfolio_weight_last_period_ann_date = total_stock_portfolio_weight_ann_dates[total_stock_portfolio_weight_ann_dates<date][-1]
        total_stock_portfolio_weight_last_period = total_stock_portfolio_weight.loc[total_stock_portfolio_weight_last_period_ann_date]
        sum_stock_portfolio_weight_last_period = total_stock_portfolio_weight_last_period.sum()
        sum_top10_stock_portfolio_weight_this_period = top10_stock_portfolio_weight_this_period.sum()
        bottom_stock_portfolio_weight_last_period = get_bottom_stock_weights_single_period(
            total_stock_portfolio_weight_last_period,
            sum_stock_portfolio_weight_last_period-sum_top10_stock_portfolio_weight_this_period)
        total_stock_portfolio_weight_composite_val_this_period = top10_stock_portfolio_weight_this_period.fillna(0) + bottom_stock_portfolio_weight_last_period.fillna(0)
        total_stock_portfolio_weight_composite_val[i, :] = total_stock_portfolio_weight_composite_val_this_period#/total_stock_portfolio_weight_composite_val_this_period.sum()
    total_stock_portfolio_weight_composite = total_stock_portfolio_weight_composite.unstack()
    total_stock_portfolio_weight_composite = total_stock_portfolio_weight_composite.dropna()
    return total_stock_portfolio_weight_composite[total_stock_portfolio_weight_composite>0]

def get_bottom_stock_weights_single_period(stock_weights, bottom_ratio):
    stock_weights_values = stock_weights.fillna(0).values
    stock_weights_values.sort()
    cum_stock_weights_values = stock_weights_values.cumsum()
    bottom_line_stock_weight_idx = cum_stock_weights_values.searchsorted(bottom_ratio)
    bottom_line_stock_weight = stock_weights_values[bottom_line_stock_weight_idx-1] # !!!!

    stock_weights[stock_weights > bottom_line_stock_weight] = 0
    return stock_weights

def get_bottom_stock_weights(stock_weights, bottom_ratio):
    stock_weights_values = stock_weights.copy(deep=True).values.sort(1)
    stock_weights_sorted_values = stock_weights.copy(deep=True).values.sort(1)
    cum_stock_weights_values = stock_weights_sorted_values.cumsum(1)
    for i in range(len(cum_stock_weights_values)):
        bottom_line_stock_weight_idx = cum_stock_weights_values[i, :].searchsorted(bottom_ratio)
        bottom_line_stock_weight = stock_weights_sorted_values[i, bottom_line_stock_weight_idx]
        stock_weights_values[i, stock_weights_values[i, :]>bottom_line_stock_weight] = 0
    return stock_weights
