import numpy as np
import pandas as pd
from chameqs_common.data.aindex import get_index_weights_by_name
from chameqs_common.utils.array_util import fill_empty_rows_with_previous_not_empty_row
from chameqs_common.data.barra import BARRA_FACTOR_NAMES
from chameqs_red.data.barra import get_specific_risk_df
from chameqs_common.data.query import Query


def get_citic_industry_name_map(universe):
    return universe.get_citics_highest_level_industries()["industry_name"].to_dict()


def get_csi1000_index_weights_on_factor_calculation_dates(universe, factor_calculation_dates):
    csi1000_index_weights = Query().from_("wind", "aindexcsi1000weight").\
        select("s_con_windcode", "trade_dt", "weight", "opdate").df()
    csi1000_index_weights = csi1000_index_weights.pivot(values="weight", index="trade_dt", columns="s_con_windcode")
    csi1000_index_weights.index = [int(i.strftime("%Y%m%d")) for i in csi1000_index_weights.index]
    csi1000_index_weights.columns = csi1000_index_weights.columns.astype(float).astype(int)
    csi1000_index_weights = universe.align_df_with_trade_dates_and_symbols(csi1000_index_weights).loc[csi1000_index_weights.index[0]:, :]
    return (csi1000_index_weights/100).shift(-1).reindex(factor_calculation_dates) #Effective date is T+1


def get_barra_industries_daily(universe):
    raw_barra_industries = universe.get_transform("barra", "am_marketdata_stock_barra_rsk", "ind").fillna(
        method="pad")
    return get_industries_one_hot_encoding(raw_barra_industries)


def get_citic_industries_daily(universe):
    raw_citic_industries = universe.get_citics_highest_level_industry_symbols().fillna(method="pad")
    return get_industries_one_hot_encoding(raw_citic_industries)


def get_barra_style_exposure_daily(universe, style_factors_list=None):
    barra_style_exposure = {}
    if style_factors_list is None:
        style_factors_list = BARRA_FACTOR_NAMES
    for f in style_factors_list:
        barra_style_exposure[f] = universe.align_df_with_trade_dates(
            universe.get_transform("barra", "am_marketdata_stock_barra_rsk", f).
                fillna(method="pad")).values
    return barra_style_exposure


def get_barra_specific_risk_daily(universe):
    barra_specific_risk = get_specific_risk_df()
    barra_specific_risk = barra_specific_risk.reindex(universe.get_trade_dates()).fillna(method="pad")
    barra_specific_risk = barra_specific_risk.T.reindex(universe.get_symbols().columns).T.fillna(method="pad") #Fillna??
    return barra_specific_risk.values


def get_index_weights_on_factor_calculation_dates(universe, factor_calculation_dates, bench_index_name_or_weight_path: str):
    if bench_index_name_or_weight_path.startswith("file://"):
        index_weights = pd.read_csv(bench_index_name_or_weight_path.removeprefix("file://"))
        index_weights = index_weights.set_index(index_weights.columns[0])
        index_weights.columns = index_weights.columns.astype(float).astype(int)
        index_weights = index_weights.sort_index(axis=1)
    else:
        index_weights = get_index_weights_by_name(bench_index_name_or_weight_path)
    index_weights = universe.align_df_with_trade_dates_and_symbols(index_weights)
    index_weights = fill_empty_rows_with_previous_not_empty_row(index_weights)
    if bench_index_name_or_weight_path == "CSI1000":
        index_weights.loc[:20151001] = np.nan
    return (index_weights / 100).shift(1).fillna(0).reindex(factor_calculation_dates)  # .values  # ??????


def get_industries_one_hot_encoding(raw_industries):
    raw_latest_industries = raw_industries.values[-1, :]
    latest_unique_industries = np.unique(raw_latest_industries[~np.isnan(raw_latest_industries)])
    industries = {}
    for industry_code in latest_unique_industries:
        industries[industry_code] = (raw_industries == industry_code).astype(int).values
    return latest_unique_industries, industries


def get_barra_industries_on_factor_calculation_dates(universe, factor_calculation_dates):
    raw_barra_industries = universe.get_transform("barra", "am_marketdata_stock_barra_rsk", "ind").fillna(
        method="pad").reindex(factor_calculation_dates)
    return get_industries_one_hot_encoding(raw_barra_industries)


def get_citic_industries_on_factor_calculation_dates(universe, factor_calculation_dates):
    raw_citic_industries = universe.get_citics_highest_level_industry_symbols().fillna(method="pad").reindex(factor_calculation_dates)
    return get_industries_one_hot_encoding(raw_citic_industries)


def get_barra_style_exposure_on_factor_calculation_dates(universe, factor_calculation_dates, style_factors_list=None):
    barra_style_exposure = {}
    if style_factors_list is None:
        style_factors_list = BARRA_FACTOR_NAMES
    for f in style_factors_list:
        barra_style_exposure[f] = universe.align_df_with_trade_dates(
            universe.get_transform("barra", "am_marketdata_stock_barra_rsk", f).
                fillna(method="pad")).reindex(factor_calculation_dates).values
    return barra_style_exposure


def get_barra_specific_risk_on_factor_calculation_dates(universe, factor_calculation_dates):
    barra_specific_risk = get_specific_risk_df()
    barra_specific_risk = barra_specific_risk.reindex(universe.get_trade_dates()).fillna(method="pad")
    barra_specific_risk = barra_specific_risk.T.reindex(universe.get_symbols().columns).T.fillna(method="pad") #Fillna??
    return barra_specific_risk.reindex(factor_calculation_dates).values


def get_one_day_industries(latest_unique_industries, barra_industries, date_i):
    one_day_industries = []
    for industry_code in latest_unique_industries:
        ind_here = barra_industries[industry_code][date_i]
        if len(one_day_industries) > 0:
            one_day_industries = np.vstack((one_day_industries, ind_here))
        else:
            one_day_industries = ind_here
    return one_day_industries


def get_one_day_barra_style_exposure(barra_style_exposure, date_i, style_factors_list=None):
    one_day_barra_style_exposure = []
    if style_factors_list is None:
        style_factors_list = BARRA_FACTOR_NAMES
    for f in style_factors_list:
        barra_expo_here = barra_style_exposure[f][date_i]
        if len(one_day_barra_style_exposure) > 0:
            one_day_barra_style_exposure = np.vstack((one_day_barra_style_exposure, barra_expo_here))
        else:
            one_day_barra_style_exposure = barra_expo_here
    return one_day_barra_style_exposure

