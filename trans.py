import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import random

from sklearn.neighbors import KernelDensity
import xgboost
import statsmodels.api as sm

from sklearn.base import TransformerMixin
from chameqs_common.utils.decorator import ignore_warning


@ignore_warning
def transform_factor_value(factor_df: pd.DataFrame, transformer: TransformerMixin) -> pd.DataFrame:
    result_df = pd.DataFrame(np.nan, index=factor_df.index, columns=factor_df.columns)
    for date, factor_value in factor_df.iterrows():
        not_nan_indexes = np.where(~np.isnan(factor_value))[0]
        if not_nan_indexes.size > 0:
            result_df.loc[date].iloc[not_nan_indexes] = transformer.fit_transform(
                factor_value.iloc[not_nan_indexes].values.reshape(not_nan_indexes.size, 1)).reshape(
                not_nan_indexes.size)
    return result_df


def trans_to_target_density(current_df, target_historical_data, look_back_window: int):
    target_historical_data = ((target_historical_data.T - target_historical_data.mean(1))).T
    target_historical_data = target_historical_data.values
    current_values = current_df.values
    transformed_df = pd.DataFrame(np.nan, index=current_df.index, columns=current_df.columns)
    transformed_values = transformed_df.values

    for row in range(look_back_window, len(current_df)):
        current_data_this_day = current_values[row, :]
        current_data_this_day_notna_idx = ~np.isnan(current_data_this_day)
        if current_data_this_day_notna_idx.sum() == 0:
            continue
        historical_data_hist = pd.Series(target_historical_data[row-look_back_window:row].flatten()).dropna() # row-look_back_window:row
        dens = generate_density_from_historical_distribution(historical_data_hist)
        transformed_values[row, current_data_this_day_notna_idx] = from_density_to_values_inference(dens, current_data_this_day[current_data_this_day_notna_idx])
    return transformed_df


def generate_density_from_historical_distribution(historical_data):
    # , kernel="gaussian", bandwidth=0.01
    # historical_data_hist = pd.Series(historical_data.iloc[-look_back_window:].values.flatten()).dropna()
    # historical_data_hist = historical_data_hist.values.reshape((len(historical_data_hist), 1))
    # model = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    # model.fit(historical_data_hist)
    dens = sm.nonparametric.KDEUnivariate(historical_data)
    dens.fit()
    return dens


def from_density_to_values_inference(dens, values_need_to_trans):
    num_of_points = len(values_need_to_trans)
    values_infered = dens.icdf[[int((i/num_of_points)*len(dens.icdf)) for i in range(num_of_points)]] # can do winsorize here
    values_transformed = values_infered[np.argsort(np.argsort(values_need_to_trans))]
    return values_transformed
