import numpy as np
import pandas as pd
from chameqs_common.utils.decorator import ignore_warning
from sklearn import preprocessing
from sklearn.base import TransformerMixin

def linear_neutralize_only_one_explain_variable(y: np.ndarray, x: np.ndarray):
    # Default rowwise
    x[np.isnan(y)] = np.nan
    y[np.isnan(x)] = np.nan

    resid = np.full((x.shape[0], x.shape[1]), np.nan)
    for row_i in range(x.shape[0]):
        x_here = x[row_i, :]
        y_here = y[row_i, :]
        symbol_indexes = np.where(~np.isnan(x_here))
        x_here = x_here[symbol_indexes]
        y_here = y_here[symbol_indexes]

        if len(x_here) < 100:
            continue

        x_here = np.vstack([np.ones(len(x_here)), x_here]).T
        result = linear_neutralizer_do_calculation(x_here, y_here)
        resid[row_i, symbol_indexes] = result[1]
    return resid


def linear_neutralizer_do_calculation(parameters, factor_value, weight=None):
    if weight is not None:
        weight_sqrt = np.sqrt(weight)
        parameters *= weight_sqrt[:, np.newaxis]
        factor_value *= weight_sqrt
    betas = np.linalg.lstsq(parameters, factor_value, rcond=None)[0]
    residuals = factor_value - parameters.dot(betas)
    return betas, residuals


@ignore_warning
def quantile_transform_factor_value(factor_df: pd.DataFrame, ) -> pd.DataFrame:
    result_df = pd.DataFrame(np.nan, index=factor_df.index, columns=factor_df.columns)
    for date, factor_value in factor_df.iterrows():
        not_nan_indexes = np.where(~np.isnan(factor_value))[0]
        if not_nan_indexes.size > 0:
            result_df.loc[date].iloc[not_nan_indexes] = preprocessing.QuantileTransformer(n_quantiles=5000,
                                                                                          output_distribution="normal").fit_transform(
                factor_value.iloc[not_nan_indexes].values.reshape(not_nan_indexes.size, 1)).reshape(
                not_nan_indexes.size)
    return result_df


def transform_factor_value(factor_df: pd.DataFrame, transformer: TransformerMixin) -> pd.DataFrame:
    result_df = pd.DataFrame(np.nan, index=factor_df.index, columns=factor_df.columns)
    for date, factor_value in factor_df.iterrows():
        not_nan_indexes = np.where(~np.isnan(factor_value))[0]
        if not_nan_indexes.size > 0:
            result_df.loc[date].iloc[not_nan_indexes] = transformer.fit_transform(
                factor_value.iloc[not_nan_indexes].values.reshape(not_nan_indexes.size, 1)).reshape(
                not_nan_indexes.size)
    return result_df


# =========== func ===========
import numpy as np
import pandas as pd

# -------- objective functions
def rmspe(y_true, y_pred):
    return np.sqrt(np.sum((y_true - y_pred )**2))


def corrcoef(y_true, y_pred):
    return -np.abs(np.sum((y_true -np.mean(y_true)) *(y_pred -np.mean(y_pred)))/ \
           np.sqrt(np.sum((y_true -np.mean(y_true))**2) *np.sum((y_pred -np.mean(y_pred))**2)))


def feval_rmspe(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    return 'RMSPE', rmspe(y_true, y_pred), False


def feval_rcorr(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    return 'RCorr', corrcoef(y_true, y_pred), False


# -------- Copula
# https://en.wikipedia.org/wiki/Copula_(probability_theory)
def amk_trans(f, theta):
    return np.log((1-theta*(1-f))/f)


def amk_trans_inverse(f, theta):
    return (1 - theta)/(np.exp(f)-theta)


def trans_to_categorical_features(factor, nbins=5):
    not_all_na_idx = (~factor.isnull()).sum(1) > 0
    factor = factor[not_all_na_idx]
    categorical_factor = factor.apply(lambda x: pd.qcut(x.rank(method='first'), nbins, labels=[int(i)+1 for i in range(nbins)]), axis=1)
    categorical_factor = categorical_factor.reindex(not_all_na_idx.index)
    return pd.DataFrame(np.where(categorical_factor.isnull(), 0, categorical_factor), index=categorical_factor.index, columns=categorical_factor.columns)






