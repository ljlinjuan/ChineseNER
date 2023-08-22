import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from chameqs_common.utils.decorator import ignore_warning
from chameqs_common.config.trading_strategy_config import TradingStrategyConfig
from chameqs_common.data.universe import Universe, SharedUniverse
from chameqs_common.data.barra import BARRA_FACTOR_NAMES, BARRA_INDUSTRIES_MAP, Barra


class PureFactor():
    def __init__(self, universe: Universe, trading_strategy_config: TradingStrategyConfig):
        self.universe = universe
        self.trading_strategy_config = trading_strategy_config
        self.symbols = self.universe.get_symbols().columns.values
        self.factor_calculation_dates = trading_strategy_config.factor_calculation_dates
        self.price_df = self.universe.wind_common.get_price_df_by_type("adjusted_close")
        self.factor_dates_price = self.price_df.loc[self.factor_calculation_dates]
        self.factor_dates_return = self.factor_dates_price.diff(axis=0) / self.factor_dates_price.shift(axis=0, periods=1)
        self.tradable_symbols_on_factor_calculation_dates = self.universe.get_tradable_symbols().loc[self.factor_calculation_dates]
        self.barra_factor_names = ["beta", "btop", "earnyild", "growth", "leverage", "liquidty",
                                   "momentum", "resvol", "sizefactor", "sizenl"]

    def get_barra_industries_on_factor_calculation_dates(self):
        raw_barra_industries = self.universe.get_transform("barra", "am_marketdata_stock_barra_rsk", "ind").fillna(method="pad").reindex(self.factor_calculation_dates)
        raw_latest_barra_industries = raw_barra_industries.values[-1, :]
        latest_unique_barra_industries = np.unique(raw_latest_barra_industries[~np.isnan(raw_latest_barra_industries)])
        barra_industries = {}
        for barra_industry_code in latest_unique_barra_industries:
            barra_industries[barra_industry_code] = (raw_barra_industries == barra_industry_code).astype(int).values
        return latest_unique_barra_industries, barra_industries

    def get_barra_style_exposure_on_factor_calculation_dates(self):
        barra_style_exposure = {}
        for f in self.barra_factor_names:
            barra_style_exposure[f] = self.universe.align_df_with_trade_dates(
                self.universe.get_transform("barra", "am_marketdata_stock_barra_rsk", f).
                    fillna(method="pad")).reindex(self.factor_calculation_dates).values
        return barra_style_exposure

    def get_one_day_barra_industries(self, latest_unique_barra_industries, barra_industries, date_i):
        one_day_barra_industries = []
        for barra_industry_code in latest_unique_barra_industries:
            barra_ind_here = barra_industries[barra_industry_code][date_i]
            if len(one_day_barra_industries) > 0:
                one_day_barra_industries = np.vstack((one_day_barra_industries, barra_ind_here))
            else:
                one_day_barra_industries = barra_ind_here
        return one_day_barra_industries

    def get_one_day_barra_style_exposure(self, barra_style_exposure, date_i):
        one_day_barra_style_exposure = []
        for f in self.barra_factor_names:
            barra_expo_here = barra_style_exposure[f][date_i]
            if len(one_day_barra_style_exposure) > 0:
                one_day_barra_style_exposure = np.vstack((one_day_barra_style_exposure, barra_expo_here))
            else:
                one_day_barra_style_exposure = barra_expo_here
        return one_day_barra_style_exposure

    def get_one_day_factor_value(self, factor_values, date_i):
        one_day_factor_value = []
        for f in factor_values:
            factor_value_here = factor_values[f][date_i]
            if len(one_day_factor_value) > 0:
                one_day_factor_value = np.vstack((one_day_factor_value, factor_value_here))
            else:
                one_day_factor_value = factor_value_here
        return one_day_factor_value

    def construct_pure_factor(self, factor:pd.DataFrame, factor_name:str):

        latest_unique_barra_industries, barra_industries = self.get_barra_industries_on_factor_calculation_dates()
        barra_style_exposure = self.get_barra_style_exposure_on_factor_calculation_dates()
        barra_fields = list(map(lambda x:BARRA_INDUSTRIES_MAP[x], latest_unique_barra_industries)) + self.barra_factor_names
        mkv = self.universe.wind_common.get_free_circulation_market_value_na_filled().reindex(
            self.factor_calculation_dates)
        factor = factor.reindex(self.factor_calculation_dates)
        stock_return = self.factor_dates_return.shift(-1)

        weights = pd.DataFrame(index=self.factor_calculation_dates, columns=factor.columns, data=0.0)
        pure_factor_return = pd.DataFrame(index=self.factor_calculation_dates, columns=["country"]+[factor_name]+barra_fields)

        for date_i, date in enumerate(self.factor_calculation_dates):
            if date_i < 30:
                continue
            # print(date_i)
            one_day_barra_industries = self.get_one_day_barra_industries(latest_unique_barra_industries, barra_industries, date_i)
            one_day_barra_style_exposure = self.get_one_day_barra_style_exposure(barra_style_exposure, date_i)

            one_day_result = self.get_one_day_pure_factor_port_weight(factor.loc[date].values.reshape(1, factor.shape[1]),
                                                                      one_day_barra_industries,
                                                                      one_day_barra_style_exposure,
                                                                      self.tradable_symbols_on_factor_calculation_dates.loc[date].values,
                                                                      mkv.loc[date].values,
                                                                      stock_return.loc[date].values,
                                                                      self.symbols.size
                                                                      )
            # weights.loc[date] = one_day_result[0]
            pure_factor_return.loc[date] = one_day_result[1]
        return pure_factor_return

    def construct_list_of_pure_factors(self, factors:dict[pd.DataFrame], factor_names:list[str]):
        for k, v in factors.items():
            df = v.reindex(self.factor_calculation_dates).fillna(method="pad")
            df = df.T.reindex(self.universe.get_symbols().columns).T
            factors[k] = df.values
        # factors = {k:v.reindex(self.factor_calculation_dates).fillna(method="pad").values for k,v in factors.items()}

        latest_unique_barra_industries, barra_industries = self.get_barra_industries_on_factor_calculation_dates()
        barra_style_exposure = self.get_barra_style_exposure_on_factor_calculation_dates()
        barra_fields = list(map(lambda x:BARRA_INDUSTRIES_MAP[x], latest_unique_barra_industries)) + self.barra_factor_names
        mkv = self.universe.wind_common.get_free_circulation_market_value_na_filled().reindex(
            self.factor_calculation_dates)
        stock_return = self.factor_dates_return.shift(-1)

        # weights = pd.DataFrame(index=self.factor_calculation_dates, columns=self.universe.get_symbols().columns, data=0.0)
        pure_factor_return = pd.DataFrame(index=self.factor_calculation_dates, columns=["country"]+factor_names+barra_fields)

        for date_i, date in enumerate(self.factor_calculation_dates):
            if date_i < 30:
                continue
            # print(date_i)
            one_day_barra_industries = self.get_one_day_barra_industries(latest_unique_barra_industries, barra_industries, date_i)
            one_day_barra_style_exposure = self.get_one_day_barra_style_exposure(barra_style_exposure, date_i)
            one_day_factor_values = self.get_one_day_factor_value(factors, date_i)
            factor_values_not_allna_mask = (~np.isnan(one_day_factor_values)).sum(1) != 0
            one_day_factor_values = one_day_factor_values[factor_values_not_allna_mask]
            if len(one_day_factor_values) == 0:
                continue
            one_day_result = self.get_one_day_pure_factor_port_weight(one_day_factor_values,
                                                                      one_day_barra_industries,
                                                                      one_day_barra_style_exposure,
                                                                      self.tradable_symbols_on_factor_calculation_dates.loc[date].values,
                                                                      mkv.loc[date].values,
                                                                      stock_return.loc[date].values,
                                                                      self.symbols.size
                                                                      )
            # weights.loc[date] = one_day_result[0]
            pure_factor_return.loc[date][[True] + list(factor_values_not_allna_mask)+[True]*(len(one_day_barra_industries)+len(one_day_barra_style_exposure))] = one_day_result[1]
        return pure_factor_return

    def get_one_day_pure_factor_port_weight(self, factor, barra_ind, barra_style_expo, one_day_tradable_symbols,
                                            mkv, stock_return, symbol_size):
        na_cond = ~np.isnan(factor.sum(axis=0))
        na_cond = np.logical_and(na_cond, ~np.isnan(barra_style_expo.sum(axis=0)))
        use_symbols = np.logical_and(na_cond, np.logical_or(one_day_tradable_symbols,  na_cond))
        use_index = np.where(use_symbols)[0]

        # Calculate pure factor port weights
        r = stock_return[use_index]
        X = np.hstack((factor.T[use_index], barra_ind.T[use_index], barra_style_expo.T[use_index]))
        V = np.diag(mkv[use_index])
        R = np.diag(np.ones(X.shape[1]))
        R = np.vstack((np.zeros(X.shape[1]), R))
        R[0, :(barra_ind.shape[1])] = 1 # dummy variables
        X = np.vstack((np.ones(X.shape[0]), X.T)).T

        XR = X.dot(R)
        omega = R.dot(np.linalg.pinv(XR.T.dot(V).dot(XR))).dot(XR.T).dot(V) # pure factor portfolio weight
        f = omega.dot(r) # pure factor return

        one_day_weight = np.full((X.shape[1], symbol_size), 0, dtype=float)
        one_day_weight[:, use_index] = omega

        # Debarra port weight
        debarra_short_pure_factor_weight = np.full((symbol_size), 0, dtype=float)
        debarra_equal_weight = np.full((X.shape[0]), 1/X.shape[0], dtype=float)
        debarra_short_pure_factor_weight[use_index] = (debarra_equal_weight @ X) @ omega
        return debarra_short_pure_factor_weight, f

    @ignore_warning
    def quantile_transform_factor_value(self, factor_df: pd.DataFrame,) -> pd.DataFrame:
        result_df = pd.DataFrame(np.nan, index=factor_df.index, columns=factor_df.columns)
        for date, factor_value in factor_df.iterrows():
            not_nan_indexes = np.where(~np.isnan(factor_value))[0]
            if not_nan_indexes.size > 0:
                result_df.loc[date].iloc[not_nan_indexes] = preprocessing.QuantileTransformer(n_quantiles=5000, output_distribution="normal").fit_transform(
                    factor_value.iloc[not_nan_indexes].values.reshape(not_nan_indexes.size, 1)).reshape(
                    not_nan_indexes.size)
        return result_df


def run():
    universe_name = "all_2010"
    mongo_host = "localhost"
    mongo_port = 27018
    universe = SharedUniverse.load(mongo_host, mongo_port, universe_name)
    trading_config = TradingStrategyConfig(universe, "20d")
    pf = PureFactor(universe, trading_config)

    path = r"D:\Data\CombinedFactor\by_realized_ic" #\by_factor_date_ic
    factor_values = {}
    # factor_name_list = ["MoneyFlowInflow", "MoneyFlowVolumePercentage", "MoneyFlowLSDifference"]
    # factor_name_list = ["VolPhysicalMode0", "VolPhysicalMode1", "VolPhysicalMode3", "Turnover"] #, "Turnover"
    # factor_name_list = ["MomentumPhysical", "MomentumDirectional", "Reversal", "ReversalCandel", "ReversalSlope", "ReversalReturn",] # "Reversal", "ReversalCandel", "ReversalSlope", "ReversalReturn",
    # factor_name_list = ["MomentumPhysical", "MomentumDirectional"]
    # factor_name_list = ["Turnover", "TurnoverBias", "TurnoverSkew", "TurnoverSlope", "VolPhysicalMode3", "VolPhysicalMode0"]
    # factor_name_list = ["NetProfitDerivativeIndicatorsSUE", "NetProfitDerivativeIndicatorsYOY"] # , "NetProfitDerivativeIndicatorsSUE", "NetProfitDerivativeIndicatorsValue"
    factor_name_list = ["DvdPayoutRatioLP", "DvdPayoutRatioValue"]
    # factor_name_list = ["IntradayPriceFactorkurtRet", "IntradayPriceFactorkurtVolume", "IntradayPriceFactorskewRet",
    #                     "IntradayPriceFactorskewVolume", "IntradayPriceFactorstdRet", "IntradayPriceFactorstdVolume"
    #                     ]
    # factor_name_list = ["MoneyFlowLSDifference", "IntradayPriceFactorstdRet", "ReversalReturn", "MutualFundHoldingsDeviation", "VolPhysicalMode0",
    #                     "NetProfitDerivativeIndicatorsLP", "NetProfitDerivativeIndicatorsSUE"]

    # factor_name = "MutualFundHoldingsCount"
    # factor = pd.read_parquet(os.path.join(path, factor_name))
    # pf.construct_pure_factor(factor, factor_name)
    # factor_name_list = ["MutualFundHoldings", "MutualFundHoldingsCount", "MutualFundHoldingsDeviation", "MutualFundHoldingsMkv", "MutualFundHoldingsSlope"]
    # factor_name_list = ["MoneyFlowInflow", "MoneyFlowVolumePercentage", "MoneyFlowLSDifference"]
    # factor_name_list  += [f"nl_{i}" for i in factor_name_list]
    for f in factor_name_list:
        df = pd.read_parquet(os.path.join(path, f))
        df.columns = df.columns.astype(int)
        if f[:2] == "nl":
            df = np.log(df.abs())
        # df = (df-df.mean(1))/df.std(1)
        factor_values[f] = df #.fillna(0) # pf.quantile_transform_factor_value(df)

    # factor_df = pd.read_parquet(os.path.join(path, factor_name))
    pf.construct_list_of_pure_factors(factor_values, factor_name_list)


if __name__ == "__main__":
    run()
