import logging
import random
import time
import os
import xgboost
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.tree import export_graphviz
from sklearn.metrics import log_loss
from sklearn import preprocessing
from xgboost import plot_importance
from sklearn.base import TransformerMixin
from chameqs_common.config.trading_strategy_config import TradingStrategyConfig
from chameqs_common.utils.decorator import ignore_warning
from chameqs_common.data.universe import Universe
from chameqs_common.factor.combined_factor.factor_definition.combined_factor import CombinedFactor
from chameqs_common.factor.single_factor.processor.filler import IndustryAndMarketMeanFiller
from chameqs_common.factor.single_factor.store.single_factor_source import SingleFactorSource
from chameqs_common.utils.calculator import calc_rowwise_nan_correlation, calc_rowwise_nan_rank_correlation
from chameqs_common.ml.prepare.return_.lagged_index_excess_return_querier import LaggedIndexExcessReturnQuerier
from chameqs_common.backtest.common.mode import BacktestMode
from chameqs_red_local.factor.combined_factor.models.lgbm_model.funcs import *
from chameqs_red_local.factor.combined_factor.expect_return.common_oper_for_combinor import CombinorCommonOper
from chameqs_red_local.factor.combined_factor.expect_return.event_driven_signal_for_combinor import EventDrivenSignal


class ExpectReturnGenerator(CombinedFactor):
    def __init__(self,
                 name: str, universe: Universe, trading_strategy_config: TradingStrategyConfig, price_type: str,
                 factor_sources: list[SingleFactorSource], icir_rolling_window: int,
                 icir_min_rolling_window: int) -> None:
        super().__init__(
            name, universe, trading_strategy_config,
            {"Price Type": price_type, "ICIR Rolling Window": icir_rolling_window,
             "ICIR Min Rolling Window": icir_min_rolling_window})
        self.common_oper = CombinorCommonOper(universe, trading_strategy_config)
        self.event_driven = EventDrivenSignal(universe)
        self.price_df = self.universe.wind_common.get_price_df_by_type(price_type)
        self.factor_sources = factor_sources
        self.barra_style_factors = np.array(["beta", "btop", "earnyild", "growth", "leverage",
                                             "liquidty", "momentum", "resvol", "sizefactor","sizenl"])
        self.alpha_factors = None
        self.categorical_features = []
        self.factors_aggregate_config = self.read_factors_aggregate_config()

        self.icir_rolling_window = icir_rolling_window
        self.icir_min_rolling_window = icir_min_rolling_window
        self.trading_frequency = 5
        self.years = np.unique(self.factor_calculation_dates // 10000)
        self.training_year_length = 2
        self.predicting_year_length = 1

    def read_factors_aggregate_config(self):
        path = "D:\Data"
        conf = pd.read_excel(os.path.join(path, "factor_categories.xlsx"), sheet_name="part_202206", index_col=0)
        return conf

    def get_df(self, last_date=None) -> pd.DataFrame:

        top_category_factors: dict[str, pd.DataFrame] = {}
        for factor_source in self.factor_sources:
            self.__get_top_category_factor_value_on_factor_calcualtion_dates(top_category_factors, factor_source)
        common_factors = self.common_oper.load_barra_common_factors(self.barra_style_factors)
        for each_f, each_fv in common_factors.items():
            top_category_factors[each_f] = each_fv
        self.__transform_top_category_factors_value(top_category_factors)

        # top_category_factors: dict[str, pd.DataFrame] = {}
        # layer2_top_category_factors: dict[str, pd.DataFrame] = {}
        # single_factors_by_category = {}
        # for factor_source in self.factor_sources:
        #     self.common_oper.load_all_single_factors(single_factors_by_category, factor_source)
        #     # self.common_oper.load_layer1_combined_factors_by_icir(top_category_factors, factor_source)

        combined_factor = self.combine_top_category_factors_as_expected_return(top_category_factors) #single_factors_by_category, top_category_factors
        # combined_factor = self.__combine_top_category_factors_as_expected_return(layer2_top_category_factors)
        return combined_factor

    @ignore_warning
    def combine_top_category_factors_as_expected_return(self, top_category_factors: dict[str, pd.DataFrame]):
        predict_classes = ['worst', 'middle', 'best']
        tradable = self.universe.get_tradable_symbols()
        return_labels = self.get_return_labels(tradable)
        self.generate_features(top_category_factors)
        predicts = None

        for i in self.years[self.training_year_length:]:
            print(f"Start training year {i}")
            train_dates = self.common_oper.get_trading_days_in_range(int(f"{i-self.training_year_length}0101"), int(f"{i-1}1231"))
            test_dates = self.common_oper.get_trading_days_in_range(int(f"{i}0101"), int(f"{i+self.predicting_year_length-1}1231"))

            x_train, y_train = self.get_train_dataset(top_category_factors, train_dates, return_labels)
            test_dataset = self.get_test_dataset(top_category_factors, test_dates, tradable)

            model, predict = self.train_xgboost_models(x_train, y_train, test_dataset)
            # model, predict = self.train_lgbm_models(x_train, y_train, test_dataset)
            if predicts is None:
                predicts = pd.DataFrame(predict, index=test_dataset.index, columns=predict_classes)
            else:
                predicts = pd.concat([predicts, pd.DataFrame(predict, index=test_dataset.index, columns=predicts.columns)])

        expect_return = (predicts["best"] - predicts["worst"]).unstack().T
        expect_return = self.universe.align_df_with_symbols(expect_return)
        return expect_return.reindex(self.factor_calculation_dates)

    def train_xgboost_models(self, x_train, y_train, test_dataset):
        start_time = time.time()
        x_train_, x_validation, y_train_, y_validation = train_test_split(x_train, y_train, test_size=0.25)
        dtrain = xgboost.DMatrix(x_train_, label=y_train_, missing=np.NaN, enable_categorical=True)
        dvalidation = xgboost.DMatrix(x_validation, label=y_validation, missing=np.NaN, enable_categorical=True)
        param = {'max_depth': 4, 'eta': 0.01, 'objective': 'multi:softprob', 'num_class': 3, 'colsample_bylevel': 0.8}
        watchlist = [(dtrain, 'train'), (dvalidation, 'eval')]
        model = xgboost.train(param, dtrain, num_boost_round=1000, early_stopping_rounds=10, evals=watchlist,
                              verbose_eval=30)
        end_time = time.time()
        print("training time:", end_time - start_time)
        logging.info((f"Start predicting"))
        predict = model.predict(xgboost.DMatrix(test_dataset, enable_categorical=True),
                                iteration_range=(0, model.best_iteration + 1))
        return model, predict

    def train_lgbm_models(self, X_train, y_train, test_dataset):
        start_time = time.time()
        # x_train_, x_validation, y_train_, y_validation = train_test_split(x_train, y_train, test_size=0.25)
        clf = lgb.LGBMClassifier()
        clf.fit(X_train, y_train)

        end_time = time.time()
        print("training time:", end_time - start_time)
        logging.info((f"Start predicting"))
        predict = clf.predict_proba(test_dataset)
        return clf, predict

    def get_return_labels(self, tradable):
        # return_df = self.common_oper.get_stock_debarra_excess_return(lag=self.trading_frequency, mode="industry").replace(0, np.nan)
        return_df = self.common_oper.get_stock_return(lag=self.trading_frequency, price_type="adjusted_vwap")
        return_df = return_df.mask(~tradable)
        return_label = pd.DataFrame(np.nan, index=return_df.index, columns=return_df.columns)
        return_df_val = return_df.values
        return_label_val = return_label.values

        for row in range(len(return_df_val)):
            if (~np.isnan(return_df_val[row, :])).sum() != 0:
                return_label_val[row, :] = pd.qcut(return_df_val[row, :], 11, labels=False)
        return_label = return_label.mask(~((return_label == 0)|(return_label == 10)|(return_label == 5)))
        return_label[return_label == 5] = 1
        return_label[return_label == 10] = 2

        # for row in range(len(return_df_val)):
        #     if (~np.isnan(return_df_val[row, :])).sum() != 0:
        #         return_label_val[row, :] = pd.qcut(return_df_val[row, :], 5, labels=False)
        # return_label = return_label.mask(~((return_label == 0)|(return_label == 2)|(return_label == 4)))
        # return_label[return_label == 2] = 1
        # return_label[return_label == 4] = 2

        return return_label

    def generate_features(self, top_category_factors: dict[str, pd.DataFrame], categorical_part=0.2):
        industry_symbols = self.universe.get_citics_highest_level_industry_symbols().fillna(method="ffill")
        common_factors = self.common_oper.load_barra_common_factors(self.barra_style_factors)
        # for each_f in list(top_category_factors.keys()):
        #     top_category_factors[f"{each_f}_sqr"] = self.common_oper._transform_factor_value(top_category_factors[each_f]**2, preprocessing.QuantileTransformer(n_quantiles=5000, output_distribution="normal"))

        for each_f, each_fv in common_factors.items():
            top_category_factors[each_f] = each_fv
        self.common_oper._transform_top_category_factors_value(top_category_factors)
        self.alpha_factors = list(top_category_factors.keys())

        top_category_factors["industry"] = industry_symbols
        self.categorical_features += ["industry"]
        # for k in random.sample(self.alpha_factors, int(len(self.alpha_factors)*categorical_part)):
        #     top_category_factors[f"categorical_{k}"] = trans_to_categorical_features(top_category_factors[k])
        #     self.categorical_features += [f"categorical_{k}"]

        # # Add event driven signal
        # positive_selected_symbols, negtive_selected_symbols = self.event_driven.get_event_driven_signal_filtered_by_jump_score(hold_days=10)
        # top_category_factors["pos_announcements"] = positive_selected_symbols.fillna(0).astype(int)
        # top_category_factors["neg_announcements"] = negtive_selected_symbols.fillna(0).astype(int)
        # self.categorical_features += ["pos_announcements", "neg_announcements"]

    def get_train_dataset(self, top_category_factors: dict[str, pd.DataFrame], dates: np.ndarray, return_label):
        train_set = pd.DataFrame()
        for day in dates:
            factors = []
            for top_category in self.alpha_factors+self.categorical_features:
                factors.append(top_category_factors[top_category].loc[day])
            factors = pd.DataFrame(np.array(factors), index=self.alpha_factors+self.categorical_features, columns=self.price_df.columns).T
            factors.loc[:, 'returns_label'] = return_label.loc[day]
            train = factors[~factors['returns_label'].isna()]
            train_set = shuffle(pd.concat([train_set, train], ignore_index=True))
        x_train = train_set.drop(columns=['returns_label'])
        x_train[self.alpha_factors] = x_train[self.alpha_factors].astype(float)
        for cate in self.categorical_features:
            x_train[cate] = x_train[cate].astype("category")
        y_train = train_set['returns_label']
        return x_train, y_train

    @ignore_warning
    def get_test_dataset(self, top_category_factors: dict[str, pd.DataFrame], dates: np.ndarray, tradable: pd.DataFrame):
        test_set = pd.DataFrame({each_category: each_category_value.loc[dates, :].unstack() for each_category, each_category_value in top_category_factors.items()})
        tradable_these_days = tradable.loc[dates, :].unstack()
        test_set = test_set[tradable_these_days.values]
        for cate in self.categorical_features:
            test_set[cate] = test_set[cate].astype("category").values
        return test_set

    def __get_top_category_factor_value_on_factor_calcualtion_dates(self,
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
                top_category_factors[top_category] = pd.DataFrame(0, index=self.factor_calculation_dates,
                                                                  columns=self.symbols)
            top_category_factor_df = top_category_factors[top_category]
            factor_value_on_calculation_dates = factor_source.factor_value_store.get_factor_intermediate_status(
                factor_namespace, factor_name, factor_type, IndustryAndMarketMeanFiller,
                dates=self.factor_calculation_dates.tolist(),
                kwargs={"target_symbols": self.symbols,
                        "all_symbols_sorted_by_list_date": self.universe.wind_common.all_symbols_sorted_by_list_date})
            rolling_icir_matched_calculation_dates = self._get_rolling_icir_matched_factor_calculation_dates(
                factor_source, factor_namespace, factor_name, factor_type)
            weighted_factor_value = factor_value_on_calculation_dates.multiply(rolling_icir_matched_calculation_dates,
                                                                               axis=0)
            top_category_factor_df[(np.isnan(top_category_factor_df) & ~np.isnan(weighted_factor_value))] = 0
            top_category_factor_df += weighted_factor_value
            self._log_composite_factor(factor_namespace, factor_name, factor_type, factor_source.backtest_namespace)
            logging.info(
                f"Done combine [factor names space: {factor_namespace}, factor name: {factor_name}, factor type: {factor_type}] into [top category: {top_category}]")

    def _get_rolling_icir_matched_factor_calculation_dates(self,
            factor_source: SingleFactorSource, factor_namespace: str, factor_name: str, factor_type: str) -> pd.Series:
        factor_ic = factor_source.factor_ic_store.get_factor_rank_ic(factor_namespace, factor_name, factor_type, factor_source.backtest_namespace, BacktestMode.TOTAL)
        factor_ic = self.universe.align_df_with_trade_dates(factor_ic)
        factor_ic_on_trading_execution_dates = factor_ic.loc[self.trading_execution_dates]
        factor_rolling_icir: pd.Series = factor_ic_on_trading_execution_dates.rolling(self.icir_rolling_window, min_periods=self.icir_min_rolling_window).mean()/ \
            factor_ic_on_trading_execution_dates.rolling(self.icir_rolling_window, min_periods=self.icir_min_rolling_window).std()
        factor_rolling_icir_shift_to_match_factor_calculation_dates = pd.Series(index=self.factor_calculation_dates, data=np.nan)
        if self.is_last_date_factor_calculation_date:
            factor_rolling_icir_shift_to_match_factor_calculation_dates.iloc[1:] = factor_rolling_icir
        else:
            factor_rolling_icir_shift_to_match_factor_calculation_dates.iloc[:] = factor_rolling_icir.shift(periods=1)
        return factor_rolling_icir_shift_to_match_factor_calculation_dates

    def __transform_top_category_factors_value(self, top_category_factors: dict[str, pd.DataFrame]):
        for top_category in top_category_factors:
            logging.info(f"Start transform [top category: {top_category}] factor value")
            top_category_factors[top_category] = self.__transform_factor_value(top_category_factors[top_category],
                                                                               preprocessing.RobustScaler())
            logging.info(f"Done transform [top category: {top_category}] factor value")

    def __transform_factor_value(self, factor_df: pd.DataFrame, transformer: TransformerMixin) -> pd.DataFrame:
        result_df = pd.DataFrame(np.nan, index=factor_df.index, columns=factor_df.columns)
        for date, factor_value in factor_df.iterrows():
            not_nan_indexes = np.where(~np.isnan(factor_value))[0]
            if not_nan_indexes.size > 0:
                result_df.loc[date].iloc[not_nan_indexes] = transformer.fit_transform(
                    factor_value.iloc[not_nan_indexes].values.reshape(not_nan_indexes.size, 1)).reshape(
                    not_nan_indexes.size)
        return result_df
