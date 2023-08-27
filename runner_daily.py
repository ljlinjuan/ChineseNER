import pandas as pd
import numpy as np
import time
import logging

import scipy.linalg
from sklearn import preprocessing
import multiprocessing as mp
from chameqs_common.data.universe import Universe
from chameqs_common.config.trading_strategy_config import TradingStrategyConfig
from chameqs_common.data.barra import BARRA_FACTOR_NAMES, BARRA_INDUSTRIES_MAP, Barra
from chameqs_common.portfolio.single_portfolio.stocks.optimizer.optimizer import StocksPortfolioOptimizer
from chameqs_red_local.optimizer.datafeed import *
from chameqs_red_local.optimizer.mosek.vanilla_max_expect_return import mosek_max_expect_return


class Optimizer(StocksPortfolioOptimizer):
    def __init__(self,
                 combined_factor_namespace: str,
                 combined_factor_name: str,
                 portfolio_name: str,
                 universe: Universe,
                 trading_strategy_config: TradingStrategyConfig,
                 bench_index_name_or_weight_path: str,
                 annualized_tracking_error_in_percent: float = 1,
                 max_allocation_ratio: float = 0.03,
                 industry_bound: float = 0.02,
                 style_exposure_bound: float = 0.1,
                 weight_in_index_budget: float = 0,
                 weight_deviation: float = 0.02,
                 max_portfolio_turnover: float = 0.2,
                 parallel_processes: int = 6
                 ) -> None:
        super().__init__(
            combined_factor_namespace, combined_factor_name, portfolio_name, universe, trading_strategy_config,
            description={
                "Bench Index": bench_index_name_or_weight_path,
                "Annualized Tracking Error": annualized_tracking_error_in_percent / 100,
                "Max Allocation Ratio": max_allocation_ratio,
                "Industry Bound": industry_bound,
                "Style Exposure Bound": style_exposure_bound,
                "Weight In Index Budget": weight_in_index_budget,
                "Weight Deviation": weight_deviation,
                "Max Portfolio Turnover": max_portfolio_turnover
            })

        self.parallel_processes = parallel_processes
        self.universe = universe
        self.factor_calculation_dates = self.universe.get_trade_dates() #trading_strategy_config.factor_calculation_dates
        self.symbols = self.universe.get_symbols().columns.values
        self.tradable_symbols_on_factor_calculation_dates = self.universe.get_tradable_symbols().loc[self.factor_calculation_dates]
        self.industry_name_map = get_citic_industry_name_map(self.universe)

        self.bench_index_name_or_weight_path = bench_index_name_or_weight_path
        self.annualized_tracking_error = (annualized_tracking_error_in_percent/100)
        self.max_allocation_ratio = max_allocation_ratio
        self.industry_bound = industry_bound
        self.style_exposure_bound = style_exposure_bound
        self.weight_in_index_budget = weight_in_index_budget
        self.weight_deviation = weight_deviation
        self.max_portfolio_turnover = max_portfolio_turnover

    def get_weights(self, expect_return):
        weights = pd.DataFrame(index=self.factor_calculation_dates, columns=self.symbols, data=0.0)

        # Prepare optimizer inputs
        bench_index_weights = get_csi1000_index_weights_on_factor_calculation_dates(self.universe, self.factor_calculation_dates)
        is_in_bench_index = bench_index_weights != 0

        latest_unique_barra_industries, barra_industries = get_barra_industries_on_factor_calculation_dates(self.universe, self.factor_calculation_dates)
        latest_unique_citic_industries, citic_industries = get_citic_industries_on_factor_calculation_dates(self.universe, self.factor_calculation_dates)
        style_factors_list = BARRA_FACTOR_NAMES.copy()
        barra_style_exposure = get_barra_style_exposure_on_factor_calculation_dates(self.universe, self.factor_calculation_dates, style_factors_list)
        barra_specific_risk = get_barra_specific_risk_on_factor_calculation_dates(self.universe, self.factor_calculation_dates)

        are_bench_index_weight_not_na_on_date = bench_index_weights.sum(1) != 0
        first_bench_index_weight_not_nan_date = np.where(are_bench_index_weight_not_na_on_date)[0][0]
        are_expected_return_not_na_on_date = (~expect_return.loc[self.factor_calculation_dates].isna()).any(axis=1)
        first_expected_return_not_nan_date = np.where(are_expected_return_not_na_on_date)[0][0]
        first_optimize_date = max(first_bench_index_weight_not_nan_date, first_expected_return_not_nan_date)

        # pool = mp.Pool(processes=self.parallel_processes)
        for date_i in range(first_optimize_date, self.factor_calculation_dates.size):
            date = self.factor_calculation_dates[date_i]
            if date < 20221001:
                continue
            one_day_barra_industries = get_one_day_industries(latest_unique_barra_industries, barra_industries, date_i)
            one_day_citic_industries = get_one_day_industries(latest_unique_citic_industries, citic_industries, date_i)
            one_day_barra_style_exposure = get_one_day_barra_style_exposure(barra_style_exposure, date_i, style_factors_list)
            barra_fields = list(map(lambda x: BARRA_INDUSTRIES_MAP[x], latest_unique_barra_industries)) + style_factors_list
            one_day_barra_covariance_matrix = Barra.get_first_backward_cov_matrix(date, barra_fields) / 10000
            one_day_barra_specific_risk = barra_specific_risk[date_i, :] / 10000

            one_day_weight_result = self.get_one_day_weight(expect_return.loc[date].values,
                                                            self.tradable_symbols_on_factor_calculation_dates.loc[date].values,
                                                            date_i,
                                                            self.factor_calculation_dates.size,
                                                            self.symbols.size,
                                                            one_day_barra_style_exposure,
                                                            one_day_barra_industries,
                                                            one_day_citic_industries,
                                                            one_day_barra_covariance_matrix,
                                                            one_day_barra_specific_risk,
                                                            is_in_bench_index[date_i],
                                                            bench_index_weights[date_i]
                                                            )
            if one_day_weight_result is not None:
                weights.loc[date] = one_day_weight_result[0]
        return weights

    def get_one_day_weight(self,
                           one_day_expected_return,
                           one_day_tradable_symbols,
                           date_i, total_date_size, symbol_size,
                           one_day_barra_style_exposure,
                           one_day_barra_industries, one_day_citic_industries,
                           one_day_barra_covariance_matrix,
                           one_day_barra_specific_risk,
                           one_day_is_in_bench_index, one_day_bench_index_weights
                           ):
        tic = time.time()
        logging.info(f'Period {date_i + 1}/{total_date_size} optimization start...')

        if (sum(~np.isnan(one_day_expected_return)) == 0) | (sum(one_day_expected_return) == 0):
            return None

        symbols = one_day_tradable_symbols
        na_cond = ~np.isnan(one_day_expected_return)

        na_cond = np.logical_and(na_cond, ~np.isnan(one_day_barra_style_exposure.sum(axis=0)))
        isin_idx_this_day = one_day_is_in_bench_index
        # use_symbols = np.logical_and(na_cond, np.logical_or(symbols, isin_idx_this_day))
        use_symbols = np.logical_or(isin_idx_this_day, np.logical_and(na_cond, symbols))
        use_index = np.where(use_symbols)[0]
        # Symbols in bench index but not in our universe.
        set0_symbols = np.logical_and(use_symbols, ~symbols)
        set0_index = np.where(set0_symbols)[0]

        used_single_expected_return = one_day_expected_return[use_index]
        used_one_day_barra_style_exposure = one_day_barra_style_exposure.T[use_index]
        used_one_day_barra_industries = one_day_barra_industries.T[use_index]
        used_one_day_citic_industries = one_day_citic_industries.T[use_index]
        used_one_day_barra_specific_risk = one_day_barra_specific_risk[use_index]
        used_one_day_bench_index_weights = one_day_bench_index_weights[use_index]
        used_one_day_isin_idx_this_day = isin_idx_this_day.T[use_index]
        used_single_expected_return[np.isnan(used_single_expected_return)] = 0
        used_one_day_barra_style_exposure[np.isnan(used_one_day_barra_style_exposure)] = 0
        used_one_day_barra_industries[np.isnan(used_one_day_barra_industries)] = 0
        used_one_day_barra_specific_risk[np.isnan(used_one_day_barra_specific_risk)] = 0
        set0_vector = np.zeros(symbol_size)
        set0_vector[set0_index] = 1
        set0_vector = set0_vector[use_index]

        # ---------- mosek optimization
        barra_this_day = np.vstack((used_one_day_barra_industries.T, used_one_day_barra_style_exposure.T)).T
        VT = scipy.linalg.cholesky(one_day_barra_covariance_matrix.values, lower=False)
        VT = VT.dot(barra_this_day.T)
        GT = np.vstack([VT, np.diag(np.sqrt(used_one_day_barra_specific_risk))])
        TE = self.annualized_tracking_error
        n = len(use_index)
        max_alloc = self.max_allocation_ratio
        ind_bound = self.industry_bound
        style_expo_bound = self.style_exposure_bound
        wgt_in_idx_budget = self.weight_in_index_budget
        wgt_deviation = self.weight_deviation

        # Some period can not reach optimal result under above constraints
        flag = True
        count = 0
        while flag:
            count += 1
            try:
                opt_w = mosek_max_expect_return(n,
                                                 used_single_expected_return,
                                                 GT,
                                                 self.annualized_tracking_error,
                                                 used_one_day_citic_industries.T, # Use citic industry here!!
                                                 ind_bound,
                                                 used_one_day_barra_style_exposure.T,
                                                 style_expo_bound,
                                                 used_one_day_bench_index_weights,
                                                 set0_vector,
                                                 self.weight_deviation,
                                                 used_one_day_isin_idx_this_day.astype(float),
                                                 self.weight_in_index_budget
                                                 )
                flag = False
            except:
                logging.info(f'Warning!!! {date_i + 1}/{total_date_size} can not reach optimal solution', exc_info=True)
                TE += 1
                max_alloc += 0.005
                ind_bound += 0.02
                wgt_in_idx_budget -= 0.05
                wgt_deviation += 0.005

        one_day_weight = np.full((symbol_size), 0, dtype=float)
        one_day_weight[use_index] = opt_w

        logging.info(f'{date_i + 1}/{total_date_size} done in {np.round(time.time() - tic, 2)}s')
        logging.info(f"Weights in benchmark: {one_day_weight[use_index].dot(used_one_day_isin_idx_this_day.astype(float))}")
        return one_day_weight, date_i
