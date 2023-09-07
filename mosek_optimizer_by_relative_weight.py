import logging
from re import M
from time import time

import numpy as np
import pandas as pd
import scipy.linalg
from chameqs_common.config.trading_strategy_config import TradingStrategyConfig
from chameqs_common.data.aindex import get_index_weights_by_name
from chameqs_common.data.barra import BARRA_FACTOR_NAMES, BARRA_INDUSTRIES_MAP, Barra
from chameqs_common.data.universe import Universe
from chameqs_common.portfolio.single_portfolio.stocks.optimizer.optimizer import StocksPortfolioOptimizer
from chameqs_common.utils.array_util import fill_empty_rows_with_previous_not_empty_row
from mosek.fusion import Domain, Expr, Model, ObjectiveSense, Var


class MosekOptimizer(StocksPortfolioOptimizer):
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
            transaction_cost: float = 0,
            weight_in_index_budget: float = 0.7,
            allocate_portfolio_ratio: float = 1.0,
            weight_deviation: float = 0.02,
            max_portfolio_turnover: float = 0.8) -> None:
        super().__init__(
            combined_factor_namespace, combined_factor_name, portfolio_name, universe, trading_strategy_config,
            description = {
                "Bench Index": bench_index_name_or_weight_path,
                "Annualized Tracking Error": annualized_tracking_error_in_percent / 100,
                "Max Allocation Ratio": max_allocation_ratio,
                "Industry Bound": industry_bound,
                "Style Exposure Bound": style_exposure_bound,
                "Transaction Cost": transaction_cost,
                "Weight In Index Budget": weight_in_index_budget,
                "Allocate Portfolio Ratio": allocate_portfolio_ratio,
                "Weight Deviation": weight_deviation,
                "Max Portfolio Turnover": max_portfolio_turnover
            })
        self.bench_index_name_or_weight_path = bench_index_name_or_weight_path
        self.barra_factor_names = BARRA_FACTOR_NAMES
        self.annualized_tracking_error = annualized_tracking_error_in_percent**2
        self.max_allocation_ratio = max_allocation_ratio
        self.industry_bound = industry_bound
        self.style_exposure_bound = style_exposure_bound
        self.transaction_cost = transaction_cost
        self.weight_in_index_budget = weight_in_index_budget
        self.allocate_portfolio_ratio = allocate_portfolio_ratio
        self.weight_deviation = weight_deviation
        self.max_portfolio_turnover = max_portfolio_turnover

    def get_weights(self, expected_return: pd.DataFrame) -> pd.DataFrame:
        bench_index_weights = self.get_index_weights_on_factor_calculation_dates(self.bench_index_name_or_weight_path)
        is_in_bench_index = bench_index_weights != 0

        weights = pd.DataFrame(index=self.factor_calculation_dates, columns=self.symbols, data=0.0)
        weights_hedged = pd.DataFrame(index=self.factor_calculation_dates, columns=self.symbols, data=0.0)

        latest_unique_barra_industries, barra_industries = self.get_barra_industries_on_factor_calculation_dates()
        barra_style_exposure = self.get_barra_style_exposure_on_factor_calculation_dates()
        barra_fields = list(
            map(lambda x: BARRA_INDUSTRIES_MAP[x], latest_unique_barra_industries)) + self.barra_factor_names

        previous_weights = np.zeros(self.symbols.size)
        count = 0

        are_bench_index_weight_not_na_on_date = (~bench_index_weights.isna()).any(axis=1)
        first_bench_index_weight_not_nan_date = are_bench_index_weight_not_na_on_date[are_bench_index_weight_not_na_on_date == True].index[0]
        bench_index_weights = bench_index_weights.fillna(0)

        are_expected_return_not_na_on_date = (~expected_return.loc[self.factor_calculation_dates].isna()).any(axis=1)
        first_expected_return_not_nan_date = are_expected_return_not_na_on_date[are_expected_return_not_na_on_date == True].index[0]
        
        first_optimize_date = max(first_bench_index_weight_not_nan_date, first_expected_return_not_nan_date)

        for date_i in range(0, self.factor_calculation_dates.size):
            date = self.factor_calculation_dates[date_i]
            if date < first_optimize_date:
                continue

            if count == 0:
                max_turnover = 1
            else:
                max_turnover = self.max_portfolio_turnover

            one_day_barra_industries = self.get_one_day_barra_industries(latest_unique_barra_industries, barra_industries, date_i)
            one_day_barra_style_exposure = self.get_one_day_barra_style_exposure(barra_style_exposure, date_i)
            one_day_barra_covariance_matrix = Barra.get_first_backward_cov_matrix(date, barra_fields)
            one_day_weight_result = self.get_one_day_weight(
                expected_return.loc[date].values, self.tradable_symbols_on_factor_calculation_dates.loc[date].values,
                date_i, self.factor_calculation_dates.size, self.symbols.size,
                one_day_barra_style_exposure, one_day_barra_industries, one_day_barra_covariance_matrix,
                is_in_bench_index.loc[date].values, bench_index_weights.loc[date].values, previous_weights, max_turnover)
            if one_day_weight_result is not None:
                previous_weights = one_day_weight_result[0]
                weights.loc[date] = one_day_weight_result[0]
                weights_hedged.loc[date] = one_day_weight_result[1]
                count += 1
        weights = self.allocate_portfolio(weights, bench_index_weights)
        # weights_hedged = self.allocate_portfolio(weights_hedged, bench_index_weights)
        return weights

    def get_index_weights_on_factor_calculation_dates(self, bench_index_name_or_weight_path: str) -> pd.DataFrame:
        if bench_index_name_or_weight_path.startswith("file://"):
            index_weights = pd.read_csv(bench_index_name_or_weight_path.removeprefix("file://"))
            index_weights = index_weights.set_index(index_weights.columns[0])
            index_weights.columns = index_weights.columns.astype(float).astype(int)
            index_weights = index_weights.sort_index(axis=1)
        else:
            index_weights = get_index_weights_by_name(bench_index_name_or_weight_path)
        index_weights = self.universe.align_df_with_trade_dates_and_symbols(index_weights)
        index_weights = fill_empty_rows_with_previous_not_empty_row(index_weights)
        return (index_weights / 100).shift(1).reindex(self.factor_calculation_dates)  # ??????

    def get_barra_industries_on_factor_calculation_dates(self):
        raw_barra_industries = self.universe.get_transform("barra", "am_marketdata_stock_barra_rsk", "ind").fillna(
            method="pad").reindex(self.factor_calculation_dates)
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

    def get_one_day_weight(self,
                           one_day_expected_return,
                           one_day_tradable_symbols,
                           date_i, total_date_size, symbol_size,
                           one_day_barra_style_exposure, one_day_barra_industries, one_day_barra_covariance_matrix,
                           one_day_is_in_bench_index, one_day_bench_index_weights, previous_weights,
                           max_turnover=1):
        tic = time()
        logging.info(f'Period {date_i + 1}/{total_date_size} optimization start...')

        # shouldn't have na in comb_factor...??? first few periods
        if (sum(~np.isnan(one_day_expected_return)) == 0) | (sum(one_day_expected_return) == 0):
            return None

        symbols = one_day_tradable_symbols
        na_cond = ~np.isnan(one_day_expected_return)

        na_cond = np.logical_and(na_cond, ~np.isnan(one_day_barra_style_exposure.sum(axis=0)))
        isin_idx_this_day = one_day_is_in_bench_index
        use_symbols = np.logical_and(na_cond, np.logical_or(symbols, isin_idx_this_day))
        use_index = np.where(use_symbols)[0]
        # symbols in bench index but not in our universe.
        set0_symbols = np.logical_and(use_symbols, ~symbols)
        set0_index = np.where(set0_symbols)[0]

        used_single_expected_return = one_day_expected_return[use_index]
        used_one_day_barra_style_exposure = one_day_barra_style_exposure.T[use_index]
        used_one_day_barra_industries = one_day_barra_industries.T[use_index]
        used_one_day_bench_index_weights = one_day_bench_index_weights[use_index]
        used_one_day_isin_idx_this_day = isin_idx_this_day.T[use_index]
        set0_vector = np.zeros(symbol_size)
        set0_vector[set0_index] = 1
        set0_vector = set0_vector[use_index]

        style_factor_len = one_day_barra_style_exposure.shape[0]
        ind_factor_len = one_day_barra_industries.shape[0]

        # ---------- mosek optimization
        GT = scipy.linalg.cholesky(one_day_barra_covariance_matrix.values, lower=False)
        barra_this_day = np.vstack((used_one_day_barra_industries.T, used_one_day_barra_style_exposure.T)).T
        GT = GT.dot(barra_this_day.T)
        TE = self.annualized_tracking_error
        n = len(use_index)
        max_alloc = self.max_allocation_ratio  # !!! 细致处理需要定位到每只票在指数中的权重，可以之后再加
        ind_bound = self.industry_bound
        style_expo_bound = self.style_exposure_bound
        transaction_cost = self.transaction_cost
        wgt_in_idx_budget = self.weight_in_index_budget
        wgt_deviation = self.weight_deviation

        # some period can not reach optimal result under above constraints
        flag = True
        while flag:
            try:
                opt_w = MosekOptimizer.mosek_sharpe_opt_by_diff_new(n, used_single_expected_return, GT,
                                                        max_alloc, used_one_day_bench_index_weights, TE,
                                                        used_one_day_barra_industries.T, ind_bound,
                                                        used_one_day_barra_style_exposure.T, style_expo_bound,
                                                        previous_weights[use_index], transaction_cost,
                                                        style_factor_len, ind_factor_len,
                                                        set0_vector,
                                                        used_one_day_isin_idx_this_day.astype(float), wgt_in_idx_budget,
                                                        wgt_deviation, max_turnover)
                flag = False
            except:
                logging.info(f'Warning!!! {date_i + 1}/{total_date_size} can not reach optimal solution', exc_info=True)
                used_single_expected_return = (
                                                          used_single_expected_return - used_single_expected_return.mean()) / used_single_expected_return.std()  # Standarize !!!
                TE += 1
                max_alloc += 0.005
                ind_bound += 0.02
                wgt_deviation += 0.005
                max_turnover += 0.2

        one_day_weight = np.full((symbol_size), 0, dtype=float)
        one_day_weight_hedged = np.full((symbol_size), 0, dtype=float)
        one_day_weight[use_index] = (opt_w + used_one_day_bench_index_weights)/(opt_w + used_one_day_bench_index_weights).sum()
        one_day_weight_hedged[use_index] = opt_w# - used_one_day_bench_index_weights
        logging.info(f'{date_i + 1}/{total_date_size} done in {np.round(time() - tic, 2)}s')
        logging.info(f"Weights in benchmark: {one_day_weight[use_index].dot(used_one_day_isin_idx_this_day.astype(float))}")
        return one_day_weight, one_day_weight_hedged, date_i

    def allocate_portfolio(self, port_weight, bench_index_weights) -> pd.DataFrame:
        first_weight_not_nan_factor_calculation_date = port_weight.index[
            np.where(~(port_weight == 0).all(axis=1))[0][0]]
        port_weight = port_weight * self.allocate_portfolio_ratio + bench_index_weights * (
                    1 - self.allocate_portfolio_ratio)
        port_weight.loc[port_weight.index < first_weight_not_nan_factor_calculation_date] = 0
        return port_weight

    @staticmethod
    def mosek_sharpe_opt_by_diff(n, mu, GT, max_alloc, idx_wgt, TE,
                                 ind_dist, ind_bound,
                                 style_expo, style_bound,
                                 wgt_last_period, transaction_cost,
                                 style_factor_len, ind_factor_len,
                                 set0_index,
                                 isin_idx, wgt_in_idx_budget, wgt_deviation):

        with Model("Maximize Sharpe Ratio") as M:
            # Redirect log output from the solver to stdout for debugging.
            # if uncommented.
            # M.setLogHandler(sys.stdout)

            y = M.variable("y", n)
            z = M.variable("z", 1, Domain.greaterThan(0.0))
            t = M.variable("t", 1, Domain.greaterThan(0.0))

            M.constraint('budget', Expr.sum(y), Domain.equalsTo(0))
            M.constraint('expcted return', Expr.sub(Expr.dot(mu, y),
                                                    z), Domain.equalsTo(0))
            M.constraint('risk', Expr.vstack(t, Expr.mul(GT, y)), Domain.inQCone())
            # Risk neutral with index / or tracking error
            M.constraint('tracking error', Expr.sub(t,
                                                    Expr.mul(TE, z)
                                                    ), Domain.lessThan(0))
            M.constraint('weight deviation-', Expr.sub(y,
                                                       Expr.mulElm([-wgt_deviation] * n, Var.repeat(z, n))
                                                       ), Domain.greaterThan(0))

            M.constraint('weight deviation', Expr.sub(y,
                                                      Expr.mulElm([wgt_deviation] * n, Var.repeat(z, n))
                                                      ), Domain.lessThan(0))
            # Industry distribution
            M.constraint('industry-', Expr.sub(Expr.mul(ind_dist, y),
                                               Expr.mulElm([-ind_bound] * ind_factor_len,
                                                           Var.repeat(z, ind_factor_len)),
                                               ), Domain.greaterThan(0))

            M.constraint('industry', Expr.sub(Expr.mul(ind_dist, y),
                                              Expr.mulElm([ind_bound] * ind_factor_len, Var.repeat(z, ind_factor_len)),
                                              ), Domain.lessThan(0))
            # Style exposure
            M.constraint('style-', Expr.sub(Expr.mul(style_expo, y),
                                            Expr.mulElm([-style_bound] * style_factor_len,
                                                        Var.repeat(z, style_factor_len)),
                                            ), Domain.greaterThan(0))

            M.constraint('style', Expr.sub(Expr.mul(style_expo, y),
                                           Expr.mulElm([style_bound] * style_factor_len,
                                                       Var.repeat(z, style_factor_len)),
                                           ), Domain.lessThan(0))
            M.constraint('isin_idx_budget', Expr.sub(Expr.sum(Expr.mulElm(isin_idx, y)),
                                                     Expr.dot([wgt_in_idx_budget], z)), Domain.greaterThan(0))
            M.constraint('long only', Expr.sub(y,
                                               Expr.mulElm(-idx_wgt, Var.repeat(z, n))
                                               ), Domain.greaterThan(0))
            # M.constraint('turnover', Expr.sub(Expr.sum(c), Expr.mul(z, max_turnover)), Domain.lessThan(0))

            # M.constraint('tz', Expr.hstack(c, Expr.constTerm(n, 1.0), Expr.sub(y,
            #                                                                    Expr.mulElm(wgt_last_period,
            #                                                                                Var.repeat(z, n))
            #                                                                    )
            #                                ), Domain.inPPowerCone(2.9/3.0))

            # bench index members not in universe
            M.constraint('ST', Expr.mulElm(Expr.sub(y,
                                                    Expr.mulElm(-idx_wgt, Var.repeat(z, n)))
                                           , set0_index), Domain.equalsTo(0))
            M.objective('obj', ObjectiveSense.Minimize, t)

            # Solves the model.
            M.solve()

            return y.level(), z.level()

    @staticmethod
    def mosek_sharpe_opt_by_diff_new(n, mu, GT, max_alloc, idx_wgt, TE,
                                 ind_dist, ind_bound,
                                 style_expo, style_bound,
                                 wgt_last_period, transaction_cost,
                                 style_factor_len, ind_factor_len,
                                 set0_index,
                                 isin_idx, wgt_in_idx_budget, wgt_deviation, max_turnover):

        with Model("Maximize Expect Return") as M:
            # Redirect log output from the solver to stdout for debugging.
            # if uncommented.
            # M.setLogHandler(sys.stdout)

            x = M.variable("x", n)
            z = M.variable("z", n, Domain.greaterThan(0.0))

            M.constraint('budget', Expr.sum(Expr.add(x, idx_wgt)), Domain.equalsTo(1))
            # Risk neutral with index / or tracking error
            M.constraint('tracking', Expr.vstack(TE, Expr.mul(GT, x)), Domain.inQCone())

            M.constraint('weight deviation-', Expr.sub(x,
                                                       -wgt_deviation
                                                       ), Domain.greaterThan(0))

            M.constraint('weight deviation', Expr.sub(x,
                                                      wgt_deviation
                                                      ), Domain.lessThan(0))
            # Industry distribution
            M.constraint('industry-', Expr.sub(Expr.mul(ind_dist, x),
                                               -ind_bound
                                               ), Domain.greaterThan(0))

            M.constraint('industry', Expr.sub(Expr.mul(ind_dist, x),
                                              ind_bound
                                              ), Domain.lessThan(0))
            # Style exposure
            M.constraint('style-', Expr.sub(Expr.mul(style_expo, x),
                                            -style_bound
                                            ), Domain.greaterThan(0))

            M.constraint('style', Expr.sub(Expr.mul(style_expo, x),
                                           style_bound
                                           ), Domain.lessThan(0))
            # M.constraint('isin_idx_budget', Expr.sub(Expr.sum(Expr.mulElm(isin_idx, x)),
            #                                          wgt_in_idx_budget), Domain.greaterThan(0))
            M.constraint('long only', Expr.sub(x,
                                               -idx_wgt
                                               ), Domain.greaterThan(0))

            # M.constraint('turnover +', Expr.sub(Expr.sub(x, wgt_last_period), z), Domain.lessThan(0))
            # M.constraint('turnover -', Expr.add(Expr.sub(x, wgt_last_period), z), Domain.greaterThan(0))
            # M.constraint('turnover sum', Expr.sub(Expr.sum(z), max_turnover), Domain.equalsTo(0))

            # M.constraint('tz', Expr.hstack(c, Expr.constTerm(n, 1.0), Expr.sub(y,
            #                                                                    Expr.mulElm(wgt_last_period,
            #                                                                                Var.repeat(z, n))
            #                                                                    )
            #                                ), Domain.inPPowerCone(2.9/3.0))

            # bench index members not in universe
            M.constraint('ST', Expr.mulElm(Expr.sub(x,
                                                    -idx_wgt)
                                           , set0_index), Domain.equalsTo(0))
            M.objective('obj', ObjectiveSense.Maximize, Expr.dot(mu, x))

            # Solves the model.
            M.solve()

            return x.level()

