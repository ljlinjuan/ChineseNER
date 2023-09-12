import pandas as pd
import numpy as np

from chameqs_common.utils.decorator import ignore_warning
from chameqs_common.data.tradable_filter import ListDateTradableFilter, STTradableFilter, SuspensionTradableFilter, UpLimitTradableFilter, DownLimitTradableFilter
from chameqs_common.data.universe import Universe


class TradesDrivenBackTest():
    def __init__(self, universe: Universe):
        self.init_balance = 1e8
        self.universe = universe
        self.trade_dates = self.universe.get_trade_dates()
        self.all_symbols = self.universe.get_tradable_symbols().columns
        self.eod_px = self.universe.wind_common.get_price_df_by_type("adjusted_close")
        self.buyside_tradable = self.universe.align_df_with_trade_dates_and_symbols(self.get_buyside_tradable_wider())
        self.sellside_tradable = self.universe.align_df_with_trade_dates_and_symbols(self.get_sellside_tradable())
        self.buyside_trading_cost = 0.0003
        self.sellside_trading_cost = 0.001+0.0003

    def get_portfolio_nav(self, target_weight: pd.DataFrame,
                      trade_price: pd.DataFrame):
        total_balance = pd.Series(index=self.trade_dates, data=self.init_balance)
        current_position = pd.DataFrame(0, index=self.trade_dates, columns=self.all_symbols)
        current_weight = pd.Series(0, index=self.trade_dates)
        transaction_fee = pd.Series(0, index=self.trade_dates)

        total_balance_val = total_balance.values
        transaction_fee_val = transaction_fee.values
        current_weight_val = current_weight.values
        target_weight_val = target_weight.values
        current_position_val = current_position.values
        trade_price_val = trade_price.values
        eod_valuation_px_val = self.eod_px.values
        buyside_tradable_val = self.buyside_tradable.values
        sellside_tradable_val = self.sellside_tradable.values

        for dt_i, dt in enumerate(self.trade_dates):
            target_pos_today = (self.init_balance * target_weight_val[dt_i, :]) / trade_price_val[dt_i, :] # 不用考虑除权出息
            target_pos_today = np.array([0 if np.isnan(i) else i for i in target_pos_today])

            if dt_i == 0:
                trades_today = target_pos_today
                buy_trades_today = trades_today * ((trades_today > 0) & buyside_tradable_val[dt_i, :])
                sell_trades_today = trades_today * ((trades_today < 0) & sellside_tradable_val[dt_i, :])
                trades_available = buy_trades_today + sell_trades_today
                current_position_val[dt_i, :] = trades_available
                current_position_weight_when_rebal = target_weight_val[dt_i, :] * (trades_available != 0)

                trades_today_pnl = trades_available @ np.nan_to_num(eod_valuation_px_val[dt_i, :] - trade_price_val[dt_i, :], nan=0)
                transaction_fee_today = buy_trades_today @ np.nan_to_num(trade_price_val[dt_i, :]) * self.buyside_trading_cost + \
                                  abs(sell_trades_today) @ np.nan_to_num(trade_price_val[dt_i, :]) * self.sellside_trading_cost
                pnl = trades_today_pnl - transaction_fee_today
            else:
                overnight_position = current_position_val[dt_i - 1] #
                trades_today = target_pos_today - overnight_position
                buy_trades_today = trades_today * ((trades_today > 0) & buyside_tradable_val[dt_i, :])
                sell_trades_today = trades_today * ((trades_today < 0) & sellside_tradable_val[dt_i, :])
                trades_available = buy_trades_today + sell_trades_today

                current_position_weight_when_last_rebal = current_position_weight_when_rebal.copy()
                current_position_weight_when_rebal = current_position_weight_when_last_rebal * (trades_available == 0) + target_weight_val[dt_i, :] * (trades_available != 0)
                symbols_need_rebal = abs(current_position_weight_when_rebal - current_position_weight_when_last_rebal) > 0
                trades_available *= symbols_need_rebal
                current_position_val[dt_i, :] = current_position_val[dt_i - 1, :] + trades_available

                overnight_pnl = overnight_position @ np.nan_to_num(eod_valuation_px_val[dt_i, :] - eod_valuation_px_val[dt_i-1, :], nan=0)
                trades_today_pnl = trades_available @ np.nan_to_num(eod_valuation_px_val[dt_i, :] - trade_price_val[dt_i, :], nan=0)
                transaction_fee_today = (buy_trades_today*symbols_need_rebal) @ np.nan_to_num(trade_price_val[dt_i, :]) * self.buyside_trading_cost + \
                                  abs((sell_trades_today*symbols_need_rebal)) @ np.nan_to_num(trade_price_val[dt_i, :]) * self.sellside_trading_cost
                pnl = overnight_pnl + trades_today_pnl - transaction_fee_today

            transaction_fee_val[dt_i] = transaction_fee_today
            total_balance_val[dt_i] = total_balance_val[dt_i - 1] + pnl
            current_weight_val[dt_i] = current_position_val[dt_i, :] @ np.nan_to_num(eod_valuation_px_val[dt_i, :])

        return total_balance, transaction_fee, current_weight

    @ignore_warning
    def get_buyside_tradable(self):
        buyside_tradable = self.universe.get_symbols()

        tradable_filters = [ListDateTradableFilter(after_list_trade_days=120),
                            STTradableFilter(after_st_trade_days=20),
                            SuspensionTradableFilter(after_suspenion_trade_days=5),
                            UpLimitTradableFilter()
                            ]
        for tradable_filter in tradable_filters:
            buyside_tradable = buyside_tradable & tradable_filter.get_filter_df(self.universe)
        return buyside_tradable

    @ignore_warning
    def get_buyside_tradable_wider(self):
        buyside_tradable = self.universe.get_symbols()

        tradable_filters = [SuspensionTradableFilter(after_suspenion_trade_days=0),
                            UpLimitTradableFilter()
                            ]
        for tradable_filter in tradable_filters:
            buyside_tradable = buyside_tradable & tradable_filter.get_filter_df(self.universe)
        return buyside_tradable


    @ignore_warning
    def get_sellside_tradable(self):
        sellside_tradable = self.universe.get_symbols()

        tradable_filters = [DownLimitTradableFilter(),
                            SuspensionTradableFilter(after_suspenion_trade_days=0)
                            ]
        for tradable_filter in tradable_filters:
            sellside_tradable = sellside_tradable & tradable_filter.get_filter_df(self.universe)
        return sellside_tradable
