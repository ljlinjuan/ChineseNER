from typing import Tuple

import numpy as np
import pandas as pd
from hqd import Universe
from hqd.decorator import timeit


class TradingStrategyConfig:
    @timeit
    def __init__(self, universe: Universe, trading_frequency: str) -> None:
        self.universe = universe
        self.trading_frequency = trading_frequency
        self.trade_dates = pd.Series(self.universe.get_trade_dates())
        self.trading_execution_dates, self.factor_calculation_dates = self.__get_trading_execution_and_factor_calculation_dates()
    
    def to_dict(self):
        return {
            "trading_frequency": self.trading_frequency
        }

    def __get_trading_execution_and_factor_calculation_dates(self) -> Tuple[pd.Series, pd.Series]:
        trading_execution_dates_indexes = self.__get_trading_execution_dates_indexes()
        return self.trade_dates[trading_execution_dates_indexes], self.trade_dates[trading_execution_dates_indexes - 1]

    def __get_trading_execution_dates_indexes(self) -> np.ndarray:
        if self.trading_frequency == "daily":
            return self.__get_dates_indexes(lambda date: date.day, 1)
        if self.trading_frequency == "weekly":
            return self.__get_dates_indexes(lambda date: date.isocalendar()[1], 4) # why 4?
        elif self.trading_frequency == "monthly":
            return self.__get_dates_indexes(lambda date: date.month, 14)
        elif self.trading_frequency == "quarterly":
            return self.__get_dates_indexes(lambda date: (date.month - 1) // 3 + 1, 34)
        raise Exception("Does not support trading frequency: {}".format(self.trading_frequency))
    
    def __get_dates_indexes(self, func, minimum_trading_days_gap) -> np.ndarray:
        extract = self.trade_dates.apply(func)
        # diff() is taking first diff backward, so we will get the first date of each period.
        # If we want to get last date of each period as execution date, we can add a parameter to control.
        # always add latest trade date to see the latest trading outcome.
        # last date in the time range should be trading execution date.
        switch_indexes = np.append(np.where(extract.diff() != 0), len(self.trade_dates) - 1)
        # first date in the time range should be factor calculation date, second date should be trading execution date.
        switch_indexes[0] = 1
        while True:
            index_of_index_to_remove = None
            for index_of_index, index in enumerate(switch_indexes):
                if index_of_index > 0 and (index - switch_indexes[index_of_index - 1]) < minimum_trading_days_gap:
                    index_of_index_to_remove = index_of_index
                    break
            if index_of_index_to_remove != None:
                switch_indexes = np.delete(switch_indexes, index_of_index_to_remove)
            else:
                break
        return switch_indexes