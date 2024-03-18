import numpy as np
import pandas as pd
from Data.universe import Universe
from Factors.generator import SingleFactorsFactory, SingleFactor


'''
Data missing at 2021-02-11 11:00:00 18486 !!
'''


class PriceFactorGenerator(SingleFactorsFactory):
    def __init__(self, window) -> None:
        super().__init__()
        self.parameters_combination = self._gen_parameters_iter_product(window=window)
        self.current_params = next(self.parameters_combination, False)

    def _has_next_factor(self) -> bool:
        return self.current_params

    def _get_next_factor(self):
        factor = PriceFactor(**self.current_params)
        self.current_params = next(self.parameters_combination, False)
        return factor


class PriceFactor(SingleFactor):
    def __init__(self, window: int):
        super().__init__(f"price_{window}", ["SimplePrice"])
        self.window = window

    def get_df(self, universe: Universe) -> pd.DataFrame:
        # # All Prices
        # open = universe.load_field("open")
        # high = universe.load_field("high")
        # low = universe.load_field("low")
        # close = universe.load_field("close")
        #
        # volume = universe.load_field("volume")
        # amount = universe.load_field("quote_asset_volume")
        # buyer_amount = universe.load_field("taker_buy_quote_asset_volume")
        # buyer_volume = universe.load_field("taker_buy_base_asset_volume")
        #
        # vwap_total = amount / volume
        # vwap_buyer = buyer_amount / buyer_volume
        # vwap_seller = (amount-buyer_amount) / (volume-buyer_volume)

        open = universe.load_field("open").shift(self.window)
        high = universe.load_field("high").rolling(self.window).max()
        low = universe.load_field("low").rolling(self.window).min()
        close = universe.load_field("close")

        volume = universe.load_field("volume").rolling(self.window).sum()
        amount = universe.load_field("quote_asset_volume").rolling(self.window).sum()
        buyer_amount = universe.load_field("taker_buy_quote_asset_volume").rolling(self.window).sum()
        buyer_volume = universe.load_field("taker_buy_base_asset_volume").rolling(self.window).sum()

        vwap_total = amount / volume
        vwap_buyer = buyer_amount / buyer_volume
        vwap_seller = (amount-buyer_amount) / (volume-buyer_volume)

        # factor = self.calc_price_tri_sort_price_imbalance(volume, buyer_volume, volume-buyer_volume)
        # factor = self.calc_price_tri_price_imbalance(vwap_seller, vwap_total, vwap_buyer)
        # factor = self.calc_price_tri_sort_price_imbalance(vwap_seller, vwap_total, vwap_buyer)
        # factor = self.calc_price_imbalance(buyer_volume, volume-buyer_volume)
        factor = (buyer_amount-(amount-buyer_amount)) / amount
        factor = factor / factor.rolling(self.window*2).std()
        # factor = (factor.rolling(int(self.window/2)).median() - factor.rolling(self.window).median()) / factor.rolling(self.window).std()
        factor = (factor.T - factor.mean(1)).T
        return self.clean_factor(factor)


    def calc_price_imbalance(self, a, b):
        return (a-b) / (a+b)


    def calc_price_tri_sort_price_imbalance(self, a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame):
        price_df = pd.DataFrame({"a": a.unstack(), "b": b.unstack(), "c": c.unstack()})
        max = price_df.max(1)
        min = price_df.min(1)
        mid = price_df.sum(1) - max - min
        imb = ((max-mid) - (mid-min)) / (max-min)
        return imb.unstack().T


    def calc_price_tri_price_imbalance(self, a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame):
        return ((a-b) - (b-c)) / (a-c)

    def clean_factor(self, factor):
        factor[factor==np.inf] = np.nan
        factor[factor == -np.inf] = np.nan
        return factor