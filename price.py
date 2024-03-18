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

        volume = universe.load_field("volume")
        amount = universe.load_field("quote_asset_volume")
        buyer_amount = universe.load_field("taker_buy_quote_asset_volume")
        buyer_volume = universe.load_field("taker_buy_base_asset_volume")
        seller_amount = amount - buyer_amount
        seller_volume = volume - buyer_volume

        vwap_total_p = amount.rolling(self.window).sum() / volume.rolling(self.window).sum()
        vwap_buyer_p = buyer_amount.rolling(self.window).sum() / buyer_volume.rolling(self.window).sum()
        vwap_seller_p = seller_amount.rolling(self.window).sum() / seller_volume.rolling(self.window).sum()
        vwap_total = (amount/volume).rolling(self.window).mean()
        vwap_buyer = (buyer_amount/buyer_volume).rolling(self.window).mean()
        vwap_seller = (seller_amount/seller_volume).rolling(self.window).mean()

        # factor = self.calc_price_tri_sort_price_imbalance(volume, buyer_volume, volume-buyer_volume)
        # factor = self.calc_price_tri_price_imbalance(vwap_seller, vwap_total, vwap_buyer)
        # factor = self.calc_price_tri_sort_price_imbalance(vwap_seller, vwap_total, vwap_buyer)
        # factor = self.calc_price_imbalance(buyer_volume, volume-buyer_volume)
        # factor = (close-open) / (high-low)
        # factor = factor / factor.rolling(self.window*2).std()
        # factor = (factor.rolling(int(self.window/2)).median() - factor.rolling(self.window).median()) / factor.rolling(self.window).std()


        factor = self.calc_price_list_stats({"open": open, "high":high, "low":low, "close":close,
                                             "vwap_t_p": vwap_total_p, "vwap_b_p": vwap_buyer_p, "vwap_s_p": vwap_seller_p,
                                             # "vwap_t":vwap_total, "vwap_b": vwap_buyer, "vwap_s":vwap_seller,
                                             })

        factor = factor / factor.rolling(self.window).mean()
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


    def calc_price_list_stats(self, prices: dict[pd.DataFrame], stat_str="std"):
        df = {}
        for k,v in prices.items():
            df[k] = v.unstack()
        df = pd.DataFrame(df)
        if stat_str == "std":
            stat = (df[(df.T>df.median(1)).T].median(1) - df[(df.T<=df.median(1)).T].median(1))#/df.median(1)
        return stat.unstack().T


    def clean_factor(self, factor):
        factor[factor == np.inf] = np.nan
        factor[factor == -np.inf] = np.nan
        return factor