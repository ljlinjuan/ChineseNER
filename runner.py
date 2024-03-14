import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR


nobs = 4
reg_win = 200
tickers = ["SHFE.cu", "SHFE.al", "SHFE.zn", "SHFE.rb", "INE.sc", "DCE.i", "DCE.jm", "CFFEX.IF"]


data = np.load("Data_Virtual_Price.p", allow_pickle=True)
df = data[tickers]["2019":].fillna(method="ffill") # !
df = df.pct_change().fillna(0.)


def gen_var_forecast(df):

    forecast = pd.DataFrame()
    for i in range(200, len(df)+nobs, nobs):
        df_train = df.iloc[i-200:i-nobs]
        df_test = df.iloc[i-nobs:i]
        model = VAR(df_train)
        model_fitted = model.fit(2)
        lag_order = model_fitted.k_ar
        forecast_input = df_train.values[-lag_order:]
        fc = model_fitted.forecast(y=forecast_input, steps=nobs)
        forecast = forecast.append(pd.DataFrame(fc, index=df_test.index, columns=df.columns))
        print()

    return 0


gen_var_forecast(df)
