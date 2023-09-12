import numpy as np
import pandas as pd

annual_working_days = 250

def generate_performance_summary(
        daily_returns: pd.Series, 
        back_tracking_years: list[int] = None,
        skip_zero_return_head = False) -> pd.DataFrame:
    performance_start_date = __get_performance_start_date(daily_returns, skip_zero_return_head)
    summary = {}
    if performance_start_date is None:
        summary["total"] = __calculate_performance(daily_returns)
    else:
        filtered_daily_returns = daily_returns[daily_returns.index >= performance_start_date]
        summary["total"] = __calculate_performance(filtered_daily_returns)
        __fill_annual_performance(filtered_daily_returns, summary)
        __fill_back_tracking_performance(filtered_daily_returns, back_tracking_years, summary)
    return pd.DataFrame.from_dict(summary, orient="index", columns=['annual_return', 'vol', 'max_dd', 'sharpe_ratio', 'calmar_ratio', "win_rate"])

def __get_performance_start_date(daily_returns: pd.Series, skip_zero_return_head: bool) -> int:
    if not skip_zero_return_head:
        return daily_returns.index[0]
    for date_i in range(daily_returns.index.size):
        if daily_returns.iloc[date_i] != 0.0:
            if date_i > 0:
                return daily_returns.index[date_i - 1]
            else:
                return daily_returns.index[date_i]
    return None

def __fill_annual_performance(daily_returns: pd.Series, summary: dict[str, list]):
    daily_returns.index = pd.to_datetime([str(each) for each in daily_returns.index], format='%Y-%m-%d')
    year_idx = pd.Series(daily_returns.index).apply(lambda x:x.year)
    year_idx_all = year_idx.unique()
    for year in year_idx_all:
        ret_this_year = daily_returns.iloc[np.where(year_idx==year)]
        summary[year] = __calculate_performance(ret_this_year)

def __fill_back_tracking_performance(daily_returns: pd.Series, back_tracking_years: list[int], summary: dict[str, list]):
    if back_tracking_years is None:
        return
    for year in back_tracking_years:
        back_index = annual_working_days * year
        if back_index < daily_returns.index.size - 1:
            back_tracking_return = daily_returns.iloc[-back_index:]
            summary["{}_years".format(year)] = __calculate_performance(back_tracking_return)

def __calculate_performance(daily_returns: pd.Series):
    if (daily_returns == 0.0).all():
        annual_return = annual_return_std = max_draw_down = sharpe_ratio = calmar_ratio = win_rate = np.nan
    else:
        daily_returns = daily_returns.fillna(0.0)
        compound_nets = (1 + daily_returns).cumprod()
        total_return = compound_nets.iloc[-1] / compound_nets.iloc[0] - 1
        annual_return = (1 + total_return) ** (annual_working_days / daily_returns.size) - 1
        annual_return_std = daily_returns.std() * np.sqrt(annual_working_days)  # annualized
        sharpe_ratio = annual_return / annual_return_std  # annualized
        max_draw_down = np.max((np.maximum.accumulate(compound_nets) - (compound_nets)) / np.maximum.accumulate(compound_nets))
        calmar_ratio = annual_return / max_draw_down
        win_rate = daily_returns[daily_returns > 0].size / daily_returns.size
    return [annual_return, annual_return_std, max_draw_down, sharpe_ratio, calmar_ratio, win_rate]