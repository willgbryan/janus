import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from typing import List
from autots import AutoTS, load_live_daily, create_regressor


filepath = "C:/Users/willb/OneDrive/Documents/GitHub/janus/auto_ts_out/tsla_apple_60_horizon"


def get_history(key: str, ticker_list: List[str]):

    historical_data = pd.DataFrame()
    
    for ticker in ticker_list:
        yf_df = yf.Ticker(ticker)
        data_hist = yf_df.history(period="2y")
        data_hist.index.name = "Date"
        data_hist = data_hist.reset_index().rename(columns={'Date': 'date'})
        data_hist = data_hist.reset_index(drop=True)
        data_hist["date"] = pd.DatetimeIndex(data_hist["date"]).strftime('%Y-%m-%d')
        data_hist = data_hist.reset_index(drop=True)
        data_hist["date"] = pd.to_datetime(data_hist["date"])
        data_hist["ticker"] = ticker
        data_hist["industry"] = key
        plt.plot(data_hist["date"], data_hist["Close"])
        historical_data.append(data_hist)
   



def plot_auto_ts(filepath: str):
    forecast_data = pd.read_csv(filepath)
    col_set = forecast_data.columns.to_list()

    for col in col_set:
        pass

