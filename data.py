import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

"""
This script exists as sandbox for messing around with yfinance
data fetches, it is not intended to be a production data source and will
change frequently.
"""

ticker_list_tech = ["AMZN", "AAPL", "GOOGL"]
data_df = pd.DataFrame()
​
# Example fetch for Amazon data
for i in ticker_list_tech:
​
    yf_df = yf.Ticker(i)
    data_hist = yf_df.history(period="10y")
    data_hist.index.name = "Date"
    data_hist = data_hist.reset_index().rename(columns={'Date': 'date'})
    data_hist = data_hist.reset_index(drop=True)
    data_hist["date"] = pd.DatetimeIndex(data_hist["date"]).strftime('%Y-%m-%d')
    data_hist = data_hist.reset_index(drop=True)
    data_hist["date"] = pd.to_datetime(data_hist["date"])
    data_hist["ticker"] = i
    data_hist["industry"] = "tech"
    plt.plot(data_hist.index, data_hist["Close"])
    plt.show()
    print(f"dataframe shape ={data_hist.shape}")
    print(data_hist["date"])
data_df = data_df.append(data_hist)
print(data_df)