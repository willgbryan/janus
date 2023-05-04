import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

ticker_list_tech = [
    "AMZN", 
    "AAPL", 
    "GOOGL", 
    "MSFT", 
    "NVDA", 
    "META", 
    "TSLA", 
    "AMD", 
    "INTC", 
    "CRM",    
    # "SOFI",
    # "PYPL",
    # "SNOW",
    # "ADBE",
    # "UBER",
    # "ABNB",
]

ticker_list_market_index = [
    # "^NDX", 
    # "^GSPC", 
    # "^DJI"
]

ticker_list_commodities = []
ticker_lists = [ticker_list_tech, ticker_list_market_index, ticker_list_commodities]
data_frames = []

for x in ticker_lists:
    for i in ticker_list_tech:
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
        plt.plot(data_hist["date"], data_hist["Close"])
        plt.show()
        print(i)
        print(f"dataframe shape = {data_hist.shape}")
        print(data_hist["date"])

        data_frames.append(data_hist)
    

data_df = pd.concat(data_frames, ignore_index=True)
print(data_df)