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
    "SOFI",
    "PYPL",
    "SNOW",
    "ADBE",
    "UBER",
    "ABNB",
    "FUBO",
    "AVGO",
    "ORCL",
    "ASML",
    "CSCO",
    "ACN",
]

ticker_list_energy = [
    "XOM",
    "CVX",
    "SHEL",
    "TTE",
    "COP",
    "BP",
    "EQNR",
    "ENB",
]


ticker_list_market_index = [
    "^NDX", 
    "^GSPC", 
    "^DJI",
    "^RUT",
    "^IXIC",
    
]

ticker_list_financial_services = [
    "BRK-A",
    "BRK-B",
    "V",
    "JPM",
    "MA",
    "BAC",
    "WFC",
    "MS",
    "HSBC",
]
ticker_lists = {
    "tech":ticker_list_tech, 
    "index":ticker_list_market_index, 
    "financial":ticker_list_financial_services,
    "energy":ticker_list_energy,
}

data_frames = []

for key, value in ticker_lists.items():
    for i in value:
        yf_df = yf.Ticker(i)
        data_hist = yf_df.history(period="10y")
        data_hist.index.name = "Date"
        data_hist = data_hist.reset_index().rename(columns={'Date': 'date'})
        data_hist = data_hist.reset_index(drop=True)
        data_hist["date"] = pd.DatetimeIndex(data_hist["date"]).strftime('%Y-%m-%d')
        data_hist = data_hist.reset_index(drop=True)
        data_hist["date"] = pd.to_datetime(data_hist["date"])
        data_hist["ticker"] = i
        data_hist["industry"] = key
        plt.plot(data_hist["date"], data_hist["Close"])
        data_frames.append(data_hist)
   

data_df = pd.concat(data_frames, ignore_index=True)
print(data_df)