import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

"""
This script exists as sandbox for messing around with yfinance
data fetches, it is not intended to be a production data source and will
change frequently.
"""

# Example fetch for Amazon data
amzn_df = yf.Ticker("AMZN")
amzn_hist = amzn_df.history(period="1mo")
        
# Need to double check this derivative attempt but the idea is there
amzn_hist["derivative"] = amzn_hist["Close"] - amzn_hist["Close"].shift(1)
plt.plot(amzn_hist.index, amzn_hist["Close"])
plt.show()

