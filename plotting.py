"""As is, this is only intended to be used in a notebook environment"""

for i in ticker_list_tech:
    forecast_df = results_df.loc[results_df["ticker"] == i]
    hist_df = data_df.loc[data_df["ticker"] == i]
    hist_df = hist_df[hist_df["date"] >= pd.to_datetime("2023-01-01")]
    hist_close = hist_df.loc[hist_df["date"] == pd.Timestamp("2023-04-28"), "Close"].values[0]

    # Get the "value-0.5" forecast value on 2023-01-23
    forecast_value = forecast_df.loc[forecast_df["date"] == pd.Timestamp("2023-05-01"), "Close-0.5"].values[0]

    # Calculate the difference
    diff = hist_close - forecast_value
    print(f"The difference between the forecast value and historical 'Close' value is: {diff:.2f}")

    forecast_df["Close-0.5"] = forecast_df["Close-0.5"] + diff
    forecast_df["Close-0.25"] = forecast_df["Close-0.25"] + diff
    forecast_df["Close-0.75"] = forecast_df["Close-0.75"] + diff

    # Plot data
    plt.plot(hist_df["date"], hist_df["Close"], label=f"{i} Historical Data")
    plt.plot(forecast_df["date"], forecast_df["Close-0.5"], label=f"{i} Forecast 0.5")
    # plt.plot(forecast_df["date"], forecast_df["Close-0.75"], label=f"{i} Forecast 0.75")
    # plt.plot(forecast_df["date"], forecast_df["Close-0.25"], label=f"{i} Forecast 0.25")
    plt.xticks(rotation=45)
    print(forecast_df)
    plt.legend()
    plt.show()