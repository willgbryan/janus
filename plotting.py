for i in ticker_list_tech:
    forecast_df = results_df.loc[results_df["ticker"] == i]
    hist_df = data_df.loc[data_df["ticker"] == i]
    hist_df = hist_df[hist_df["date"] >= pd.to_datetime("2023-01-01")]
    hist_close = hist_df.loc[hist_df["date"] == pd.Timestamp("2023-04-28"), "Close"].values[0]

    # Filter data to include only the rows where the date column is equal to 2023-05-05
    percent_diff_df = hist_df[hist_df["date"] == pd.Timestamp("2023-05-05")]

    # Get the "value-0.5" forecast value on 2023-05-01
    forecast_value = forecast_df.loc[forecast_df["date"] == pd.Timestamp("2023-05-01"), "Close-0.5"].values[0]

    # Calculate the difference
    diff = hist_close - forecast_value
    print(f"The difference between the forecast value and historical 'Close' value is: {diff:.2f}")

    forecast_df["Close-0.5"] = forecast_df["Close-0.5"] + diff
    forecast_df["Close-0.25"] = forecast_df["Close-0.25"] + diff
    forecast_df["Close-0.75"] = forecast_df["Close-0.75"] + diff

    # Calculate percent difference between historical "Close" value and forecasted value on 2023-05-01
    forecast_value_0505 = forecast_df.loc[forecast_df["date"] == pd.Timestamp("2023-05-05"), "Close-0.5"].values[0]
    hist_close_0505 = percent_diff_df.loc[percent_diff_df["date"] == pd.Timestamp("2023-05-05"), "Close"].values[0]
    percent_diff = abs((hist_close_0505 - forecast_value_0505) / hist_close_0505) * 100

    # Create card with percent difference
    card_text = f"Percent Difference on 2023-05-05\n{percent_diff:.2f}%"

    # Plot data
    plt.plot(hist_df["date"], hist_df["Close"], label=f"{i} Historical Data")
    plt.plot(forecast_df["date"], forecast_df["Close-0.5"], label=f"{i} Forecast 0.5")
    # plt.plot(forecast_df["date"], forecast_df["Close-0.75"], label=f"{i} Forecast 0.75")
    # plt.plot(forecast_df["date"], forecast_df["Close-0.25"], label=f"{i} Forecast 0.25")
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')

    # Add card to plot
    plt.gcf().text(0.87, 0.17, card_text, fontsize=9, ha='right', va='bottom', bbox=dict(facecolor='gray', alpha=0.2))

    plt.show()
    print(forecast_df.head)
