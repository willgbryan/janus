import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.style.use("seaborn-darkgrid")

hist_color = "#00e6a8"  # Teal
forecast_color = "#ff4d6d"  # Pink
card_bg_color = "#242526"  # Dark gray
card_text_color = "#ffffff"  # White

for i in ticker_list_tech:
    forecast_df = results_df.loc[results_df["ticker"] == i]
    hist_df = data_df.loc[data_df["ticker"] == i]
    hist_df = hist_df[hist_df["date"] >= pd.to_datetime("2023-01-01")]
    hist_close = hist_df.loc[hist_df["date"] == pd.Timestamp("2023-04-28"), "Close"].values[0]

    percent_diff_df = hist_df[hist_df["date"] == pd.Timestamp("2023-05-05")]

    forecast_value = forecast_df.loc[forecast_df["date"] == pd.Timestamp("2023-05-01"), "Close-0.5"].values[0]

    diff = hist_close - forecast_value
    print(f"The difference between the forecast value and historical 'Close' value is: {diff:.2f}")

    forecast_df["Close-0.5"] = forecast_df["Close-0.5"] + diff
    forecast_df["Close-0.25"] = forecast_df["Close-0.25"] + diff
    forecast_df["Close-0.75"] = forecast_df["Close-0.75"] + diff

    forecast_value_0505 = forecast_df.loc[forecast_df["date"] == pd.Timestamp("2023-05-05"), "Close-0.5"].values[0]
    hist_close_0505 = percent_diff_df.loc[percent_diff_df["date"] == pd.Timestamp("2023-05-05"), "Close"].values[0]
    percent_diff = abs((hist_close_0505 - forecast_value_0505) / hist_close_0505) * 100

    card_text = f"Percent Difference on 2023-05-05\n{percent_diff:.2f}%"

    fig, ax = plt.subplots()
    ax.plot(hist_df["date"], hist_df["Close"], label=f"{i} Historical Data", color=hist_color, linewidth=2)
    ax.plot(forecast_df["date"], forecast_df["Close-0.5"], label=f"{i} Forecast 0.5", color=forecast_color, linewidth=2)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)

    leg = plt.legend(loc='upper left', frameon=True)
    for text in leg.get_texts():
        plt.setp(text, color='black')

    plt.gcf().text(0.87, 0.17, card_text, fontsize=9, ha='right', va='bottom',
                   color=card_text_color, bbox=dict(facecolor=card_bg_color, alpha=0.8, boxstyle='round,pad=0.5'))

    plt.show()
    print(forecast_df.head)
