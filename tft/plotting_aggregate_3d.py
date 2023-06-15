import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D

bg_color = "#666666"  # Black
hist_color = "#000000"  # Light grey
forecast_color = "#00e6a8"  # Teal
card_bg_color = "#242526"  # Dark gray
card_text_color = "#ffffff"  # White

plt.rcParams['axes.facecolor'] = bg_color

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")

def date_to_num(dates):
    return mdates.date2num(dates)

for idx, i in enumerate(ticker_list_tech):
    forecast_df = results_df.loc[results_df["ticker"] == i]
    hist_df = data_df.loc[data_df["ticker"] == i]
    hist_df = hist_df[hist_df["date"] >= pd.to_datetime("2023-01-01")]

    hist_close = hist_df.loc[hist_df["date"] == pd.Timestamp("2023-05-05"), "Close"].values[0]

    percent_diff_df = hist_df[hist_df["date"] == pd.Timestamp("2023-05-05")]

    forecast_value = forecast_df.loc[forecast_df["date"] == pd.Timestamp("2023-05-08"), "Close-0.5"].values[0]

    diff = hist_close - forecast_value
    print(f"The difference between the forecast value and historical 'Close' value is: {diff:.2f}")

    # forecast_df["Close-0.5"] = forecast_df["Close-0.5"] + diff
    # forecast_df["Close-0.25"] = forecast_df["Close-0.25"] + diff
    # forecast_df["Close-0.75"] = forecast_df["Close-0.75"] + diff

#     forecast_value_0505 = forecast_df.loc[forecast_df["date"] == pd.Timestamp("2023-05-01"), "Close-0.5"].values[0]
#     hist_close_0505 = percent_diff_df.loc[percent_diff_df["date"] == pd.Timestamp("2023-05-05"), "Close"].values[0]
#     percent_diff = abs((hist_close_0505 - forecast_value_0505) / hist_close_0505) * 100

#     card_text = f"Percent Difference on 2023-05-05\n{percent_diff:.2f}%"

    x = date_to_num(hist_df["date"])
    y = [idx] * len(hist_df)
    z = hist_df["Close"]

    ax.plot(x, y, z, label=f"{i} Historical Data", color=hist_color, linewidth=2)

    x_forecast = date_to_num(forecast_df["date"])
    y_forecast = [idx] * len(forecast_df)
    z_forecast = forecast_df["Close-0.5"]

    ax.plot(x_forecast, y_forecast, z_forecast, label=f"{i} Forecast 0.5", color=forecast_color, linewidth=2)

ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax.set_xticklabels(ax.get_xticks(), rotation=45, ha="right")

ax.set_yticks(range(len(ticker_list_tech)))
ax.set_yticklabels(ticker_list_tech)

ax.set_xlabel("Date")
ax.set_ylabel("Tickers")
ax.set_zlabel("Close")

leg = ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), frameon=True)
for text in leg.get_texts():
    plt.setp(text, color="white")

plt.show()
