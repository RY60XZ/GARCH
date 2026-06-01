from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from statsmodels.stats.diagnostic import acorr_ljungbox

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = ROOT / "data" / "processed"
PLOTS_OUTPUTS_PATH = ROOT / "outputs" / "plots"
DATA_OUTPUTS_PATH = ROOT / "outputs" / "data"
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
PLOTS_OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
DATA_OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)

prices = pd.read_csv(PROCESSED_DATA_PATH / "prices.csv")
log_returns = pd.read_csv(PROCESSED_DATA_PATH / "log_returns.csv")
prices["Date"] = pd.to_datetime(prices["Date"])
log_returns["Date"] = pd.to_datetime(log_returns["Date"])
indices = log_returns.columns.tolist()[1:]

def format_date_axis(ax):
    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", rotation=30)

for index in indices:
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(data = log_returns, x = "Date", y = index, ax = ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Log Returns * 100")
    ax.set_title(f"Log Returns for {index}")
    format_date_axis(ax)
    plt.savefig(PLOTS_OUTPUTS_PATH / f"{index}_log_returns_plot.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(data = prices, x = "Date", y = index, ax = ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Prices")
    ax.set_title(f"Prices for {index}")
    format_date_axis(ax)
    plt.savefig(PLOTS_OUTPUTS_PATH / f"{index}_prices_plot.png")
    plt.close(fig)

summary_stats = log_returns[indices].agg(["mean", "std", "skew", "kurtosis"]).T
summary_stats.to_csv(DATA_OUTPUTS_PATH / "summary_stats.csv")

for index in indices:
    mu = log_returns[index].mean()
    sigma = log_returns[index].std()

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.histplot(
        data = log_returns[index],
        bins = 100,
        stat = "density",
        ax = ax,
    )

    x = np.linspace(min(log_returns[index]), max(log_returns[index]), 500)
    normal_pdf = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)
    sns.lineplot(x=x, y=normal_pdf, ax=ax, color="crimson")
    ax.set_xlabel("Log Returns * 100")
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of Log Returns for {index}")
    plt.savefig(PLOTS_OUTPUTS_PATH / f"{index}_log_returns_distribution.png")
    plt.close(fig)

rolling_21 = log_returns[indices].rolling(window=21).std() * np.sqrt(252)
rolling_63 = log_returns[indices].rolling(window=63).std() * np.sqrt(252)
rolling_21["Date"] = log_returns["Date"]
rolling_63["Date"] = log_returns["Date"]
rolling_21.dropna(inplace=True)
rolling_63.dropna(inplace=True)

for index in indices:
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(data = rolling_21, x = "Date", y = index, ax = ax, label = "21-day")
    sns.lineplot(data = rolling_63, x = "Date", y = index, ax = ax, label = "63-day")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Volatility (%)")
    ax.set_title(f"Rolling Annualized Volatility for {index}")
    format_date_axis(ax)
    plt.savefig(PLOTS_OUTPUTS_PATH / f"{index}_rolling_volatility_plot.png")
    plt.close(fig)

lb_tests = []

for index in indices:
    lb_returns = acorr_ljungbox(log_returns[index],lags=[10, 20], return_df=True)
    lb_returns["index"] = index
    lb_returns["series"] = "returns"
    lb_returns["lag"] = lb_returns.index
    lb_tests.append(lb_returns)

    lb_squared = acorr_ljungbox(log_returns[index] ** 2,lags=[10, 20], return_df=True)
    lb_squared["index"] = index
    lb_squared["series"] = "squared_returns"
    lb_squared["lag"] = lb_squared.index
    lb_tests.append(lb_squared)

lb_tests = pd.concat(lb_tests, ignore_index=True)
lb_tests = lb_tests[["index", "series", "lag", "lb_stat", "lb_pvalue"]]
lb_tests.to_csv(DATA_OUTPUTS_PATH / "lb_tests.csv", index=False)
