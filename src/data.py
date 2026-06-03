from pathlib import Path
import yfinance as yf
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT / "data" / "raw"
PROCESSED_DATA_PATH = ROOT / "data" / "processed"
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

indices = [
    "^GSPC",
    "^STOXX50E",
    "^N225",
    "^FTSE"
]

#we choose 2007-04-01 as the starting date because yfinance's ^STOXX50E data is missing before the date
df = yf.download(tickers=indices, start="2007-04-01", end="2026-06-01", interval="1d")["Close"]
df.to_csv(RAW_DATA_PATH / "data.csv")


df.dropna(inplace=True)
df.to_csv(PROCESSED_DATA_PATH / "prices.csv")
returns = 100 * np.log(df / df.shift(1))
returns.dropna(inplace=True)
returns.to_csv(PROCESSED_DATA_PATH / "log_returns.csv")

