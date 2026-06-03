# GARCH Volatility Modeling

This project analyzes volatility in S&P 500, Euro Stoxx 50, Nikkei 225, and FTSE 100. It uses daily closing prices from Yahoo Finance to compute log returns, examine volatility clustering through EDA, and perform preliminary analysis that motivates GARCH modeling. The project then estimates AR(1)-GARCH(1,1) models with Student-t innovations to model conditional volatility, generate volatility forecasts, and evaluate 99% Value-at-Risk performance through an out-of-sample backtest.

## Project Structure

- `src/data.py`: downloads price data and computes log returns
- `src/eda.py`: produces exploratory plots and summary statistics
- `src/GARCH.py`: estimates GARCH models, diagnostics, forecasts, and VaR backtests
- `report.md`: written project report

## Methods

- Log returns
- ADF stationarity tests
- Ljung-Box tests
- ARCH LM tests
- AR(1)-GARCH(1,1) with Student-t innovations
- Conditional volatility forecasting
- 99% VaR backtesting with Kupiec test

## Reproducing the Analysis

```bash
python src/data.py
python src/eda.py
python src/GARCH.py