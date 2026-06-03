# Volatility Modeling and Risk Forecasting with GARCH

## 1. Introduction
This project analyzes volatility in S&P 500, Euro Stoxx 50, Nikkei 225, and FTSE 100. It uses daily closing prices from Yahoo Finance to compute log returns, examine volatility clustering through EDA, and perform preliminary analysis that motivates GARCH modeling. The project then estimates AR(1)-GARCH(1,1) models with Student-t innovations to model conditional volatility, generate volatility forecasts, and evaluate 99% Value-at-Risk performance through an out-of-sample backtest.

## 2. Data
The data are obtained from Yahoo Finance using the yfinance package and include the daily closing prices of S&P 500, Euro Stoxx 50, Nikkei 225, and FTSE 100 between 2007-04-01 and 2026-04-01.

## 3. Exploratory Data Analysis

### 3.1 Price Series

![FTSE 100 prices](outputs/plots/^FTSE_prices_plot.png)

![S&P 500 prices](outputs/plots/^GSPC_prices_plot.png)

![Nikkei 225 prices](outputs/plots/^N225_prices_plot.png)

![Euro Stoxx 50 prices](outputs/plots/^STOXX50E_prices_plot.png)

### 3.2 Return Series

The log return plots show that returns fluctuate around zero but exhibit clear periods of large clustered shocks (e.g. 2008 financial crisis and 2020 COVID-19 pandemic).

![FTSE 100 log returns](outputs/plots/^FTSE_log_returns_plot.png)

![S&P 500 log returns](outputs/plots/^GSPC_log_returns_plot.png)

![Nikkei 225 log returns](outputs/plots/^N225_log_returns_plot.png)

![Euro Stoxx 50 log returns](outputs/plots/^STOXX50E_log_returns_plot.png)

### 3.3 Return Distributions

The return distributions are centered close to zero but show negative skewness and high kurtosis, indicating heavier tails than a normal distribution.

| Index | Mean | Std. Dev. | Skewness | Kurtosis |
|---|---:|---:|---:|---:|
| FTSE | 0.0114 | 1.1844 | -0.2994 | 11.1410 |
| GSPC | 0.0382 | 1.2994 | -0.6253 | 11.1280 |
| N225 | 0.0311 | 1.5496 | -0.4663 | 8.3043 |
| STOXX50E | 0.0084 | 1.4447 | -0.2565 | 8.0194 |

![FTSE 100 return distribution](outputs/plots/^FTSE_log_returns_distribution.png)

![S&P 500 return distribution](outputs/plots/^GSPC_log_returns_distribution.png)

![Nikkei 225 return distribution](outputs/plots/^N225_log_returns_distribution.png)

![Euro Stoxx 50 return distribution](outputs/plots/^STOXX50E_log_returns_distribution.png)

### 3.4 Rolling Volatility

The rolling volatility plots once again show that volatility changes over time and tends to remain elevated after large market shocks.

![FTSE 100 rolling volatility](outputs/plots/^FTSE_rolling_volatility_plot.png)

![S&P 500 rolling volatility](outputs/plots/^GSPC_rolling_volatility_plot.png)

![Nikkei 225 rolling volatility](outputs/plots/^N225_rolling_volatility_plot.png)

![Euro Stoxx 50 rolling volatility](outputs/plots/^STOXX50E_rolling_volatility_plot.png)

## 4. Preliminary Tests

### 4.1 Ljung-Box Tests on Log Returns

The p-value of the Ljung-Box Tests on squared log returns is very small (numerically zero), which indicates volatility clustering and suggests GARCH modeling.

| index | series | lag | lb_stat | lb_pvalue |
| --- | --- | --- | --- | --- |
| ^FTSE | returns | 10 | 47.7769 | 6.817e-07 |
| ^FTSE | returns | 20 | 63.5542 | 1.979e-06 |
| ^FTSE | squared_returns | 10 | 2813 | 0 |
| ^FTSE | squared_returns | 20 | 4199 | 0 |
| ^GSPC | returns | 10 | 137.7233 | 1.233e-24 |
| ^GSPC | returns | 20 | 184.5889 | 1.224e-28 |
| ^GSPC | squared_returns | 10 | 5061 | 0 |
| ^GSPC | squared_returns | 20 | 6666 | 0 |
| ^N225 | returns | 10 | 25.1781 | 0.005 |
| ^N225 | returns | 20 | 51.4986 | 0.0001347 |
| ^N225 | squared_returns | 10 | 2012 | 0 |
| ^N225 | squared_returns | 20 | 2642 | 0 |
| ^STOXX50E | returns | 10 | 34.4138 | 0.0001571 |
| ^STOXX50E | returns | 20 | 45.2548 | 0.001 |
| ^STOXX50E | squared_returns | 10 | 1690 | 0 |
| ^STOXX50E | squared_returns | 20 | 2409 | 0 |

### 4.2 Stationarity Tests

The ADF tests reject the unit-root null for time series of all four indices, so the stationarity condition is satisfied for GARCH modeling.

| index | ADF Statistic | p-value | lags used | observations | 1% critical value | 5% critical value | 10% critical value |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ^FTSE | -28.0801 | 0 | 5 | 4370 | -3.4318 | -2.8622 | -2.5671 |
| ^GSPC | -15.6464 | 1.617e-28 | 17 | 4358 | -3.4319 | -2.8622 | -2.5671 |
| ^N225 | -70.3178 | 0 | 0 | 4375 | -3.4318 | -2.8622 | -2.5671 |
| ^STOXX50E | -31.4142 | 0 | 4 | 4371 | -3.4318 | -2.8622 | -2.5671 |

### 4.3 ARCH LM Tests

The ARCH LM tests show very strong evidence of ARCH effects in all four return series after fitting an AR(1) mean model. This suggests that a GARCH-type model is appropriate for modeling these series.

| index | mean_model | nlags | lm_stat | lm_pvalue | f_stat | f_pvalue |
| --- | --- | --- | --- | --- | --- | --- |
| ^FTSE | AR(1) | 10 | 961.7744 | 3.198e-200 | 123.0391 | 1.086e-226 |
| ^GSPC | AR(1) | 10 | 1235 | 4.179e-259 | 171.7683 | 2.33e-305 |
| ^N225 | AR(1) | 10 | 785.6011 | 2.569e-162 | 95.5562 | 2.518e-179 |
| ^STOXX50E | AR(1) | 10 | 675.4491 | 1.168e-138 | 79.7057 | 6.234e-151 |

## 5. Model Specification

$$
\begin{aligned}
r_t &= \mu + \phi r_{t-1} + \varepsilon_t, \\
\varepsilon_t &= \sigma_t z_t, \\
\sigma_t^2 &= \omega + \alpha \varepsilon_{t-1}^{2} + \beta \sigma_{t-1}^{2}, \\
z_t &\sim t_\nu.
\end{aligned}
$$

## 6. Estimation Results

### 6.1 Model Fit

| index | model | log_likelihood | AIC | BIC |
| --- | --- | --- | --- | --- |
| ^FTSE | AR(1)-GARCH(1,1)-t | -5909 | 1.183e+04 | 1.187e+04 |
| ^GSPC | AR(1)-GARCH(1,1)-t | -6017 | 1.205e+04 | 1.208e+04 |
| ^N225 | AR(1)-GARCH(1,1)-t | -7392 | 1.48e+04 | 1.483e+04 |
| ^STOXX50E | AR(1)-GARCH(1,1)-t | -6939 | 1.389e+04 | 1.393e+04 |

### 6.2 Parameter Estimates

For all four indices, the AR(1)-GARCH(1, 1) model shows highly significant volatility clustering with persistent GARCH effects and fat-tailed return shocks.

| index | model | parameter | estimate | std_error | t_stat | p_value |
| --- | --- | --- | --- | --- | --- | --- |
| ^FTSE | AR(1)-GARCH(1,1)-t | Const | 0.0521 | 0.0118 | 4.4105 | 1.031e-05 |
| ^FTSE | AR(1)-GARCH(1,1)-t | ^FTSE[1] | -0.0056 | 0.0154 | -0.3646 | 0.7154 |
| ^FTSE | AR(1)-GARCH(1,1)-t | omega | 0.0311 | 0.0071 | 4.3921 | 1.123e-05 |
| ^FTSE | AR(1)-GARCH(1,1)-t | alpha[1] | 0.1311 | 0.0192 | 6.8379 | 8.037e-12 |
| ^FTSE | AR(1)-GARCH(1,1)-t | beta[1] | 0.8485 | 0.021 | 40.459 | 0 |
| ^FTSE | AR(1)-GARCH(1,1)-t | nu | 5.8568 | 0.5162 | 11.3469 | 7.686e-30 |
| ^GSPC | AR(1)-GARCH(1,1)-t | Const | 0.1104 | 0.0113 | 9.7755 | 1.435e-22 |
| ^GSPC | AR(1)-GARCH(1,1)-t | ^GSPC[1] | -0.0575 | 0.0147 | -3.9107 | 9.202e-05 |
| ^GSPC | AR(1)-GARCH(1,1)-t | omega | 0.021 | 0.0047 | 4.5121 | 6.42e-06 |
| ^GSPC | AR(1)-GARCH(1,1)-t | alpha[1] | 0.1471 | 0.0151 | 9.7149 | 2.604e-22 |
| ^GSPC | AR(1)-GARCH(1,1)-t | beta[1] | 0.8522 | 0.0132 | 64.4541 | 0 |
| ^GSPC | AR(1)-GARCH(1,1)-t | nu | 4.9918 | 0.3798 | 13.1448 | 1.823e-39 |
| ^N225 | AR(1)-GARCH(1,1)-t | Const | 0.0834 | 0.0174 | 4.8091 | 1.516e-06 |
| ^N225 | AR(1)-GARCH(1,1)-t | ^N225[1] | -0.0392 | 0.0155 | -2.5382 | 0.0111 |
| ^N225 | AR(1)-GARCH(1,1)-t | omega | 0.0645 | 0.0169 | 3.819 | 0.000134 |
| ^N225 | AR(1)-GARCH(1,1)-t | alpha[1] | 0.1144 | 0.0184 | 6.2288 | 4.7e-10 |
| ^N225 | AR(1)-GARCH(1,1)-t | beta[1] | 0.8597 | 0.022 | 39.0374 | 0 |
| ^N225 | AR(1)-GARCH(1,1)-t | nu | 6.5603 | 0.6572 | 9.9818 | 1.831e-23 |
| ^STOXX50E | AR(1)-GARCH(1,1)-t | Const | 0.0762 | 0.0153 | 4.9783 | 6.416e-07 |
| ^STOXX50E | AR(1)-GARCH(1,1)-t | ^STOXX50E[1] | -0.0331 | 0.0149 | -2.2292 | 0.0258 |
| ^STOXX50E | AR(1)-GARCH(1,1)-t | omega | 0.0353 | 0.0103 | 3.4067 | 0.0006576 |
| ^STOXX50E | AR(1)-GARCH(1,1)-t | alpha[1] | 0.1098 | 0.0192 | 5.7258 | 1.03e-08 |
| ^STOXX50E | AR(1)-GARCH(1,1)-t | beta[1] | 0.8781 | 0.0206 | 42.5703 | 0 |
| ^STOXX50E | AR(1)-GARCH(1,1)-t | nu | 5.3101 | 0.4259 | 12.4685 | 1.109e-35 |

### 6.3 Volatility Persistence

The large values of alpha[1] + beta[1] suggest high volatility persistence across four indices, with ^GSPC showing near-integrated volatility persistence at 0.9993, followed by ^STOXX50E, ^FTSE, and ^N225.

| index | omega | alpha[1] | beta[1] | alpha_beta |
| --- | --- | --- | --- | --- |
| ^GSPC | 0.021 | 0.1471 | 0.8522 | 0.9993 |
| ^STOXX50E | 0.0353 | 0.1098 | 0.8781 | 0.988 |
| ^FTSE | 0.0311 | 0.1311 | 0.8485 | 0.9795 |
| ^N225 | 0.0645 | 0.1144 | 0.8597 | 0.974 |

## 7. Model Diagnostics

The Ljung-Box tests show that the AR(1)-GARCH(1, 1) model has sufficiently removed autocorrelation in the standardized residuals across four indices. For squared standardized residuals, the model performs well in removing autocorrelation in ^FTSE and ^STOXX50E, but it does not fully capture volatility clustering for ^GSPC and ^N225.

| index | model | test | lag | stat | p_value |
| --- | --- | --- | --- | --- | --- |
| ^FTSE | AR(1)-GARCH(1,1)-t | lb_std_resid | 10 | 12.9257 | 0.2279 |
| ^FTSE | AR(1)-GARCH(1,1)-t | lb_std_resid | 20 | 23.713 | 0.2552 |
| ^FTSE | AR(1)-GARCH(1,1)-t | lb_std_resid_squared | 10 | 8.4607 | 0.5839 |
| ^FTSE | AR(1)-GARCH(1,1)-t | lb_std_resid_squared | 20 | 17.7075 | 0.6067 |
| ^GSPC | AR(1)-GARCH(1,1)-t | lb_std_resid | 10 | 14.275 | 0.1608 |
| ^GSPC | AR(1)-GARCH(1,1)-t | lb_std_resid | 20 | 31.4188 | 0.0499 |
| ^GSPC | AR(1)-GARCH(1,1)-t | lb_std_resid_squared | 10 | 5.6836 | 0.8411 |
| ^GSPC | AR(1)-GARCH(1,1)-t | lb_std_resid_squared | 20 | 11.8408 | 0.9214 |
| ^N225 | AR(1)-GARCH(1,1)-t | lb_std_resid | 10 | 8.6317 | 0.5674 |
| ^N225 | AR(1)-GARCH(1,1)-t | lb_std_resid | 20 | 15.027 | 0.7749 |
| ^N225 | AR(1)-GARCH(1,1)-t | lb_std_resid_squared | 10 | 30.0278 | 0.0008477 |
| ^N225 | AR(1)-GARCH(1,1)-t | lb_std_resid_squared | 20 | 44.6328 | 0.0012 |
| ^STOXX50E | AR(1)-GARCH(1,1)-t | lb_std_resid | 10 | 8.4025 | 0.5896 |
| ^STOXX50E | AR(1)-GARCH(1,1)-t | lb_std_resid | 20 | 23.0496 | 0.2864 |
| ^STOXX50E | AR(1)-GARCH(1,1)-t | lb_std_resid_squared | 10 | 5.9099 | 0.8228 |
| ^STOXX50E | AR(1)-GARCH(1,1)-t | lb_std_resid_squared | 20 | 20.41 | 0.4326 |

The following ACF/PACF plots show the autocorrelation structure of the standardized residuals and squared standardized residuals from the fitted AR(1)-GARCH(1,1)-t models. As most autocorrelations lie within the confidence bands after lag 0, the models appear to fit relatively well.
![FTSE 100 residual diagnostics](outputs/plots/^FTSE_garch_residual_acf_pacf.png)

![S&P 500 residual diagnostics](outputs/plots/^GSPC_garch_residual_acf_pacf.png)

![Nikkei 225 residual diagnostics](outputs/plots/^N225_garch_residual_acf_pacf.png)

![Euro Stoxx 50 residual diagnostics](outputs/plots/^STOXX50E_garch_residual_acf_pacf.png)

## 8. Volatility Forecasting

### 8.1 Conditional and Realized Volatility

![FTSE 100 conditional volatility](outputs/plots/^FTSE_garch_volatility_plot.png)

![S&P 500 conditional volatility](outputs/plots/^GSPC_garch_volatility_plot.png)

![Nikkei 225 conditional volatility](outputs/plots/^N225_garch_volatility_plot.png)

![Euro Stoxx 50 conditional volatility](outputs/plots/^STOXX50E_garch_volatility_plot.png)

### 8.2 Forecasts

| index | model | h | forecast_mean | forecast_variance | forecast_volatility | annualized_forecast_volatility |
| --- | --- | --- | --- | --- | --- | --- |
| ^FTSE | AR(1)-GARCH(1,1)-t | 21 | 0.0518 | 0.8398 | 0.9164 | 14.5478 |
| ^GSPC | AR(1)-GARCH(1,1)-t | 21 | 0.1044 | 0.8293 | 0.9106 | 14.4561 |
| ^N225 | AR(1)-GARCH(1,1)-t | 21 | 0.0803 | 3.1675 | 1.7797 | 28.2525 |
| ^STOXX50E | AR(1)-GARCH(1,1)-t | 21 | 0.0738 | 1.3499 | 1.1619 | 18.4439 |

![FTSE 100 volatility forecast](outputs/plots/^FTSE_volatility_forecast_plot.png)

![S&P 500 volatility forecast](outputs/plots/^GSPC_volatility_forecast_plot.png)

![Nikkei 225 volatility forecast](outputs/plots/^N225_volatility_forecast_plot.png)

![Euro Stoxx 50 volatility forecast](outputs/plots/^STOXX50E_volatility_forecast_plot.png)

## 9. VaR Backtesting

The 99% VaR backtesting using an 80%-20% train-test data split shows that the model's VaR forecasts are broadly well-calibrated for all four indices. However, ^GSPC has the most exceedances and is the closest to potential underestimation of tail risk.

| index | model | var_level | observations | exceedances | expected_exceedances | exceedance_rate | kupiec_lr | kupiec_pvalue |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ^FTSE | AR(1)-GARCH(1,1)-t | 99% | 875 | 10 | 8.75 | 0.0114 | 0.1724 | 0.678 |
| ^GSPC | AR(1)-GARCH(1,1)-t | 99% | 875 | 14 | 8.75 | 0.016 | 2.692 | 0.1009 |
| ^N225 | AR(1)-GARCH(1,1)-t | 99% | 875 | 9 | 8.75 | 0.0103 | 0.0071 | 0.9326 |
| ^STOXX50E | AR(1)-GARCH(1,1)-t | 99% | 875 | 11 | 8.75 | 0.0126 | 0.5404 | 0.4623 |

![FTSE 100 VaR backtest](outputs/plots/^FTSE_var_backtest_plot.png)

![S&P 500 VaR backtest](outputs/plots/^GSPC_var_backtest_plot.png)

![Nikkei 225 VaR backtest](outputs/plots/^N225_var_backtest_plot.png)

![Euro Stoxx 50 VaR backtest](outputs/plots/^STOXX50E_var_backtest_plot.png)

## 10. Limitations

One limitation of the project is that the AR(1)-GARCH(1, 1) model does not fully remove the autocorrelation of the squared standardized residuals for ^GSPC and ^N225. To address this, future projects could compare richer volatility models such as GJR-GARCH, EGARCH, or higher-order GARCH specifications. Another limitation is that each index is modeled separately, so the analysis does not capture cross-market dependence or spillover effects across global equity markets. 