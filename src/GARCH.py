from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from arch import arch_model
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import t, chi2
from scipy.special import xlogy

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = ROOT / "data" / "processed"
PLOTS_OUTPUTS_PATH = ROOT / "outputs" / "plots"
DATA_OUTPUTS_PATH = ROOT / "outputs" / "data"
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
PLOTS_OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
DATA_OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)

log_returns = pd.read_csv(PROCESSED_DATA_PATH / "log_returns.csv")
log_returns["Date"] = pd.to_datetime(log_returns["Date"])
indices = log_returns.columns.tolist()[1:]

#test if returns are stationary
adf_results = []
for index in indices:
    result = adfuller(log_returns[index])
    adf_results.append({
        "index": index,
        "ADF Statistic": result[0],
        "p-value": result[1],
        "lags used": result[2],
        "observations": result[3],
        "1% critical value": result[4]["1%"],
        "5% critical value": result[4]["5%"],
        "10% critical value": result[4]["10%"],
    })

adf_results = pd.DataFrame(adf_results)
adf_results.to_csv(DATA_OUTPUTS_PATH / "adf_results.csv", index=False)

#arch lm test
mean_model_results = []
arch_lm_results = []
ar1_residuals = pd.DataFrame({"Date": log_returns["Date"]})

for index in indices:
    series = log_returns[index]
    model = ARIMA(series, order=(1, 0, 0))
    result = model.fit()

    residuals = result.resid.dropna()
    ar1_residuals[index] = result.resid.values
    mean_model_results.append({
        "index": index,
        "model": "AR(1)",
        "const": result.params.get("const"),
        "ar_l1": result.params.get("ar.L1"),
        "sigma2": result.params.get("sigma2"),
        "log_likelihood": result.llf,
        "aic": result.aic,
        "bic": result.bic,
        "hqic": result.hqic,
    }
    )

    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(residuals, nlags=10)

    arch_lm_results.append({
        "index": index,
        "mean_model": "AR(1)",
        "nlags": 10,
        "lm_stat": lm_stat,
        "lm_pvalue": lm_pvalue,
        "f_stat": f_stat,
        "f_pvalue": f_pvalue,
    })

mean_model_results = pd.DataFrame(mean_model_results)
arch_lm_results = pd.DataFrame(arch_lm_results)

mean_model_results.to_csv(DATA_OUTPUTS_PATH / "mean_model_results.csv", index=False)
arch_lm_results.to_csv(DATA_OUTPUTS_PATH / "arch_lm_results.csv", index=False)
ar1_residuals.to_csv(DATA_OUTPUTS_PATH / "ar1_residuals.csv", index=False)

def format_date_axis(ax):
    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", rotation=30)

garch_model_results = []
garch_params = []
garch_diagnostics = []
garch_volatility_rows = []
garch_std_residual_rows = []
garch_forecast_rows = []

rolling_21 = log_returns[indices].rolling(window=21).std() * np.sqrt(252)
rolling_63 = log_returns[indices].rolling(window=63).std() * np.sqrt(252)

garch_volatility = pd.DataFrame({"Date": log_returns["Date"]})
garch_std_residuals = pd.DataFrame({"Date": log_returns["Date"]})

with open(DATA_OUTPUTS_PATH / "garch_summaries.txt", "w") as f:
    for index in indices:
        model_name = f"AR(1)-GARCH(1,1)-t"
        series = log_returns[index]
        model = arch_model(
            series,
            mean="AR",
            lags=1,
            vol="GARCH",
            q=1,
            p=1,
            dist="t",
        )
        result = model.fit(disp="off")

        garch_model_results.append(
            {
                "index": index,
                "model": model_name,
                "log_likelihood": result.loglikelihood,
                "AIC": result.aic,
                "BIC": result.bic,
            }
        )

        for param_name in result.params.index:
            garch_params.append({
                "index": index,
                "model": model_name,
                "parameter": param_name,
                "estimate": result.params[param_name],
                "std_error": result.std_err[param_name],
                "t_stat": result.tvalues[param_name],
                "p_value": result.pvalues[param_name],
            })
        conditional_volatility = result.conditional_volatility
        standardized_residuals = result.std_resid

        for row_index, date in log_returns["Date"].items():
            garch_volatility_rows.append({
                "Date": date,
                "index": index,
                "model": model_name,
                "conditional_volatility": conditional_volatility.loc[row_index],
                "annualized_conditional_volatility": conditional_volatility.loc[row_index] * np.sqrt(252),
                "realized_volatility_21": rolling_21.loc[row_index, index],
                "realized_volatility_63": rolling_63.loc[row_index, index],
            })

        for row_index, date in log_returns["Date"].items():
            garch_std_residual_rows.append({
                "Date": date,
                "index": index,
                "model": model_name,
                "standardized_residuals": standardized_residuals.loc[row_index],
            })
        forecast_duration = 21
        forecast = result.forecast(horizon=forecast_duration, reindex=False)
        forecast_mean = forecast.mean.iloc[-1]
        forecast_variance = forecast.variance.iloc[-1]
        last_date = log_returns["Date"].iloc[-1]
        forecast_dates = pd.bdate_range(start=last_date + pd.offsets.BDay(1), periods=forecast_duration)
        for h in range(1, forecast_duration+1):
            h_index = f"h.{h:02d}"
            garch_forecast_rows.append({
                "Date": forecast_dates[h-1],
                "index": index,
                "model": model_name,
                "h": h,
                "forecast_mean": forecast_mean.loc[h_index],
                "forecast_variance": forecast_variance.loc[h_index],
                "forecast_volatility": np.sqrt(forecast_variance.loc[h_index]),
                "annualized_forecast_volatility": np.sqrt(forecast_variance.loc[h_index]) * np.sqrt(252),
            })

        standardized_residuals.dropna(inplace=True)
        lb_std_resid = acorr_ljungbox(standardized_residuals, lags=[10, 20], return_df = True)
        lb_std_resid_squared = acorr_ljungbox(standardized_residuals ** 2, lags=[10, 20], return_df = True)
        for lag, row in lb_std_resid.iterrows():
            garch_diagnostics.append({
                "index": index,
                "model": model_name,
                "test": "lb_std_resid",
                "lag": lag,
                "stat": row["lb_stat"],
                "p_value": row["lb_pvalue"],
            })

        for lag, row in lb_std_resid_squared.iterrows():
            garch_diagnostics.append({
                "index": index,
                "model": model_name,
                "test": "lb_std_resid_squared",
                "lag": lag,
                "stat": row["lb_stat"],
                "p_value": row["lb_pvalue"],
            })

        f.write(f"\n\n{'=' * 80}\n")
        f.write(f"{index} {model_name} Summary\n")
        f.write(f"{'=' * 80}\n")
        f.write(result.summary().as_text())

garch_model_results = pd.DataFrame(garch_model_results)
garch_params = pd.DataFrame(garch_params)
garch_diagnostics = pd.DataFrame(garch_diagnostics)
garch_volatility = pd.DataFrame(garch_volatility_rows)
garch_std_residuals = pd.DataFrame(garch_std_residual_rows)
garch_forecast = pd.DataFrame(garch_forecast_rows)

garch_model_results.to_csv(DATA_OUTPUTS_PATH / "garch_model_results.csv", index=False)
garch_params.to_csv(DATA_OUTPUTS_PATH / "garch_params.csv", index=False)
garch_diagnostics.to_csv(DATA_OUTPUTS_PATH / "garch_diagnostics.csv", index=False)
garch_volatility.to_csv(DATA_OUTPUTS_PATH / "garch_conditional_volatility.csv", index=False)
garch_std_residuals.to_csv(DATA_OUTPUTS_PATH / "garch_standardized_residuals.csv", index=False)
garch_forecast.to_csv(DATA_OUTPUTS_PATH / "garch_forecast.csv", index=False)

# GARCH(2,2) robustness check
garch_22_model_results = []
garch_22_params = []
garch_22_diagnostics = []

for index in indices:
    model_name = "AR(1)-GARCH(2,2)-t"
    series = log_returns[index]
    model = arch_model(
        series,
        mean="AR",
        lags=1,
        vol="GARCH",
        p=2,
        q=2,
        dist="t",
    )
    result = model.fit(disp="off")

    garch_22_model_results.append({
        "index": index,
        "model": model_name,
        "log_likelihood": result.loglikelihood,
        "AIC": result.aic,
        "BIC": result.bic,
    })

    for param_name in result.params.index:
        garch_22_params.append({
            "index": index,
            "model": model_name,
            "parameter": param_name,
            "estimate": result.params[param_name],
            "std_error": result.std_err[param_name],
            "t_stat": result.tvalues[param_name],
            "p_value": result.pvalues[param_name],
        })

    standardized_residuals = result.std_resid.dropna()
    lb_std_resid = acorr_ljungbox(standardized_residuals, lags=[10, 20], return_df=True)
    lb_std_resid_squared = acorr_ljungbox(standardized_residuals ** 2, lags=[10, 20], return_df=True)

    for lag, row in lb_std_resid.iterrows():
        garch_22_diagnostics.append({
            "index": index,
            "model": model_name,
            "test": "lb_std_resid",
            "lag": lag,
            "stat": row["lb_stat"],
            "p_value": row["lb_pvalue"],
        })

    for lag, row in lb_std_resid_squared.iterrows():
        garch_22_diagnostics.append({
            "index": index,
            "model": model_name,
            "test": "lb_std_resid_squared",
            "lag": lag,
            "stat": row["lb_stat"],
            "p_value": row["lb_pvalue"],
        })

garch_22_model_results = pd.DataFrame(garch_22_model_results)
garch_22_params = pd.DataFrame(garch_22_params)
garch_22_diagnostics = pd.DataFrame(garch_22_diagnostics)
garch_model_comparison = pd.concat(
    [garch_model_results, garch_22_model_results],
    ignore_index=True,
)

garch_22_model_results.to_csv(DATA_OUTPUTS_PATH / "garch_22_model_results.csv", index=False)
garch_22_params.to_csv(DATA_OUTPUTS_PATH / "garch_22_params.csv", index=False)
garch_22_diagnostics.to_csv(DATA_OUTPUTS_PATH / "garch_22_diagnostics.csv", index=False)
garch_model_comparison.to_csv(DATA_OUTPUTS_PATH / "garch_model_comparison.csv", index=False)

for index in indices:
    index_volatility = garch_volatility.loc[garch_volatility["index"] == index]
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(
        data=index_volatility,
        x="Date",
        y="annualized_conditional_volatility",
        ax=ax,
        label="GARCH conditional volatility",
    )

    sns.lineplot(
        data=index_volatility,
        x="Date",
        y="realized_volatility_21",
        ax=ax,
        label="21-day realized volatility",
    )

    sns.lineplot(
        data=index_volatility,
        x="Date",
        y="realized_volatility_63",
        ax=ax,
        label="63-day realized volatility",
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Volatility (%)")
    ax.set_title(f"GARCH Conditional vs Realized Volatility for {index}")
    format_date_axis(ax)
    plt.savefig(PLOTS_OUTPUTS_PATH / f"{index}_garch_volatility_plot.png")
    plt.close(fig)

persistence_table = garch_params.pivot(
    index="index",
    columns="parameter",
    values="estimate",
).reset_index()

persistence_table = persistence_table[["index", "omega", "alpha[1]", "beta[1]"]]
persistence_table["alpha_beta"] = persistence_table["alpha[1]"] + persistence_table["beta[1]"]
persistence_table = persistence_table.sort_values("alpha_beta", ascending=False)
persistence_table.to_csv(DATA_OUTPUTS_PATH / "garch_persistence_table.csv", index=False)

for index in indices:
    z_t = garch_std_residuals.loc[garch_std_residuals["index"] == index, "standardized_residuals"].dropna()
    z_t_squared = z_t ** 2

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    plot_acf(z_t, lags=20, ax=axes[0, 0])
    axes[0, 0].set_title(f"ACF of Standardized Residuals for {index}")

    plot_pacf(z_t, lags=20, ax=axes[0, 1], method="ywm")
    axes[0, 1].set_title(f"PACF of Standardized Residuals for {index}")

    plot_acf(z_t_squared, lags=20, ax=axes[1, 0])
    axes[1, 0].set_title(f"ACF of Squared Standardized Residuals for {index}")

    plot_pacf(z_t_squared, lags=20, ax=axes[1, 1], method="ywm")
    axes[1, 1].set_title(f"PACF of Squared Standardized Residuals for {index}")

    fig.tight_layout()
    plt.savefig(PLOTS_OUTPUTS_PATH / f"{index}_garch_residual_acf_pacf.png")
    plt.close(fig)

for index in indices:
    historical_volatility = garch_volatility.loc[(garch_volatility["index"] == index) & (garch_volatility["Date"] >= pd.Timestamp(2021, 1, 1) )]

    forecast_volatility = garch_forecast.loc[garch_forecast["index"] == index]

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.lineplot(
        data=historical_volatility,
        x="Date",
        y="annualized_conditional_volatility",
        ax=ax,
        label="Historical conditional volatility",
    )

    sns.lineplot(
        data=forecast_volatility,
        x="Date",
        y="annualized_forecast_volatility",
        ax=ax,
        label=f"{forecast_duration}-day forecast",
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized Volatility (%)")
    ax.set_title(f"GARCH Conditional Volatility Forecast for {index}")
    format_date_axis(ax)

    plt.savefig(PLOTS_OUTPUTS_PATH / f"{index}_volatility_forecast_plot.png")
    plt.close(fig)

var_results = []
var_series_rows = []
alpha = 0.01
train_fraction = 0.8

for index in indices:
    model_name = "AR(1)-GARCH(1,1)-t"
    series = log_returns[index]
    split = int(len(series) * train_fraction)
    model = arch_model(
        series,
        mean="AR",
        lags=1,
        vol="GARCH",
        p=1,
        q=1,
        dist="t",
    )
    result = model.fit(last_obs=split, disp=False)
    forecast = result.forecast(
        horizon=1,
        start=split,
        reindex=True,
        align="target",
    )

    forecast_mean = forecast.mean["h.1"].dropna()
    forecast_variance = forecast.variance["h.1"].dropna()
    forecast_volatility = np.sqrt(forecast_variance)

    common_index = forecast_mean.index.intersection(series.index)

    actual_returns = series.loc[common_index]
    forecast_mean = forecast_mean.loc[common_index]
    forecast_volatility = forecast_volatility.loc[common_index]

    nu = result.params["nu"]
    t_quantile = t.ppf(alpha, df=nu) * np.sqrt((nu - 2) / nu)
    var_99 = forecast_mean + forecast_volatility * t_quantile
    exceedances = actual_returns < var_99
    n = len(exceedances)
    x = exceedances.sum()
    expected = alpha * n
    p_hat = x/n

    #kupiec test
    log_l_null = xlogy(x, alpha) + xlogy(n - x, 1-alpha)
    log_l_alt = xlogy(x, p_hat) + xlogy(n - x, 1-p_hat)
    kupiec_lr = -2 * (log_l_null - log_l_alt)
    kupiec_pvalue = chi2.sf(kupiec_lr, df=1)

    var_results.append({
        "index": index,
        "model": model_name,
        "var_level": "99%",
        "observations": n,
        "exceedances": x,
        "expected_exceedances": expected,
        "exceedance_rate": p_hat,
        "kupiec_lr": kupiec_lr,
        "kupiec_pvalue": kupiec_pvalue,
    })

    for row_index in common_index:
        var_series_rows.append({
            "index": index,
            "Date": log_returns.loc[row_index, "Date"],
            "actual_return": actual_returns.loc[row_index],
            "var_99": var_99.loc[row_index],
            "exceedances": exceedances.loc[row_index],
        })

var_results = pd.DataFrame(var_results)
var_series = pd.DataFrame(var_series_rows)
var_results.to_csv(DATA_OUTPUTS_PATH / "var_results.csv", index=False)
var_series.to_csv(DATA_OUTPUTS_PATH / "var_series.csv", index=False)

for index in indices:
    series = var_series[var_series["index"]== index]
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(
        data = series,
        x="Date",
        y="actual_return",
        ax=ax,
        label = "actual return"
    )
    sns.lineplot(
        data = series,
        x="Date",
        y="var_99",
        ax=ax,
        color = "crimson",
        label = "99% VaR"
    )
    exceedance_days = series[series["exceedances"] == True]
    sns.scatterplot(
        data=exceedance_days,
        x="Date",
        y="actual_return",
        ax=ax,
        color="black",
        label="VaR exceedance",
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Return / VaR")
    ax.set_title(f"99% VaR Backtest for {index}")
    format_date_axis(ax)
    plt.savefig(PLOTS_OUTPUTS_PATH / f"{index}_var_backtest_plot.png")
    plt.close(fig)
