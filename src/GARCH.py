from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from arch import arch_model
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = ROOT / "data" / "processed"
DATA_OUTPUTS_PATH = ROOT / "outputs" / "data"
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
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