### 📊 Project Overview

An end-to-end time series forecasting pipeline to predict monthly retail sales using historical transaction data from a global Superstore. The project compares classical statistical models to isolate trends and seasonal patterns for better business planning.

### 🛠️ Key Work Done

* **Preprocessing:** Cleaned and aggregated daily transaction logs into a structured monthly time series (`freq='M'`).
* **Stationarity Handling:** Used an **Augmented Dickey-Fuller (ADF) test** to verify non-stationarity, then applied a **Box-Cox transformation** ($\lambda = 0$) and **first-order differencing** to stabilize variance and remove trends.
* **Validation Split:** Evaluated models using a strict Train/Test split (first 42 months for training, final 6 months for out-of-sample testing).
* **Modeling:** Built and tuned **AR(1)**, **ARIMA(1,0,1)**, and **SARIMA(1,0,1) × (1,0,1)₁₂** models using `statsmodels`.
  
### 📈 Model Evaluation (Test RMSE)

* **SARIMA:** **11,661.64** 🏆 *(Best Model)*
* **AR:** 14,915.16
* **ARIMA:** 29,842.05

### 🎯 Significance

* **Captures Retail Cyclicality:** The top performance of the SARIMA model proves that accounting for a 12-month annual seasonality is crucial for retail forecasting.
* **Business Impact:** Provides a data-driven framework to optimize inventory levels, plan supply chain logistics, and mitigate risks of stockouts or over-stocking during peak seasons.
