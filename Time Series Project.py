import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", ConvergenceWarning)

data = pd.read_csv("Superstore_Data.csv")

data.head()

data.dtypes

data["Order Date"] = pd.to_datetime(data["Order Date"])
data.dtypes

data.head()

data.sort_values(by="Order Date", inplace=True)
data.head()

data.reset_index(drop=True, inplace=True)
data.head()

data.set_index("Order Date", inplace=True)
data.head()

data.loc["2011-01-03"]
data.loc["2011-01-03":"2012-11-10"].sum()

data.shape

data = data.groupby(pd.Grouper(freq="M")).sum()
data.shape
data.head()

data.drop(columns="Profit", inplace=True)
data.head()

data.plot()
plt.show()

# ADF Test
from statsmodels.tsa.stattools import adfuller

results = adfuller(data["Sales"])
print(f"ADF Statistic: {results[1]}")

df_train = data[0:42]
df_test = data[42:]

# Plot the time series data with the train-test split
plt.figure(figsize=(14, 6))
sns.lineplot(
    data=df_train, x="Order Date", y="Sales", marker="o", color="blue", label="Train"
)
sns.lineplot(
    data=df_test, x="Order Date", y="Sales", marker="o", color="green", label="Test"
)
plt.title("Sales Data")

from scipy.stats import boxcox

df_boxcox = pd.Series(boxcox(df_train["Sales"], lmbda=0), index=df_train.index)

# Plot the original time series data and its Box-Cox transformed version
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.lineplot(data=df_train, x="Order Date", y="Sales", marker="o", color="blue")
plt.xticks(rotation=90)
plt.title("Original Data")

plt.subplot(1, 2, 2)
sns.lineplot(x=df_boxcox.index, y=df_boxcox.values, marker="o", color="blue")
plt.xticks(rotation=90)
plt.title("Box-Cox Transformed Data [lambda = 0]")

plt.suptitle("Sales Data")


df_differenced = df_train["Sales"].diff()

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.lineplot(data=df_train, x="Order Date", y="Sales", marker="o", color="blue")
plt.xticks(rotation=90)
plt.title("Original Data")

plt.subplot(1, 2, 2)
sns.lineplot(x=df_differenced.index, y=df_differenced.values, marker="o", color="blue")
plt.xticks(rotation=90)
plt.title("Differenced Data [Order = 1]")
plt.suptitle("Sales Data")

df_boxcox = pd.Series(boxcox(df_train["Sales"], lmbda=0), index=df_train.index)
df_boxcox_diff = df_boxcox.diff()

# Plot the original time series data and its transformed version
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.lineplot(data=df_train, x="Order Date", y="Sales", marker="o", color="blue")
plt.xticks(rotation=90)
plt.title("Original Data")

plt.subplot(1, 2, 2)
sns.lineplot(x=df_boxcox_diff.index, y=df_boxcox_diff.values, marker="o", color="blue")
plt.xticks(rotation=90)
plt.title("Transformed Data [lambda = 0 and differencing order = 1]")
plt.suptitle("Sales Data")

result_after_transformation = adfuller(df_boxcox_diff[1:])
result_after_transformation[1]

from statsmodels.graphics.tsaplots import plot_acf

plot_acf(df_boxcox_diff[1:])

from statsmodels.tsa.arima.model import ARIMA

ar_model = ARIMA(df_boxcox_diff, order=(1, 0, 0))
ar_model = ar_model.fit()

train_len = 42
ar_model_preds = ar_model.predict(start=train_len, end=len(data) - 1)

ar_model_preds

df_boxcox_diff_preds = pd.concat([df_boxcox_diff, ar_model_preds])

df_boxcox_preds = df_boxcox_diff_preds.cumsum()  # cummulative series
df_boxcox_preds = df_boxcox_preds.add(df_boxcox[0])  # initial value adjustment

df_preds = np.exp(df_boxcox_preds)

plt.figure(figsize=(14, 6))
sns.lineplot(
    data=df_train, x="Order Date", y="Sales", marker="o", color="blue", label="Train"
)
sns.lineplot(
    data=df_test, x="Order Date", y="Sales", marker="o", color="green", label="Test"
)
sns.lineplot(
    x=df_preds.index[train_len:],
    y=df_preds.values[train_len:],
    marker="o",
    color="purple",
    label="Predictions",
)
plt.title("Sales Data")

from sklearn.metrics import mean_squared_error


rmse = np.sqrt(
    mean_squared_error(y_true=df_test["Sales"], y_pred=df_preds.values[train_len:])
)

rmse = np.round(rmse, 2)

performance_df = pd.DataFrame(index=[0], data={"Model": "AR", "RMSE": rmse})

performance_df.set_index("Model", inplace=True)

performance_df


arima_model = ARIMA(df_boxcox, order=(1, 0, 1))
arima_model = arima_model.fit()

train_len = 42
arima_model_preds = arima_model.predict(start=train_len, end=len(data) - 1)

df_boxcox_preds = pd.concat([df_boxcox, arima_model_preds])

df_preds = np.exp(df_boxcox_preds)

plt.figure(figsize=(14, 6))
sns.lineplot(
    data=df_train, x="Order Date", y="Sales", marker="o", color="blue", label="Train"
)
sns.lineplot(
    data=df_test, x="Order Date", y="Sales", marker="o", color="green", label="Test"
)
sns.lineplot(
    x=df_preds.index[train_len:],
    y=df_preds.values[train_len:],
    marker="o",
    color="purple",
    label="Predictions",
)
plt.title("Sales Data")

rmse = np.sqrt(
    mean_squared_error(y_true=df_test["Sales"], y_pred=df_preds.values[train_len:])
)

rmse = np.round(rmse, 2)

performance_df_temp = pd.DataFrame(index=[0], data={"Model": "ARIMA", "RMSE": rmse})

performance_df_temp.set_index("Model", inplace=True)

performance_df = pd.concat([performance_df, performance_df_temp])

performance_df

from statsmodels.tsa.statespace.sarimax import SARIMAX

sarima_model = SARIMAX(df_boxcox, order=(1, 0, 1), seasonal_order=(1, 0, 1, 12))
sarima_model = sarima_model.fit()

train_len = 42
sarima_model_preds = sarima_model.predict(start=train_len, end=len(data) - 1)

df_boxcox_preds = pd.concat([df_boxcox, sarima_model_preds])

df_preds = np.exp(df_boxcox_preds)

plt.figure(figsize=(14, 6))
sns.lineplot(
    data=df_train, x="Order Date", y="Sales", marker="o", color="blue", label="Train"
)
sns.lineplot(
    data=df_test, x="Order Date", y="Sales", marker="o", color="green", label="Test"
)
sns.lineplot(
    x=df_preds.index[train_len:],
    y=df_preds.values[train_len:],
    marker="o",
    color="purple",
    label="Predictions",
)
plt.title("Sales Data")

rmse = np.sqrt(
    mean_squared_error(y_true=df_test["Sales"], y_pred=df_preds.values[train_len:])
)

rmse = np.round(rmse, 2)

performance_df_temp = pd.DataFrame(index=[0], data={"Model": "SARIMA", "RMSE": rmse})

performance_df_temp.set_index("Model", inplace=True)

performance_df = pd.concat([performance_df, performance_df_temp])

performance_df
