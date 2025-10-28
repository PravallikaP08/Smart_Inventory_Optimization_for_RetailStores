import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import pickle
import warnings
warnings.filterwarnings("ignore")

# ================================
# 📂 1. Load your dataset
# ================================
file_path = r"C:\Users\PRAVALLIKA\Smart_Stock_Project\MileStone1\cleaned_retail_dataset_single_store_2020_2025.csv"
df = pd.read_csv(file_path)

# Adjust column names to match your file
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# Expecting columns like: product_id, date, units_sold
date_col = "date"
product_col = "product_id"
sales_col = "units_sold"

df[date_col] = pd.to_datetime(df[date_col])
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ================================
# ⚙️ 2. Helper Functions
# ================================

def train_lstm(series, n_lags=7, epochs=10):
    """Train LSTM on a 1D sales time series."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(scaled) - n_lags):
        X.append(scaled[i:i+n_lags, 0])
        y.append(scaled[i+n_lags, 0])
    if len(X) == 0:
        return None, None

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, activation="relu", input_shape=(n_lags, 1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, verbose=0)
    return model, scaler


def forecast_lstm(model, scaler, series, steps=30, n_lags=7):
    """Forecast future values using trained LSTM."""
    if model is None or scaler is None:
        return np.array([])

    data = scaler.transform(series.values.reshape(-1, 1)).flatten().tolist()
    preds = []
    for _ in range(steps):
        x_input = np.array(data[-n_lags:]).reshape((1, n_lags, 1))
        yhat = model.predict(x_input, verbose=0)
        data.append(yhat[0][0])
        preds.append(yhat[0][0])
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds


# ================================
# 🚀 3. Train Prophet + LSTM
# ================================
forecast_list = []
all_products = df[product_col].unique()[:20]  # 🔹 Limit to first 20 products only

for product_name in all_products:
    print(f"\n🔄 Training Prophet & LSTM for product {product_name}...")

    product_df = df[df[product_col] == product_name][[date_col, sales_col]]
    product_df = product_df.groupby(date_col).sum().reset_index()

    # ✅ Skip if too few data points
    if product_df[sales_col].count() < 2 or product_df[sales_col].sum() == 0:
        print(f"⚠️ Skipping {product_name} — insufficient or zero sales data")
        continue

    # ----- Prophet -----
    prophet_df = product_df.rename(columns={date_col: "ds", sales_col: "y"})
    model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model_prophet.fit(prophet_df)

    future = model_prophet.make_future_dataframe(periods=30)
    forecast_p = model_prophet.predict(future)
    yhat_prophet = forecast_p["yhat"][-30:]

    # ----- LSTM -----
    sales_series = product_df.set_index(date_col)[sales_col]
    if len(sales_series) < 10:
        print(f"⚠️ Skipping {product_name} — not enough time points for LSTM")
        continue

    lstm_model, scaler = train_lstm(sales_series)
    yhat_lstm = forecast_lstm(lstm_model, scaler, sales_series, steps=30)

    # ----- Metrics -----
    actual = sales_series[-30:] if len(sales_series) >= 30 else sales_series
    mae_prophet = mean_absolute_error(actual, yhat_prophet[:len(actual)])
    rmse_prophet = np.sqrt(mean_squared_error(actual, yhat_prophet[:len(actual)]))
    mae_lstm = mean_absolute_error(actual, yhat_lstm[:len(actual)])
    rmse_lstm = np.sqrt(mean_squared_error(actual, yhat_lstm[:len(actual)]))

    print(f"📈 Prophet → MAE: {mae_prophet:.2f}, RMSE: {rmse_prophet:.2f}")
    print(f"🤖 LSTM    → MAE: {mae_lstm:.2f}, RMSE: {rmse_lstm:.2f}")

    # ----- Save Best Model -----
    if rmse_lstm < rmse_prophet:
        print("✅ LSTM performed better — saving LSTM model & forecast")
        best_forecast = yhat_lstm
        with open(f"models/lstm_model_{product_name}.pkl", "wb") as f:
            pickle.dump({"scaler_min": scaler.data_min_, "scaler_max": scaler.data_max_}, f)
    else:
        print("✅ Prophet performed better — saving Prophet model & forecast")
        best_forecast = yhat_prophet
        with open(f"models/prophet_model_{product_name}.pkl", "wb") as f:
            pickle.dump(model_prophet, f)

    # ----- Save Forecast -----
    forecast_dates = pd.date_range(
        start=product_df[date_col].max() + pd.Timedelta(days=1),
        periods=30
    )
    temp = pd.DataFrame({
        "date": forecast_dates,
        "forecast_best": best_forecast,
        "product_id": product_name
    })
    forecast_list.append(temp)

    # ----- Plot Forecast -----
    plt.figure(figsize=(10, 5))
    plt.plot(product_df[date_col], product_df[sales_col], label="Actual", color="black")
    plt.plot(forecast_dates, best_forecast, label="Forecast", color="blue")
    plt.title(f"Sales Forecast — Product {product_name}")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"data/forecast_{product_name}.png")
    plt.close()


# ================================
# 💾 4. Save All Results
# ================================
if forecast_list:
    forecast_all = pd.concat(forecast_list)
    forecast_all.to_csv("data/forecast_results.csv", index=False)
    print("\n✅ Forecast results saved in data/forecast_results.csv")
else:
    print("\n⚠️ No valid products found for forecasting.")

print("Milestone 2 completed successfully 🎉")
