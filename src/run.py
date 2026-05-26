from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.rrule import MO
from IPython.display import display
from jupyter_server_terminals.terminalmanager import MODEL

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


df_weather = pd.read_parquet(DATA_DIR / "weather_train.parquet")
df_network = pd.read_parquet(DATA_DIR / "network_train.parquet")
df = pd.read_parquet(DATA_DIR / "target_train.parquet")


def add_calendar_features(df, year=True, week=True, day=True):

    if year:
        doy = df.index.dayofyear

        df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    if week:
        dow = df.index.dayofweek

        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    if day:
        hour = df.index.hour

        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    return df


df = add_calendar_features(df)

# Add mean weather features
df["temp_avg"] = df_weather[[col for col in df_weather.columns if col[2] == "2t"]].mean(
    axis=1
)

df["wind_avg"] = (
    df_weather[[col for col in df_weather.columns if col[2] == "100ws"]]
    .mean(axis=1)
    .interpolate()
    .bfill()
)

df["sun_avg"] = (
    df_weather[[col for col in df_weather.columns if col[2] == "ssrd"]]
    .mean(axis=1)
    .interpolate()
    .bfill()
)


from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def forecast_model(df, FEATURES, TARGET, MODEL):

    # Chronological train/validation split
    train = df.loc[df.index < "2024-01-01"].copy()
    valid = df.loc[df.index >= "2024-01-01"].copy()

    train = train.dropna(subset=[TARGET])
    valid = valid.dropna(subset=[TARGET])

    if MODEL == "RandomForestRegressor":
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )

    elif MODEL == "HistGradientBoostingRegressor":
        model = HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=42,
        )

    elif MODEL == "LGBMRegressor":
        model = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
        )

    model.fit(train[FEATURES], train[TARGET])

    # Predict
    train["pred"] = model.predict(train[FEATURES])
    valid["pred"] = model.predict(valid[FEATURES])

    # Evaluate on validation set
    mae = mean_absolute_error(valid[TARGET], valid["pred"])
    rmse = mean_squared_error(valid[TARGET], valid["pred"]) ** 0.5

    print(f"MAE:  {mae:,.2f} MW")
    print(f"RMSE: {rmse:,.2f} MW")

    return train, valid, model


# Starting with τοταλ energy load
TARGET = "FR_load_actual"

FEATURES = [
    "temp_avg",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
]


train, valid, model = forecast_model(
    df, FEATURES, TARGET, MODEL="RandomForestRegressor"
)


# Plot one month
valid.loc["2024-05", [TARGET, "pred"]].plot(figsize=(15, 5))
plt.ylabel("MW")
plt.show()


# Plot whole validation year
valid[[TARGET, "pred"]].plot(figsize=(15, 5), alpha=0.8)
plt.ylabel("MW")
plt.show()


# Feature importance
# Feature importance
feature_importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(
    ascending=False
)

plt.figure(figsize=(8, 4))
plt.bar(feature_importance.index, feature_importance.values)

plt.xticks(rotation=90)
plt.title("Feature Importance")
plt.tight_layout()

plt.show()
