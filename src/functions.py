from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"


# =========================================================
# Feature definitions
# =========================================================


LOAD_FEATURES = [
    "temp_avg",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
]

SOLAR_FEATURES = [
    "FR_capacity_solar",
    "sun_avg",
    "tcc_avg",
    "effective_solar_avg",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]

WIND_FEATURES = [
    "FR_capacity_wind",
    "wind_avg",
    "doy_sin",
    "doy_cos",
]

PRICE_FEATURES = [
    "FR_load_input",
    "FR_solar_input",
    "FR_wind_input",
    "doy_sin",
    "doy_cos",
    "temp_avg",
    "EEX_CARBON",
    "EEX_COAL",
    "EEX_GAS_PEG",
    "FR_availability_coal",
    "FR_availability_gas",
    "FR_availability_hydro",
    "FR_availability_nuclear",
]


NETWORK_COLS = [
    "EEX_CARBON",
    "EEX_COAL",
    "EEX_GAS_PEG",
    "FR_availability_coal",
    "FR_availability_gas",
    "FR_availability_hydro",
    "FR_availability_nuclear",
    "FR_capacity_solar",
    "FR_capacity_wind",
]


TARGET_CONFIG = {
    "load": {
        "target": "FR_load_actual",
        "features": LOAD_FEATURES,
        "model_type": "random_forest",
    },
    "solar": {
        "target": "FR_solar_actual",
        "features": SOLAR_FEATURES,
        "model_type": "lightgbm1",
    },
    "wind": {
        "target": "FR_wind_actual",
        "features": WIND_FEATURES,
        "model_type": "lightgbm2",
    },
    "price": {
        "target": "FR_price_actual",
        "features": PRICE_FEATURES,
        "model_type": "xgboost",
    },
}


# =========================================================
# Data loading
# =========================================================


def load_train_data():

    target_train = pd.read_parquet(DATA_DIR / "target_train.parquet")

    weather_train = pd.read_parquet(DATA_DIR / "weather_train.parquet")

    network_train = pd.read_parquet(DATA_DIR / "network_train.parquet")

    return (
        target_train,
        weather_train,
        network_train,
    )


def load_test_data():

    weather_test = pd.read_parquet(DATA_DIR / "weather_test.parquet")

    network_test = pd.read_parquet(DATA_DIR / "network_test.parquet")

    return (
        weather_test,
        network_test,
    )


# =========================================================
# Feature engineering
# =========================================================


def add_calendar_features(df):

    df = df.copy()

    doy = df.index.dayofyear
    dow = df.index.dayofweek
    hour = df.index.hour

    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    return df


def replace_long_zero_runs(series, threshold=300):
    series = series.copy()

    is_zero = series.eq(0)

    # Create a new group whenever zero/non-zero status changes
    groups = is_zero.ne(is_zero.shift()).cumsum()

    # Count length of each consecutive block
    run_lengths = is_zero.groupby(groups).transform("size")

    # Replace only zero blocks longer than threshold
    bad_runs = is_zero & (run_lengths >= threshold)

    return series.mask(bad_runs)


def add_avg_weather_features(df, weather):

    df = df.copy()

    weather_vars = {
        "temp": "2t",
        "wind": "100ws",
        "sun": "ssrd",
        "tcc": "tcc",
    }

    clean_weather = {}

    for name, variable in weather_vars.items():
        cols = [c for c in weather.columns if c[2] == variable]

        data = weather[cols].apply(replace_long_zero_runs, threshold=5000)

        clean_weather[name] = data

        df[f"{name}_avg"] = data.mean(axis=1)

    common_tile_ids = sorted(
        set(c[1] for c in clean_weather["sun"].columns)
        & set(c[1] for c in clean_weather["tcc"].columns)
    )

    effective_solar_by_tile = []

    for tile_id in common_tile_ids:
        sun_col = ("FR", tile_id, "ssrd")
        tcc_col = ("FR", tile_id, "tcc")

        effective_solar_by_tile.append(
            clean_weather["sun"][sun_col] * (1 - clean_weather["tcc"][tcc_col])
        )

    df["effective_solar_avg"] = pd.concat(effective_solar_by_tile, axis=1).mean(axis=1)

    return df


def add_network_features(df, network):

    df = df.copy()

    for col in NETWORK_COLS:
        df[col] = network[col]

    return df


# =========================================================
# Cleaning
# =========================================================


def interpolate_spurious_values(series, ratio_limit=3, gap=1):
    series = series.copy()

    prev_val = series.shift(gap)
    next_val = series.shift(-gap)

    spikes = (series > ratio_limit * prev_val) & (series > ratio_limit * next_val)

    clean = series.copy()
    clean.loc[spikes] = np.nan

    return clean.interpolate().bfill().ffill()


def clean_targets(df):

    df = df.copy()

    df["FR_solar_actual"] = interpolate_spurious_values(df["FR_solar_actual"])

    df["FR_wind_actual"] = interpolate_spurious_values(df["FR_wind_actual"])

    # Replace long runs of zeros in solar with NaN
    df["FR_solar_actual"] = replace_long_zero_runs(df["FR_solar_actual"], threshold=300)

    # Ad hoc cap of apparent price outliers in April 2022.
    # This is a dataset-specific fix and ideally should be replaced
    #   by a more general outlier detection approach.
    df.loc[
        df["FR_price_actual"].gt(500) & (df.index.to_period("M") == "2022-04"),
        "FR_price_actual",
    ] = 500

    return df


# =========================================================
# Build datasets
# =========================================================


def build_train_df(target_train, weather_train, network_train):

    df = target_train.copy()

    df = add_calendar_features(df)

    df = clean_targets(df)

    df = add_avg_weather_features(df, weather_train)

    df = add_network_features(df, network_train)

    return df


def build_test_df(weather_test, network_test):

    df = pd.DataFrame(index=weather_test.index)

    df = add_calendar_features(df)

    df = add_avg_weather_features(df, weather_test)

    df = add_network_features(df, network_test)

    return df


# =========================================================
# Model
# =========================================================


def make_model(model_type="random_forest"):

    if model_type == "lightgbm1":
        return LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            reg_alpha=0.7,
            reg_lambda=0.7,
            random_state=42,
            n_jobs=-1,
        )

    if model_type == "lightgbm2":
        return LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            reg_alpha=0.1,
            reg_lambda=0.2,
            random_state=42,
            n_jobs=-1,
        )

    if model_type == "lightgbm3":
        return LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
        )

    if model_type == "xgboost":
        return XGBRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
        )

    return RandomForestRegressor(
        n_estimators=300,
        max_depth=30,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )


def train_single_model(
    df,
    target,
    features,
    model_type="random_forest",
    return_train=False,
):
    cols = features + [target]

    train = df.loc[df.index < "2024-01-01", cols].dropna().copy()
    valid = df.loc[df.index >= "2024-01-01", cols].dropna().copy()

    if target == "FR_solar_actual":
        train = train.loc[train["sun_avg"].gt(100)]

    model = make_model(model_type)

    model.fit(train[features], train[target])

    valid["pred"] = model.predict(valid[features])

    if target == "FR_solar_actual":
        conservative_night = get_conservative_night_mask(valid.index)
        valid.loc[conservative_night, "pred"] = 0
        valid["pred"] = valid["pred"].clip(lower=0)

    mae = mean_absolute_error(valid[target], valid["pred"])
    rmse = mean_squared_error(valid[target], valid["pred"]) ** 0.5

    metrics = {
        "target": target,
        "mae": mae,
        "rmse": rmse,
    }

    print(f"{target}")
    print(f"MAE:  {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print()

    if return_train:
        train["pred"] = model.predict(train[features])

        if target == "FR_solar_actual":
            train["pred"] = train["pred"].clip(lower=0)

        return model, metrics, valid, train

    return model, metrics, valid


# =========================================================
# Training pipeline
# =========================================================


def get_conservative_night_mask(index):

    winter = (
        ((index.month == 9) & (index.day >= 21))
        | (index.month.isin([10, 11, 12, 1, 2]))
        | ((index.month == 3) & (index.day <= 21))
    )

    night_mask = np.where(
        winter,
        (index.hour <= 5) | (index.hour >= 21),
        (index.hour <= 4) | (index.hour >= 22),
    )

    return night_mask


def train_models(df, mode="validation"):
    df = df.copy()

    models = {}
    metrics = []

    if mode == "validation":
        train_mask = df.index < "2024-01-01"

        for name in ["load", "solar", "wind"]:
            config = TARGET_CONFIG[name]

            model, model_metrics, valid = train_single_model(
                df,
                target=config["target"],
                features=config["features"],
                model_type=config["model_type"],
            )

            models[name] = model
            metrics.append(model_metrics)

            input_col = f"FR_{name}_input"

            df[input_col] = np.nan
            df.loc[train_mask, input_col] = df.loc[train_mask, config["target"]]
            df.loc[valid.index, input_col] = valid["pred"]

        price_config = TARGET_CONFIG["price"]

        price_model, price_metrics, _ = train_single_model(
            df,
            target=price_config["target"],
            features=price_config["features"],
            model_type=price_config["model_type"],
        )

        models["price"] = price_model
        metrics.append(price_metrics)

        return models, metrics

    if mode == "final":
        df["FR_load_input"] = df["FR_load_actual"]
        df["FR_solar_input"] = df["FR_solar_actual"]
        df["FR_wind_input"] = df["FR_wind_actual"]

        for name, config in TARGET_CONFIG.items():
            train = df[config["features"] + [config["target"]]].dropna().copy()

            if name == "solar":
                train = train.loc[train["sun_avg"].gt(100)]

            model = make_model(config["model_type"])
            model.fit(train[config["features"]], train[config["target"]])

            models[name] = model

        return models, metrics

    raise ValueError("mode must be either 'validation' or 'final'")


# =========================================================
# Prediction pipeline
# =========================================================


def predict_2025(models, test_df):

    test_df = test_df.copy()

    predictions = pd.DataFrame(index=test_df.index)

    for name in ["load", "solar", "wind"]:
        config = TARGET_CONFIG[name]

        pred_col = f"FR_{name}_pred"
        input_col = f"FR_{name}_input"

        pred = models[name].predict(test_df[config["features"]])

        if name == "solar":
            conservative_night = get_conservative_night_mask(test_df.index)
            pred[conservative_night] = 0
            pred = np.clip(pred, 0, None)

        predictions[pred_col] = pred
        test_df[input_col] = pred

    predictions["FR_price_pred"] = models["price"].predict(
        test_df[TARGET_CONFIG["price"]["features"]]
    )

    return predictions


# =========================================================
# Full pipeline
# =========================================================


def run_pipeline():
    OUTPUT_DIR.mkdir(exist_ok=True)

    target_train, weather_train, network_train = load_train_data()
    weather_test, network_test = load_test_data()

    train_df = build_train_df(
        target_train,
        weather_train,
        network_train,
    )

    test_df = build_test_df(
        weather_test,
        network_test,
    )

    _, metrics = train_models(
        train_df,
        mode="validation",
    )

    final_models, _ = train_models(
        train_df,
        mode="final",
    )

    predictions = predict_2025(
        final_models,
        test_df,
    )

    predictions.to_parquet(OUTPUT_DIR / "predictions_2025.parquet")

    metrics_df = pd.DataFrame(metrics)

    metrics_df.to_csv(
        OUTPUT_DIR / "metrics.csv",
        index=False,
    )

    print("Saved predictions")
    print("Saved metrics")

    return final_models, metrics_df, predictions
