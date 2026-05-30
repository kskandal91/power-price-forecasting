# power-price-forecasting

A machine learning pipeline for forecasting French electricity load, solar generation, wind generation, and spot electricity prices using weather, calendar, and power network data.

## Project Overview

This project was developed as part of the Kpler Gas & Power Data Science recruitment exercise. The objective is to generate hourly forecasts for 2025 for the following targets:

- Electricity load (`FR_load_actual`)
- Solar generation (`FR_solar_actual`)
- Wind generation (`FR_wind_actual`)
- Electricity prices (`FR_price_actual`)

The modelling approach follows a sequential forecasting pipeline:

1. Forecast electricity load from weather and calendar features.
2. Forecast solar generation from weather and calendar features.
3. Forecast wind generation from weather and calendar features.
4. Forecast electricity prices using weather, calendar, network features, and the predicted values of load, solar, and wind.

Different model types were selected for each target based on validation performance:

| Target | Model |
|----------|----------|
| Load | Random Forest |
| Solar | LightGBM |
| Wind | LightGBM |
| Price | XGBoost |

The validation strategy uses historical data from 2020–2023 for training and the 2024 calendar year for out-of-sample evaluation.

## Repository Structure

```text
.
├── data/
├── notebooks/
│   ├── report.ipynb
│   ├── report.html
│   └── report.pdf
├── outputs/
│   ├── metrics.csv
│   └── predictions_2025.parquet
├── src/
│   ├── __init__.py
│   └── functions.py
├── run.py
├── requirements.txt
└── README.md
```

## Contents

- `data/` contains the datasets provided for the forecasting exercise.
- `notebooks/report.pdf` contains the final report and methodology discussion.
- `notebooks/report.html` provides a browser-friendly version of the report.
- `notebooks/report.ipynb` contains the notebook used to generate the report.
- `src/functions.py` contains the feature engineering, model training, and prediction utilities.
- `run.py` contains the end-to-end forecasting workflow.
- `outputs/` contains the generated forecasts and validation metrics.

## Installation

```bash
pip install -r requirements.txt
```

## Running the Pipeline

```bash
python run.py
```

Running the pipeline generates:

```text
outputs/predictions_2025.parquet
outputs/metrics.csv
```

## Report

The main report can be found in:

```text
notebooks/report.pdf
```

Alternative formats are also provided:

```text
notebooks/report.html
notebooks/report.ipynb
```

## Notes

Model selection and hyperparameter tuning were performed using the 2024 period as a validation set. Consequently, the reported validation metrics should be interpreted as indicative rather than fully unbiased estimates of future out-of-sample performance.

## Development Process

Exploratory data analysis (EDA), feature engineering, feature selection, and model experimentation were performed during development. To keep the submission concise and focused on reproducibility, only the final forecasting pipeline, report, and outputs are included in this repository.
