# power-price-forecasting

A machine learning pipeline for forecasting electricity load, solar generation, wind generation, and power prices using weather, calendar, and network features.

## Project Overview

This project was developed as part of a data science forecasting assessment. The objective is to generate hourly forecasts for:

* Electricity load
* Solar generation
* Wind generation
* Electricity prices

The modelling approach follows a sequential forecasting pipeline:

1. Forecast load from weather and calendar features.
2. Forecast solar generation from weather and calendar features.
3. Forecast wind generation from weather and calendar features.
4. Forecast electricity prices using weather, calendar, network features, and the predicted values of load, solar, and wind.

The validation strategy uses historical data from 2020–2023 for training and the 2024 calendar year for out-of-sample evaluation.

## Repository Structure

```text
.
├── data/
├── notebooks/
│   └── report.ipynb
├── outputs/
├── src/
│   ├── __init__.py
│   └── functions.py
├── run.py
├── requirements.txt
└── README.md
```

## Contents

* `data/` contains the input datasets provided for the forecasting exercise.
* `notebooks/report.ipynb` contains the final report and discussion of the modelling approach and results.
* `src/functions.py` contains the utility functions used throughout the forecasting pipeline.
* `run.py` contains the main forecasting workflow used to generate the submitted forecasts.
* `outputs/` contains the generated forecast outputs.

Exploratory data analysis, feature engineering, and model experimentation were performed during model development but are not included in the final report.

## Installation

```bash
pip install -r requirements.txt
```

## Running the Pipeline

```bash
python run.py
```
