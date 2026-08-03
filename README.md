# Weather Temperature Regression

Predict ambient temperature (`temperature_c`) from the Weather History dataset using **feature engineering** and a comparison of linear models:

- Linear Regression
- Ridge / RidgeCV
- Lasso / LassoCV
- ElasticNet / ElasticNetCV

This project focuses less on “which fancy model” and more on **making features the model can actually use**.

## Dataset

File: [`data/weatherHistory.csv`](data/weatherHistory.csv) (~16MB)

Public Weather History data (Szeged / Kaggle “Weather History”).

| Target | Notes |
|--------|--------|
| `temperature_c` | Regression target |
| `apparent_temperature_c` | **Dropped from features** (near-duplicate of target → leakage risk) |

## Feature engineering highlights

1. **Column cleanup** — normalize names (`Temperature (C)` → `temperature_c`)
2. **Datetime parts** — `year`, then cyclical encodings for month / day-of-week
3. **Cyclical encoding (`sin` / `cos`)**
   - month, day of week, wind bearing  
   - so Dec↔Jan and 359°↔1° stay neighbors
4. **Precipitation** — `is_snow` from `precip_type`
5. **Text → structured features** from `summary`
   - ordinal `cloud_level` (clear → overcast)
   - flags: `is_clear`, `is_foggy`, `is_breezy`, `is_windy`, `is_dry`, `is_humid`
   - ordinal `rain_level` (drizzle / light rain / rain)
6. Drop raw text / unused columns (`summary`, `daily_summary`, `loud_cover`, …)

Rule-based FE is applied before the split (same fixed rules for every row).  
**Learned** transforms (`StandardScaler`) are fit on train only via `Pipeline`.

## Method

1. Load + clean + engineer features  
2. `train_test_split(test_size=0.2, random_state=15)`  
3. For each model: `Pipeline(StandardScaler → model)`  
4. Report **MAE**, **MSE**, **R²** on the test set  

## Setup

```bash
cd weather-temperature-regression
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python train_compare.py
```

## Typical results (indicative)

On this pipeline, Linear / Ridge / *CV models usually land around:

- **MAE ≈ 2.8°C**
- **R² ≈ 0.86**

Regularization often ties with plain Linear here — feature engineering carries most of the signal.

## Project structure

```text
weather-temperature-regression/
├── README.md
├── requirements.txt
├── .gitignore
├── train_compare.py
└── data/
    └── weatherHistory.csv
```

## Related

Also see: [iris-classifier-comparison](https://github.com/mkayaaslan/iris-classifier-comparison) (classification model comparison).
