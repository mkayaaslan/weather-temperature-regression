"""
Weather Temperature Regression
Feature engineering + comparison of linear models on Weather History data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import (
    ElasticNet,
    ElasticNetCV,
    Lasso,
    LassoCV,
    LinearRegression,
    Ridge,
    RidgeCV,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent / "data" / "weatherHistory.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 15


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[()]", "", regex=True)
        .str.replace("/", "_per_", regex=False)
    )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rule-based feature engineering (no train-only learning here)."""
    df = df.copy()

    df["formatted_date"] = pd.to_datetime(df["formatted_date"], utc=True)
    df["month"] = df["formatted_date"].dt.month
    df["year"] = df["formatted_date"].dt.year
    df["day_of_week"] = df["formatted_date"].dt.dayofweek

    # Cyclical encoding: Dec/Jan and similar wrap-around neighbors stay close
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["wind_bearing_sin"] = np.sin(2 * np.pi * df["wind_bearing_degrees"] / 360)
    df["wind_bearing_cos"] = np.cos(2 * np.pi * df["wind_bearing_degrees"] / 360)

    df["is_snow"] = (df["precip_type"] == "snow").astype(int)

    df["cloud_level"] = 0
    df.loc[
        df["summary"].str.contains("partly cloudy", case=False, na=False),
        "cloud_level",
    ] = 1
    df.loc[
        df["summary"].str.contains("mostly cloudy", case=False, na=False),
        "cloud_level",
    ] = 2
    df.loc[
        df["summary"].str.contains("overcast", case=False, na=False),
        "cloud_level",
    ] = 3

    df["is_clear"] = df["summary"].str.contains("clear", case=False, na=False).astype(int)
    df["is_foggy"] = df["summary"].str.contains("foggy", case=False, na=False).astype(int)
    df["is_breezy"] = df["summary"].str.contains("breezy", case=False, na=False).astype(int)
    df["is_windy"] = df["summary"].str.contains("windy", case=False, na=False).astype(int)
    df["is_dry"] = df["summary"].str.contains("dry", case=False, na=False).astype(int)
    df["is_humid"] = df["summary"].str.contains("humid", case=False, na=False).astype(int)

    df["rain_level"] = 0
    df.loc[
        df["summary"].str.contains("drizzle", case=False, na=False),
        "rain_level",
    ] = 1
    df.loc[
        df["summary"].str.contains("light rain", case=False, na=False),
        "rain_level",
    ] = 2
    df.loc[
        df["summary"].str.contains("rain", case=False, na=False)
        & ~df["summary"].str.contains("light rain", case=False, na=False),
        "rain_level",
    ] = 3

    df = df.drop(
        columns=[
            "formatted_date",
            "month",
            "day_of_week",
            "wind_bearing_degrees",
            "precip_type",
            "summary",
            "daily_summary",
            "loud_cover",
        ]
    )
    return df


def load_xy(path: Path = DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place weatherHistory.csv under data/ "
            "(Kaggle: 'Weather History' / Szeged dataset)."
        )

    df = pd.read_csv(path)
    df = clean_column_names(df)
    df = engineer_features(df)

    # Leakage avoidance: apparent_temperature is nearly the same signal as target
    X = df.drop(columns=["temperature_c", "apparent_temperature_c"])
    y = df["temperature_c"]
    return X, y


def build_models():
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "RidgeCV": RidgeCV(cv=5),
        "LassoCV": LassoCV(cv=5),
        "ElasticNetCV": ElasticNetCV(cv=5),
    }


def evaluate_models(X_train, X_test, y_train, y_test):
    rows = []
    print("\n=== Model comparison (MAE / MSE / R²) ===\n")

    for name, model in build_models().items():
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rows.append(
            {
                "model": name,
                "mae": mae,
                "mse": mse,
                "r2": r2,
            }
        )
        print(f"{name:16s}  MAE={mae:.4f}  MSE={mse:.4f}  R2={r2:.4f}")

    return pd.DataFrame(rows).sort_values("r2", ascending=False)


def main():
    X, y = load_xy()
    print("Samples:", len(X))
    print("Features:", list(X.columns))
    print("Target: temperature_c")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    summary = evaluate_models(X_train, X_test, y_train, y_test)
    print("\n=== Summary (sorted by R²) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
