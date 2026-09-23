```python
"""
Train and save pre-trained regression models for:

1. Maximum temperature prediction
2. Minimum temperature prediction

Dataset:
    data/weather.csv

Expected columns:
    Date
    Min_Temperature
    Max_Temperature
    Humidity
    Wind_Speed
    Weather_Condition

The trained models are saved in:

    models/max_temperature_model.pkl
    models/min_temperature_model.pkl
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "weather.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load Dataset
# ============================================================

print("Loading dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded successfully: {df.shape}")
print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# Validate Required Columns
# ============================================================

required_columns = [
    "Date",
    "Min_Temperature",
    "Max_Temperature",
    "Humidity",
    "Wind_Speed",
    "Weather_Condition",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# Data Cleaning
# ============================================================

print("\nCleaning dataset...")

# Convert Date to datetime
df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

# Remove rows with invalid dates
df = df.dropna(subset=["Date"])


# Convert numerical columns to numeric
numeric_columns = [
    "Min_Temperature",
    "Max_Temperature",
    "Humidity",
    "Wind_Speed",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# Remove rows where target values are missing
df = df.dropna(
    subset=[
        "Min_Temperature",
        "Max_Temperature"
    ]
)


# Clean categorical column
df["Weather_Condition"] = (
    df["Weather_Condition"]
    .astype("string")
    .str.strip()
)


print(f"Cleaned dataset: {df.shape}")


# ============================================================
# Feature Engineering
# ============================================================

print("\nCreating date features...")

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfYear"] = df["Date"].dt.dayofyear
df["DayOfWeek"] = df["Date"].dt.dayofweek


# ============================================================
# Features and Targets
# ============================================================

feature_columns = [
    "Year",
    "Month",
    "Day",
    "DayOfYear",
    "DayOfWeek",
    "Humidity",
    "Wind_Speed",
    "Weather_Condition",
]

X = df[feature_columns]

y_max = df["Max_Temperature"]
y_min = df["Min_Temperature"]


# ============================================================
# Train/Test Split
# ============================================================

X_train, X_test, y_max_train, y_max_test = train_test_split(
    X,
    y_max,
    test_size=0.20,
    random_state=42
)

# Use the same split for minimum temperature
_, _, y_min_train, y_min_test = train_test_split(
    X,
    y_min,
    test_size=0.20,
    random_state=42
)


# ============================================================
# Preprocessing
# ============================================================

numeric_features = [
    "Year",
    "Month",
    "Day",
    "DayOfYear",
    "DayOfWeek",
    "Humidity",
    "Wind_Speed",
]

categorical_features = [
    "Weather_Condition"
]


numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# Maximum Temperature Model
# ============================================================

print("\nTraining maximum-temperature model...")

max_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


max_model.fit(
    X_train,
    y_max_train
)


# ============================================================
# Minimum Temperature Model
# ============================================================

print("Training minimum-temperature model...")

# Create a separate preprocessor so each pipeline is independent.
min_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    )
                ]
            ),
            numeric_features
        ),
        (
            "categorical",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]
            ),
            categorical_features
        )
    ]
)


min_model = Pipeline(
    steps=[
        (
            "preprocessor",
            min_preprocessor
        ),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


min_model.fit(
    X_train,
    y_min_train
)


# ============================================================
# Model Evaluation
# ============================================================

print("\nEvaluating models...")


# Maximum temperature
max_predictions = max_model.predict(X_test)

max_mae = mean_absolute_error(
    y_max_test,
    max_predictions
)

max_rmse = mean_squared_error(
    y_max_test,
    max_predictions
) ** 0.5

max_r2 = r2_score(
    y_max_test,
    max_predictions
)


# Minimum temperature
min_predictions = min_model.predict(X_test)

min_mae = mean_absolute_error(
    y_min_test,
    min_predictions
)

min_rmse = mean_squared_error(
    y_min_test,
    min_predictions
) ** 0.5

min_r2 = r2_score(
    y_min_test,
    min_predictions
)


print("\n==========================================")
print("MODEL PERFORMANCE")
print("==========================================")

print("\nMaximum Temperature Model")
print(f"MAE  : {max_mae:.3f}")
print(f"RMSE : {max_rmse:.3f}")
print(f"R²   : {max_r2:.3f}")

print("\nMinimum Temperature Model")
print(f"MAE  : {min_mae:.3f}")
print(f"RMSE : {min_rmse:.3f}")
print(f"R²   : {min_r2:.3f}")


# ============================================================
# Save Models
# ============================================================

max_model_path = (
    MODEL_DIR / "max_temperature_model.pkl"
)

min_model_path = (
    MODEL_DIR / "min_temperature_model.pkl"
)


joblib.dump(
    max_model,
    max_model_path
)

joblib.dump(
    min_model,
    min_model_path
)


# ============================================================
# Finished
# ============================================================

print("\n==========================================")
print("TRAINING COMPLETE")
print("==========================================")

print(
    f"\nMaximum-temperature model saved to:\n"
    f"{max_model_path}"
)

print(
    f"\nMinimum-temperature model saved to:\n"
    f"{min_model_path}"
)

print("\nThese models are now ready to be included")
print("in the Docker image.")
```
