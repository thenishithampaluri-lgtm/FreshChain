# ============================================================
# FRESHCHAIN - COMPLETE FEATURE ENGINEERING
# Fruit Spoilage Prediction
# ============================================================

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------- 1. CONFIGURATION ---------------------------

DATA_PATH = Path("data/Dataset.csv")

if not DATA_PATH.exists():
    DATA_PATH = Path("Dataset (1)(1).csv")

OUTPUT_DIR = Path("feature_engineering_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Class"

ORIGINAL_FEATURES = [
    "Fruit",
    "Temp",
    "Humid (%)",
    "Light (Fux)",
    "CO2 (pmm)"
]

NUMERIC_FEATURES = [
    "Temp",
    "Humid (%)",
    "Light (Fux)",
    "CO2 (pmm)"
]

# ---------------- 2. LOAD DATA ------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Dataset not found. Update DATA_PATH to your CSV location."
    )

df = pd.read_csv(DATA_PATH)

required = ORIGINAL_FEATURES + [TARGET]

missing_columns = set(required) - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

print("=" * 65)
print("FRESHCHAIN FEATURE ENGINEERING")
print("=" * 65)

print("Original dataset:", df.shape)
print(df.head())

# ---------------- 3. CLEAN DATA TYPES -----------------------

data = df[required].copy()

data["Fruit"] = (
    data["Fruit"].astype("string").str.strip()
)

data[TARGET] = (
    data[TARGET].astype("string").str.strip().str.casefold()
)

data[TARGET] = data[TARGET].replace("", pd.NA)

for col in NUMERIC_FEATURES:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Keep rows with valid target labels.
data = data.dropna(subset=[TARGET]).copy()

if data[TARGET].nunique() < 2:
    raise ValueError("At least two target classes are required.")

print("\nTarget distribution:")
print(data[TARGET].value_counts())

# ---------------- 4. FEATURE ENGINEERING --------------------

X = data[ORIGINAL_FEATURES].copy()
y = data[TARGET].copy()

# Feature 1: Temperature-humidity interaction.
X["Temp_Humidity_Interaction"] = (
    X["Temp"] * X["Humid (%)"] / 100
)

# Feature 2: Log transformation of nonnegative CO2 readings.
X["CO2_Log1p"] = np.log1p(
    X["CO2 (pmm)"].where(X["CO2 (pmm)"] >= 0)
)

# Feature 3: Nonlinear temperature term.
X["Temp_Squared"] = X["Temp"] ** 2

# Feature 4: Distance from a reference humidity level.
# This is an experimental feature, not a spoilage threshold.
X["Humidity_Distance_50"] = (
    X["Humid (%)"] - 50
).abs()

ENGINEERED_NUMERIC = NUMERIC_FEATURES + [
    "Temp_Humidity_Interaction",
    "CO2_Log1p",
    "Temp_Squared",
    "Humidity_Distance_50"
]

CATEGORICAL_FEATURES = ["Fruit"]

print("\nOriginal feature count:", len(ORIGINAL_FEATURES))
print("Engineered feature count:", X.shape[1])
print("\nEngineered feature preview:")
print(X.head())

# Save unscaled engineered features for inspection.
engineered_output = X.copy()
engineered_output[TARGET] = y

engineered_output.to_csv(
    OUTPUT_DIR / "engineered_dataset.csv",
    index=False
)

# ---------------- 5. TRAIN / TEST SPLIT ---------------------

class_counts = y.value_counts()

stratify_arg = (
    y if len(class_counts) > 1 and class_counts.min() >= 2
    else None
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=stratify_arg
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# ---------------- 6. BUILD PREPROCESSOR ---------------------

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    (
        "categorical",
        categorical_pipeline,
        CATEGORICAL_FEATURES
    ),
    (
        "numeric",
        numeric_pipeline,
        ENGINEERED_NUMERIC
    )
])

# ---------------- 7. FIT ON TRAINING DATA ONLY ---------------

X_train_array = preprocessor.fit_transform(X_train)
X_test_array = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

X_train_processed = pd.DataFrame(
    X_train_array,
    columns=feature_names,
    index=X_train.index
)

X_test_processed = pd.DataFrame(
    X_test_array,
    columns=feature_names,
    index=X_test.index
)

print("\nProcessed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)

print("\nProcessed feature names:")
print(list(feature_names))

print("\nProcessed training preview:")
print(X_train_processed.head())

# ---------------- 8. SAVE ALL OUTPUTS -----------------------

X_train_processed.assign(
    **{TARGET: y_train}
).to_csv(
    OUTPUT_DIR / "train_processed.csv",
    index=False
)

X_test_processed.assign(
    **{TARGET: y_test}
).to_csv(
    OUTPUT_DIR / "test_processed.csv",
    index=False
)

joblib.dump(
    preprocessor,
    OUTPUT_DIR / "preprocessor.joblib"
)

pd.Series(
    feature_names,
    name="feature"
).to_csv(
    OUTPUT_DIR / "processed_feature_names.csv",
    index=False
)

# Save labels separately for convenient downstream training.
y_train.rename(TARGET).to_csv(
    OUTPUT_DIR / "y_train.csv",
    index=False
)

y_test.rename(TARGET).to_csv(
    OUTPUT_DIR / "y_test.csv",
    index=False
)

print("\n" + "=" * 65)
print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nFiles saved in:", OUTPUT_DIR.resolve())

for file in sorted(OUTPUT_DIR.iterdir()):
    print(" -", file.name)

print("\nNext step: train and compare classification models.")
print("Note: engineered features must be validated against")
print("the original features before deciding whether they help.")