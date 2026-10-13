from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score
)

# =====================================================
# 1. PROJECT PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "Dataset.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACT_DIR / "logistic_regression_pipeline.joblib"
METRICS_PATH = ARTIFACT_DIR / "logistic_regression_metrics.json"

# =====================================================
# 2. LOAD DATASET
# =====================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}\n"
        "Place Dataset.csv inside the project's data folder."
    )

df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("FRESHCHAIN - LOGISTIC REGRESSION")
print("=" * 65)

print("\nDataset shape:", df.shape)
print("\nOriginal columns:", df.columns.tolist())

# Remove accidental whitespace from column names
df.columns = df.columns.astype(str).str.strip()

required_columns = [
    "Fruit",
    "Temp",
    "Humid (%)",
    "Light (Fux)",
    "CO2 (pmm)",
    "Class"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}\n"
        f"Available columns: {df.columns.tolist()}"
    )

# =====================================================
# 3. CLEAN DATA
# =====================================================

df["Fruit"] = df["Fruit"].astype("string").str.strip()

# Normalize Good, Bad and BAD to consistent labels
df["Class"] = (
    df["Class"]
    .astype("string")
    .str.strip()
    .str.casefold()
)

numeric_features = [
    "Temp",
    "Humid (%)",
    "Light (Fux)",
    "CO2 (pmm)"
]

categorical_features = ["Fruit"]

for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows without a valid target or fruit category
df = df.dropna(subset=["Class", "Fruit"]).copy()

df = df[
    df["Class"].ne("")
    & df["Fruit"].ne("")
].copy()

if df["Class"].nunique() < 2:
    raise ValueError("Logistic Regression requires at least two target classes.")

print("\nTarget distribution:")
print(df["Class"].value_counts())

print("\nMissing values after conversion:")
print(df[required_columns].isnull().sum())

# =====================================================
# 4. DEFINE FEATURES AND TARGET
# =====================================================

X = df[categorical_features + numeric_features].copy()
y = df["Class"].copy()

# Split before fitting preprocessing to prevent leakage
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# =====================================================
# 5. PREPROCESSING
# =====================================================

# Fruit: fill missing values and encode categories
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Sensor readings: impute missing values and standardize
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("categorical", categorical_pipeline, categorical_features),
    ("numeric", numeric_pipeline, numeric_features)
])

# =====================================================
# 6. LOGISTIC REGRESSION PIPELINE
# =====================================================

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        )
    )
])

# =====================================================
# 7. HYPERPARAMETER TUNING
# =====================================================

# Tune regularization strength and penalty.
# Keep all preprocessing inside the pipeline.
param_grid = [
    {
        "classifier__solver": ["liblinear"],
        "classifier__penalty": ["l1", "l2"],
        "classifier__C": [0.01, 0.1, 1.0, 10.0]
    },
    {
        "classifier__solver": ["lbfgs"],
        "classifier__penalty": ["l2"],
        "classifier__C": [0.01, 0.1, 1.0, 10.0]
    }
]

minimum_class_count = int(y_train.value_counts().min())
cv_splits = min(5, minimum_class_count)

if cv_splits < 2:
    raise ValueError(
        "Not enough training examples in each class for cross-validation."
    )

cv = StratifiedKFold(
    n_splits=cv_splits,
    shuffle=True,
    random_state=42
)

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    refit=True,
    verbose=1
)

print("\nStarting hyperparameter tuning...")
search.fit(X_train, y_train)

best_model = search.best_estimator_

print("\nBest parameters:")
print(search.best_params_)

print("\nBest cross-validation macro F1:")
print(round(search.best_score_, 4))

# =====================================================
# 8. EVALUATE ON HELD-OUT TEST DATA
# =====================================================

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print("\n" + "=" * 65)
print("TEST SET EVALUATION")
print("=" * 65)

print(f"\nAccuracy:          {accuracy:.4f}")
print(f"Balanced accuracy: {balanced_accuracy:.4f}")
print(f"Macro F1-score:    {macro_f1:.4f}")

print("\nClassification report:")
report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)
print(report)

labels = sorted(y.unique())

print("\nConfusion matrix (rows = actual, columns = predicted):")
cm = confusion_matrix(y_test, y_pred, labels=labels)
print("Class order:", labels)
print(cm)

# ROC-AUC: binary or multiclass, when supported
try:
    y_proba = best_model.predict_proba(X_test)

    if len(labels) == 2:
        auc = roc_auc_score(
            y_test,
            y_proba[:, list(best_model.classes_).index(labels[1])],
            labels=labels
        )
    else:
        auc = roc_auc_score(
            y_test,
            y_proba,
            labels=best_model.classes_,
            multi_class="ovr",
            average="macro"
        )

    print(f"\nROC-AUC: {auc:.4f}")

except (ValueError, AttributeError) as exc:
    auc = None
    print("\nROC-AUC could not be calculated:", exc)

# =====================================================
# 9. SAVE MODEL AND METRICS
# =====================================================

joblib.dump(best_model, MODEL_PATH)

metrics = {
    "model": "Logistic Regression",
    "dataset_rows_used": int(len(df)),
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "classes": [str(c) for c in labels],
    "accuracy": float(accuracy),
    "balanced_accuracy": float(balanced_accuracy),
    "macro_f1": float(macro_f1),
    "roc_auc": float(auc) if auc is not None else None,
    "best_parameters": search.best_params_,
    "best_cv_macro_f1": float(search.best_score_)
}

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("\nSaved model to:", MODEL_PATH)
print("Saved metrics to:", METRICS_PATH)
print("\nTraining and evaluation completed successfully.")
