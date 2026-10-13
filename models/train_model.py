import json
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)

from .data_loader import load_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACT_DIR / "logistic_regression_pipeline.joblib"
METRICS_PATH = ARTIFACT_DIR / "logistic_regression_metrics.json"

CATEGORICAL_FEATURES = ["Fruit"]
NUMERIC_FEATURES = [
    "Temp",
    "Humid (%)",
    "Light (Fux)",
    "CO2 (pmm)",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def train_model():
    df = load_data()

    for col in NUMERIC_FEATURES:
        df[col] = __import__("pandas").to_numeric(
            df[col], errors="coerce"
        )

    df = df.dropna(subset=["Class"]).copy()

    X = df[FEATURES]
    y = df["Class"]

    if y.nunique() < 2:
        raise ValueError("Training requires at least two target classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "model": "Logistic Regression",
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "macro_f1": float(
            f1_score(y_test, predictions, average="macro")
        ),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
    }

    joblib.dump(model, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("Logistic Regression training complete!")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1:          {metrics['macro_f1']:.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))
    print("\nModel saved to:", MODEL_PATH)
    print("Metrics saved to:", METRICS_PATH)

    return model, metrics


if __name__ == "__main__":
    train_model()
