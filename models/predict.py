from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "logistic_regression_pipeline.joblib"

FEATURES = [
    "Fruit",
    "Temp",
    "Humid (%)",
    "Light (Fux)",
    "CO2 (pmm)",
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run "
            "'python -m models.train_model' first."
        )

    return joblib.load(MODEL_PATH)


def predict_spoilage(
    fruit,
    temperature,
    humidity,
    light,
    co2,
):
    model = load_model()

    sample = pd.DataFrame([{
        "Fruit": fruit,
        "Temp": temperature,
        "Humid (%)": humidity,
        "Light (Fux)": light,
        "CO2 (pmm)": co2,
    }], columns=FEATURES)

    prediction = str(model.predict(sample)[0])

    result = {
        "prediction": prediction,
        "status": (
            "Spoilage detected"
            if prediction == "bad"
            else "Good condition"
        ),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(sample)[0]
        result["probabilities"] = {
            str(label): round(float(probability), 4)
            for label, probability in zip(model.classes_, probabilities)
        }

    return result


if __name__ == "__main__":
    result = predict_spoilage(
        fruit="Apple",
        temperature=25,
        humidity=70,
        light=100,
        co2=500,
    )

    print("Prediction result:")
    print(result)
