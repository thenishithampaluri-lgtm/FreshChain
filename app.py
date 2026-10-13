"""
FreshChain Flask application.

This file is written to work with the model files already present:
    models/data_loader.py
    models/load.py
    models/eda.py
    models/feature_engg.py
    models/logistic_regression.py
    models/predict.py
    models/train_model.py

No existing model files need to be renamed or edited.
Run from the project root with: python app.py
"""

import json
import logging
from pathlib import Path

import pandas as pd
from flask import Flask, flash, render_template, request, url_for
from werkzeug.exceptions import HTTPException

from models.data_loader import load_data
from models.train_model import train_model, MODEL_PATH, METRICS_PATH
from models.predict import predict_spoilage

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "Dataset.csv"
CHART_DIR = PROJECT_ROOT / "static" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = "freshchain-local-development-key"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("freshchain")

NUMERIC_COLUMNS = ["Temp", "Humid (%)", "Light (Fux)", "CO2 (pmm)"]
FEATURE_COLUMNS = ["Fruit"] + NUMERIC_COLUMNS


def get_dataset():
    """Load the CSV through the existing data loader and normalize labels."""
    df = load_data().copy()
    df.columns = df.columns.astype(str).str.strip()

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Fruit"] = df["Fruit"].astype("string").str.strip()
    df["Class"] = df["Class"].astype("string").str.strip().str.casefold()
    df = df.dropna(subset=["Fruit", "Class"]).copy()
    df = df[df["Fruit"].ne("") & df["Class"].ne("")].copy()
    return df


def html_table(frame, index=False):
    """Render a dataframe as an escaped HTML table."""
    return frame.to_html(
        index=index,
        classes="dataframe",
        border=0,
        na_rep="—",
        escape=True,
    )


def read_metrics():
    """Read saved training metrics, if training has already run."""
    try:
        if METRICS_PATH.exists():
            with METRICS_PATH.open("r", encoding="utf-8") as file:
                result = json.load(file)
            return result if isinstance(result, dict) else None
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not read saved model metrics")
    return None


def create_eda_charts(df):
    """Create PNG charts using matplotlib; does not depend on notebook code."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CHART_DIR.mkdir(parents=True, exist_ok=True)
    chart_urls = {}

    def save_chart(filename, draw):
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        try:
            draw(ax, plt)
            fig.savefig(CHART_DIR / filename, dpi=130, bbox_inches="tight")
        finally:
            plt.close(fig)
        chart_urls[filename.removesuffix(".png")] = url_for(
            "static", filename="charts/" + filename
        )

    def class_plot(ax, plt):
        counts = df["Class"].value_counts()
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_title("Fruit Condition Distribution")
        ax.set_xlabel("Condition")
        ax.set_ylabel("Records")

    def fruit_plot(ax, plt):
        counts = df["Fruit"].value_counts().head(15)
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_title("Records by Fruit")
        ax.set_xlabel("Fruit")
        ax.set_ylabel("Records")
        ax.tick_params(axis="x", rotation=35)

    def sensor_plot(ax, plt):
        for column in NUMERIC_COLUMNS:
            values = df[column].dropna()
            if not values.empty:
                ax.hist(values, bins=24, alpha=0.45, label=column)
        ax.set_title("Sensor Reading Distributions")
        ax.set_xlabel("Sensor value")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)

    def correlation_plot(ax, plt):
        corr = df[NUMERIC_COLUMNS].corr()
        image = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=35, ha="right")
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index)
        for row in range(len(corr.index)):
            for col in range(len(corr.columns)):
                value = corr.iloc[row, col]
                if pd.notna(value):
                    ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title("Sensor Feature Correlation")

    save_chart("class_distribution.png", class_plot)
    save_chart("fruit_distribution.png", fruit_plot)
    save_chart("sensor_distributions.png", sensor_plot)
    save_chart("correlation.png", correlation_plot)
    return chart_urls


def render_error(title, message, status=500):
    """Return a readable error page when a template or data operation fails."""
    try:
        return render_template("error.html", title=title, message=message), status
    except Exception:
        # A plain response still gives a useful error if error.html is absent.
        return f"{title}: {message}", status


@app.route("/", endpoint="index")
def dashboard():
    try:
        df = get_dataset()
        class_counts = df["Class"].value_counts().to_dict()
        return render_template(
            "index.html",
            rows=len(df),
            total_records=len(df),
            fruit_count=int(df["Fruit"].nunique()),
            fruit_options=sorted(df["Fruit"].dropna().astype(str).unique().tolist()),
            classes=class_counts,
            class_distribution=class_counts,
            metrics=read_metrics(),
        )
    except Exception:
        logger.exception("Dashboard failed")
        return render_error(
            "Dashboard could not load",
            "Check that data/Dataset.csv exists and contains the required columns.",
        )


@app.route("/data", endpoint="data_loading")
def data_loading_page():
    try:
        df = get_dataset()
        schema = pd.DataFrame({
            "Column": df.columns,
            "Data type": [str(df[col].dtype) for col in df.columns],
            "Non-null values": [int(df[col].notna().sum()) for col in df.columns],
            "Missing values": [int(df[col].isna().sum()) for col in df.columns],
            "Unique values": [int(df[col].nunique(dropna=True)) for col in df.columns],
        })
        class_counts = (
            df["Class"].value_counts()
            .rename_axis("Class")
            .reset_index(name="Records")
        )
        return render_template(
            "data_loading.html",
            rows=len(df),
            columns=len(df.columns),
            missing=int(df.isna().sum().sum()),
            table=html_table(df.head(12)),
            column_info=html_table(schema, index=False),
            class_distribution=html_table(class_counts, index=False),
            data_path=str(DATA_PATH),
            fruit_count=int(df["Fruit"].nunique()),
        )
    except Exception:
        logger.exception("Dataset page failed")
        return render_error(
            "Dataset overview unavailable",
            "Check data/Dataset.csv and verify that it has the required columns.",
        )


@app.route("/eda", endpoint="eda")
def eda_page():
    try:
        df = get_dataset()
        charts = create_eda_charts(df)
        summary_frame = df.describe(include="all").transpose().fillna("")
        summary_html = html_table(summary_frame.reset_index(names="Feature"), index=False)
        numeric_summary = html_table(
            df[NUMERIC_COLUMNS].describe().transpose().reset_index(names="Feature"),
            index=False,
        )
        return render_template(
            "eda.html",
            rows=len(df),
            fruit_count=int(df["Fruit"].nunique()),
            class_count=int(df["Class"].nunique()),
            charts=charts,
            summary=summary_html,
            summary_html=summary_html,
            numeric_summary=numeric_summary,
            class_distribution=df["Class"].value_counts().to_dict(),
        )
    except Exception:
        logger.exception("EDA page failed")
        return render_error("EDA unavailable", "The dataset could not be analysed.")


@app.route("/features", endpoint="feature_engineering")
def feature_engineering_page():
    try:
        df = get_dataset()
        engineered = df.copy()

        engineered["Temp_Humidity_Interaction"] = (
            engineered["Temp"] * engineered["Humid (%)"] / 100
        )
        engineered["CO2_Log1p"] = engineered["CO2 (pmm)"].where(
            engineered["CO2 (pmm)"] >= 0
        ).map(lambda value: __import__("math").log1p(value) if pd.notna(value) else float("nan"))
        engineered["Temp_Squared"] = engineered["Temp"] ** 2
        engineered["Humidity_Distance_50"] = (
            engineered["Humid (%)"] - 50
        ).abs()

        engineered_columns = [
            "Temp_Humidity_Interaction",
            "CO2_Log1p",
            "Temp_Squared",
            "Humidity_Distance_50",
        ]
        preview_columns = FEATURE_COLUMNS + engineered_columns
        preview = engineered[preview_columns].head(10)
        analysis = pd.DataFrame({
            "Feature": preview_columns,
            "Description": [
                "Fruit category",
                "Temperature sensor reading",
                "Humidity sensor reading",
                "Light sensor reading",
                "CO2 sensor reading",
                "Temperature × humidity / 100",
                "Log-transformed nonnegative CO2",
                "Squared temperature reading",
                "Absolute distance from humidity value 50",
            ],
        })
        return render_template(
            "feature_engg.html",
            engineered_feature_count=len(engineered_columns),
            missing_values=int(engineered[preview_columns].isna().sum().sum()),
            feature_preview=html_table(preview),
            feature_analysis=html_table(analysis, index=False),
            features=preview_columns,
        )
    except Exception:
        logger.exception("Feature engineering page failed")
        return render_error(
            "Feature engineering unavailable",
            "The feature preview could not be generated.",
        )


@app.route("/train", methods=["GET", "POST"], endpoint="model_training")
def model_training_page():
    training_message = ""
    if request.method == "POST":
        try:
            _, metrics = train_model()
            training_message = "Logistic Regression training completed successfully."
            flash(training_message, "success")
        except Exception as exc:
            logger.exception("Training failed")
            training_message = f"Training failed: {exc}"
            flash(training_message, "error")

    metrics = read_metrics()
    return render_template(
        "model_training.html",
        metrics=metrics,
        model_status="Trained" if MODEL_PATH.exists() else "Not trained",
        selected_model="Logistic Regression",
        selected_algorithm="Logistic Regression",
        training_message=training_message,
        training_rows=(metrics or {}).get("training_rows", ""),
        testing_rows=(metrics or {}).get("testing_rows", ""),
        comparison_table=None,
        model_filename=MODEL_PATH.name,
    )


@app.route("/logistic-regression", methods=["GET", "POST"], endpoint="logistic_regression")
def logistic_regression_page():
    # Reuse the existing Logistic Regression training module and template route.
    return model_training_page()


@app.route("/predict", methods=["GET", "POST"], endpoint="predict")
def prediction_page():
    try:
        df = get_dataset()
        fruit_options = sorted(df["Fruit"].dropna().astype(str).unique().tolist())
        default_fruit = fruit_options[0] if fruit_options else ""

        form_data = {
            "fruit": default_fruit,
            "temperature": 25.0,
            "humidity": 70.0,
            "light": 100.0,
            "co2": 500.0,
        }
        prediction = None
        probabilities = {}
        confidence = None

        if request.method == "POST":
            try:
                form_data = {
                    "fruit": request.form.get("fruit", default_fruit).strip(),
                    "temperature": float(request.form.get("temperature", "25")),
                    "humidity": float(request.form.get("humidity", "70")),
                    "light": float(request.form.get("light", "100")),
                    "co2": float(request.form.get("co2", "500")),
                }
                result = predict_spoilage(
                    fruit=form_data["fruit"],
                    temperature=form_data["temperature"],
                    humidity=form_data["humidity"],
                    light=form_data["light"],
                    co2=form_data["co2"],
                )
                prediction = result.get("prediction")
                probabilities = result.get("probabilities", {})
                confidence = max(probabilities.values()) if probabilities else None
            except ValueError:
                flash("Please enter valid numeric sensor readings.", "error")
            except FileNotFoundError:
                flash("Train the model before making a prediction.", "error")
            except Exception as exc:
                logger.exception("Prediction failed")
                flash(f"Prediction failed: {exc}", "error")

        return render_template(
            "predict.html",
            fruit_options=fruit_options,
            fruits=fruit_options,
            prediction=prediction,
            result=prediction,
            confidence=confidence,
            probabilities=probabilities,
            form_data=form_data,
            form=form_data,
        )
    except Exception:
        logger.exception("Prediction page failed")
        return render_error(
            "Prediction unavailable",
            "The dataset could not be loaded.",
        )


# Compatibility endpoints used by older versions of the templates.
app.add_url_rule("/data", endpoint="data_loading_page", view_func=data_loading_page)
app.add_url_rule("/eda", endpoint="eda_page", view_func=eda_page)
app.add_url_rule("/features", endpoint="feature_engineering_page", view_func=feature_engineering_page)
app.add_url_rule("/train", endpoint="model_training_page", view_func=model_training_page)
app.add_url_rule("/logistic-regression", endpoint="logistic_regression_page", view_func=logistic_regression_page)
app.add_url_rule("/predict", endpoint="prediction_page", view_func=prediction_page)


@app.errorhandler(HTTPException)
def handle_http_error(error):
    if error.code == 404:
        return render_error("Page not found", "The requested page does not exist.", 404)
    return render_error("Request error", error.description, error.code or 500)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
