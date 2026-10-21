from pathlib import Path
import json

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Restrict allowed origins before public deployment.

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "models"
DATA_DIR = PROJECT_DIR / "data"

MODEL_PATH = MODEL_DIR / "best_model.joblib"
METADATA_PATH = MODEL_DIR / "model_info.json"
TRAIN_DATA_PATH = DATA_DIR / "X_train.csv"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. Run model training first."
    )
if not TRAIN_DATA_PATH.exists():
    raise FileNotFoundError(
        f"Training data not found: {TRAIN_DATA_PATH}. Run data preparation first."
    )

model = joblib.load(MODEL_PATH)
X_train = pd.read_csv(TRAIN_DATA_PATH)
FEATURE_COLUMNS = X_train.columns.tolist()

metadata = {}
if METADATA_PATH.exists():
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)


def prepare_employee(employee):
    if not isinstance(employee, dict):
        raise ValueError("Employee data must be a JSON object.")

    missing = [column for column in FEATURE_COLUMNS if column not in employee]
    if missing:
        raise ValueError("Missing required features: " + ", ".join(missing))

    row = pd.DataFrame(
        [{column: employee[column] for column in FEATURE_COLUMNS}],
        columns=FEATURE_COLUMNS,
    )

    numeric_columns = X_train.select_dtypes(include="number").columns
    for column in numeric_columns:
        row[column] = pd.to_numeric(row[column], errors="coerce")

    if row.isnull().any().any():
        raise ValueError("Employee data contains empty or invalid values.")

    return row


def probability_for_positive_class(probabilities):
    classes = list(model.classes_)
    return float(probabilities[classes.index(1)]) if 1 in classes else None


def risk_label(probability):
    if probability is None:
        return "Unavailable"
    return "High" if probability >= 0.5 else "Lower"


@app.get("/api/health")
def health():
    return jsonify({
        "status": "healthy",
        "message": "Employee Churn Prediction API is running.",
    })


@app.get("/api/features")
def get_features():
    features = []
    for column in FEATURE_COLUMNS:
        series = X_train[column]
        feature = {
            "name": column,
            "type": "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical",
        }
        if feature["type"] == "categorical":
            values = series.dropna().unique().tolist()
            if len(values) <= 30:
                feature["options"] = [str(value) for value in values]
        features.append(feature)

    return jsonify({"count": len(features), "features": features})


@app.get("/api/model-info")
def get_model_info():
    return jsonify({
        "model": metadata,
        "input_feature_count": len(FEATURE_COLUMNS),
        "input_features": FEATURE_COLUMNS,
    })


@app.get("/api/evaluation/chart/<chart_name>")
def evaluation_chart(chart_name):
    allowed_charts = {
        "model_comparison.png",
        "confusion_matrix.png",
        "feature_importance.png",
    }
    if chart_name not in allowed_charts:
        return jsonify({"error": "Chart not found."}), 404

    chart_path = DATA_DIR / chart_name
    if not chart_path.is_file():
        return jsonify({
            "error": f"{chart_name} has not been generated. Run model training again."
        }), 404

    return send_from_directory(str(DATA_DIR), chart_name)


@app.post("/api/predict")
def predict():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Send employee data as a JSON object."}), 400

        employee = data.get("employee", data)
        row = prepare_employee(employee)
        prediction = int(model.predict(row)[0])

        probability = None
        if hasattr(model, "predict_proba"):
            probability = probability_for_positive_class(model.predict_proba(row)[0])

        return jsonify({
            "prediction": prediction,
            "result": "Left (Yes)" if prediction == 1 else "Stayed (No)",
            "attrition_probability": probability,
            "risk_level": risk_label(probability),
            "note": (
                "This is a model estimate, not a definitive judgment about an employee."
            ),
        })
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Check the backend terminal."}), 500


@app.post("/api/predict-batch")
def predict_batch():
    try:
        if "file" not in request.files:
            return jsonify({
                "error": "Upload a CSV using the form field named 'file'."
            }), 400

        file = request.files["file"]

        if not file.filename or not file.filename.lower().endswith(".csv"):
            return jsonify({"error": "Please upload a .csv file."}), 400

        employees = pd.read_csv(file)
        
        # Save employee IDs before selecting model features.
        employee_numbers = (
            employees["EmployeeNumber"].tolist()
            if "EmployeeNumber" in employees.columns
            else [None] * len(employees)
        )

        if employees.empty:
            return jsonify({
                "error": "The CSV contains no employee rows."
            }), 400

        # Preserve the employee identifier separately.
        # It must never be passed to the trained model.
        if "EmployeeNumber" in employees.columns:
            employee_numbers = employees["EmployeeNumber"].tolist()
        else:
            employee_numbers = [None] * len(employees)

        # Check the required model features.
        missing = [
            column for column in FEATURE_COLUMNS
            if column not in employees.columns
        ]

        if missing:
            return jsonify({
                "error": "CSV is missing required features.",
                "missing_features": missing,
            }), 400

        # Pass only the features used during training.
        model_input = employees[FEATURE_COLUMNS].copy()

        numeric_columns = X_train.select_dtypes(
            include="number"
        ).columns

        for column in numeric_columns:
            model_input[column] = pd.to_numeric(
                model_input[column],
                errors="coerce"
            )

        if model_input.isnull().any().any():
            return jsonify({
                "error": (
                    "CSV contains empty or invalid feature values. "
                    "Check the required columns."
                )
            }), 400

        predictions = model.predict(model_input)

        probabilities = (
            model.predict_proba(model_input)
            if hasattr(model, "predict_proba")
            else None
        )

        classes = list(model.classes_) if probabilities is not None else []

        results = []

        for index, prediction in enumerate(predictions):
            probability = (
                float(probabilities[index][classes.index(1)])
                if probabilities is not None and 1 in classes
                else None
            )

            results.append({
                "row": index + 1,
                "employee_number": (
                    str(employee_numbers[index])
                    if pd.notna(employee_numbers[index])
                    else None
                ),
                "prediction": int(prediction),
                "result": (
                    "Left (Yes)" if int(prediction) == 1
                    else "Stayed (No)"
                ),
                "attrition_probability": probability,
                "risk_level": risk_label(probability),
            })

        return jsonify({
            "total_employees": len(results),
            "results": results,
            "note": (
                "Demo predictions from a fictional dataset; "
                "not employment decisions."
            ),
        })

    except Exception:
        app.logger.exception("Batch prediction failed")
        return jsonify({
            "error": "Batch prediction failed. Check the backend terminal."
        }), 500


if __name__ == "__main__":
    print("Loaded model:", metadata.get("best_model", "Unknown"))
    print("API ready at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
