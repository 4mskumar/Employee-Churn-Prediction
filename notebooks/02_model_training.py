"""
STEP 2 — TRAIN AND COMPARE MACHINE-LEARNING MODELS
Run after Step 1:
    python notebooks/02_model_training_COMPLETE.py

A single sklearn Pipeline handles numeric scaling and categorical encoding so the API can
reuse exactly the same preprocessing at prediction time.
"""
from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


NUMERIC_COLUMNS = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "Education",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

CATEGORICAL_COLUMNS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def build_preprocessor():
    """Fill missing numeric values and scale them; fill and one-hot encode categories."""
    numeric_pipeline = Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
        ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS)
    ])

def calculate_metrics(model, X_test, y_test):
    """Return common classification metrics. Positive class 1 means Attrition=Yes."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4)
    }

def main():
    print("=" * 65)
    print("STEP 2: TRAIN AND COMPARE MODELS")
    print("=" * 65)

    needed = [DATA_DIR / "X_train.csv", DATA_DIR / "X_test.csv",
              DATA_DIR / "y_train.csv", DATA_DIR / "y_test.csv"]
    missing = [str(path) for path in needed if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Training files are missing. Run Step 1 first:\n"
            "python notebooks/01_data_preparation_COMPLETE.py\nMissing:\n" + "\n".join(missing)
        )

    X_train = pd.read_csv(DATA_DIR / "X_train.csv")[FEATURE_COLUMNS]
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")[FEATURE_COLUMNS]
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")["Attrition"].astype(int)
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")["Attrition"].astype(int)

    # Scale positive-class weight for XGBoost so it pays more attention to minority labels.
    positive_count = max(1, int(y_train.sum()))
    negative_count = max(1, int(len(y_train) - y_train.sum()))
    imbalance_ratio = negative_count / positive_count

    model_definitions = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=imbalance_ratio, eval_metric="logloss",
            random_state=42, n_jobs=4, use_label_encoder=False
        )
    }

    trained = {}
    metric_rows = []
    for name, classifier in model_definitions.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline([
            ("preprocessing", build_preprocessor()),
            ("model", classifier)
        ])
        pipeline.fit(X_train, y_train)
        metrics = calculate_metrics(pipeline, X_test, y_test)
        trained[name] = pipeline
        metric_rows.append({"model": name, **metrics})
        print("Metrics:", metrics)
        print(classification_report(
            y_test, pipeline.predict(X_test),
            target_names=["Stayed (No)", "Left (Yes)"], zero_division=0
        ))

    results = pd.DataFrame(metric_rows).sort_values("f1", ascending=False)
    print("\nMODEL COMPARISON (sorted by F1 score):")
    print(results.to_string(index=False))

    # Select using F1, which balances precision and recall; do not assume one model is always best.
    best_name = str(results.iloc[0]["model"])
    best_model = trained[best_name]
    model_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(best_model, model_path)

    # Save metadata for the API/dashboard.
    metadata = {
        "best_model": best_name,
        "metrics": {row["model"]: {k: row[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
                    for row in metric_rows},
        "feature_columns": FEATURE_COLUMNS,
        "target": "Attrition",
        "data_note": "Synthetic demo dataset; metrics are for learning, not real-world HR decisions."
    }
    with open(MODEL_DIR / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    results.to_csv(DATA_DIR / "model_comparison.csv", index=False)
    ax = results.set_index("model")[["precision", "recall", "f1", "roc_auc"]].plot(
        kind="bar", figsize=(11, 6), ylim=(0, 1), rot=0
    )
    ax.set_title("Model comparison on held-out synthetic test data")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(DATA_DIR / "model_comparison.png", dpi=150)
    plt.close()

    # Plot a confusion matrix for the selected model.
    predictions = best_model.predict(X_test)
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, predictions),
        display_labels=["Stayed (No)", "Left (Yes)"]
    ).plot()
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(DATA_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # Permutation importance works with the complete pipeline and original columns.
    try:
        from sklearn.inspection import permutation_importance
        importance = permutation_importance(
            best_model, X_test, y_test, scoring="f1", n_repeats=5, random_state=42, n_jobs=1
        )
        importance_df = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std
        }).sort_values("importance_mean", ascending=False)
        importance_df.to_csv(DATA_DIR / "feature_importance.csv", index=False)
        top = importance_df.head(15).sort_values("importance_mean")
        plt.figure(figsize=(10, 7))
        plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
        plt.title("Permutation importance (change in F1 when feature is shuffled)")
        plt.xlabel("Mean F1 decrease")
        plt.tight_layout()
        plt.savefig(DATA_DIR / "feature_importance.png", dpi=150)
        plt.close()
    except Exception as error:
        print("Feature importance plot could not be generated:", error)

    print(f"\nBest model by F1: {best_name}")
    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {MODEL_DIR / 'model_info.json'}")
    print("\nStep 2 complete. Next run:")
    print("python backend/app_COMPLETE.py")

if __name__ == "__main__":
    main()
