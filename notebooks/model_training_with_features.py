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

print("\n" + "="*70)
print("="*70)


needed = [DATA_DIR / "X_train.csv", DATA_DIR / "X_test.csv",
          DATA_DIR / "y_train.csv", DATA_DIR / "y_test.csv"]
missing = [str(path) for path in needed if not path.exists()]
if missing:
    raise FileNotFoundError(
        "Training files are missing. Run Step 1 first:\n"
        "python notebooks/01_data_preparation_WITH_FEATURES.py\nMissing:\n" + "\n".join(missing)
    )

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv")["Attrition"].astype(int)
y_test = pd.read_csv(DATA_DIR / "y_test.csv")["Attrition"].astype(int)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

NUMERIC_COLUMNS = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
CATEGORICAL_COLUMNS = X_train.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric columns: {len(NUMERIC_COLUMNS)}")
print(f"Categorical columns: {len(CATEGORICAL_COLUMNS)}")

FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

print("\n" + "="*70)
print("BUILDING PREPROCESSOR PIPELINE")
print("="*70)

def build_preprocessor():
    """Fill missing values and scale/encode features."""
    numeric_pipeline = Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
        ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS)
    ])

preprocessor = build_preprocessor()
print("✓ Preprocessor built")

print("\n" + "="*70)
print("DEFINING MODELS")
print("="*70)

# Calculate class weight for imbalanced data
positive_count = max(1, int(y_train.sum()))
negative_count = max(1, int(len(y_train) - y_train.sum()))
imbalance_ratio = negative_count / positive_count

print(f"\nClass balance:")
print(f"  No (0): {negative_count} employees")
print(f"  Yes (1): {positive_count} employees")
print(f"  Imbalance ratio: {imbalance_ratio:.2f}")

model_definitions = {
    "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=250, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=imbalance_ratio, eval_metric="logloss",
        random_state=42, n_jobs=4, verbosity=0
    )
}


print("\n" + "="*70)
print("TRAINING MODELS")
print("="*70)

trained = {}
metric_rows = []

for name, classifier in model_definitions.items():
    print(f"\n>>> Training {name}...")
    pipeline = Pipeline([
        ("preprocessing", build_preprocessor()),
        ("model", classifier)
    ])
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4)
    }
    
    trained[name] = pipeline
    metric_rows.append({"model": name, **metrics})
    
    print(f"\nMetrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(
        y_test, predictions,
        target_names=["Stayed (No)", "Left (Yes)"], zero_division=0
    ))

print("\n" + "="*70)
print("MODEL COMPARISON (Sorted by F1-Score)")
print("="*70)

results = pd.DataFrame(metric_rows).sort_values("f1", ascending=False)
print("\n" + results.to_string(index=False))

# Select best model by F1-score
best_name = str(results.iloc[0]["model"])
best_model = trained[best_name]
print(f"\n>>> BEST MODEL: {best_name}")
print(f"    F1-Score: {results.iloc[0]['f1']:.4f}")
print(f"    Accuracy: {results.iloc[0]['accuracy']:.4f}")
print(f"    Precision: {results.iloc[0]['precision']:.4f}")
print(f"    Recall: {results.iloc[0]['recall']:.4f}")
print(f"    ROC-AUC: {results.iloc[0]['roc_auc']:.4f}")


print("\n" + "="*70)
print("SAVING MODEL AND METADATA")
print("="*70)

# Save the best model
model_path = MODEL_DIR / "best_model.joblib"
joblib.dump(best_model, model_path)
print(f"\n✓ Model saved to: {model_path}")

# Save metadata for the API/dashboard
metadata = {
    "best_model": best_name,
    "metrics": {row["model"]: {k: row[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
                for row in metric_rows},
    "feature_columns": FEATURE_COLUMNS,
    "numeric_columns": NUMERIC_COLUMNS,
    "categorical_columns": CATEGORICAL_COLUMNS,
    "target": "Attrition",
    "data_note": "Real Kaggle HR dataset; improved with 15 engineered features"
}

metadata_path = MODEL_DIR / "model_info.json"
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Metadata saved to: {metadata_path}")

# Save model comparison
results.to_csv(DATA_DIR / "model_comparison.csv", index=False)
print(f"✓ Model comparison saved to: {DATA_DIR / 'model_comparison.csv'}")

print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# Plot 1: Model Comparison
ax = results.set_index("model")[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot(
    kind="bar", figsize=(12, 6), ylim=(0, 1), rot=45
)
ax.set_title("Model Comparison on Test Data (With Engineered Features)")
ax.set_ylabel("Score")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(DATA_DIR / "model_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ model_comparison.png created")

# Plot 2: Confusion Matrix
predictions = best_model.predict(X_test)
cm = confusion_matrix(y_test, predictions)
ConfusionMatrixDisplay(cm, display_labels=["Stayed (No)", "Left (Yes)"]).plot()
plt.title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.savefig(DATA_DIR / "confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ confusion_matrix.png created")

# Plot 3: Feature Importance (if available)
try:
    from sklearn.inspection import permutation_importance
    
    print("\nCalculating feature importance...")
    importance = permutation_importance(
        best_model, X_test, y_test, scoring="f1", n_repeats=5, random_state=42, n_jobs=-1
    )
    importance_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std
    }).sort_values("importance_mean", ascending=False)
    
    importance_df.to_csv(DATA_DIR / "feature_importance.csv", index=False)
    
    # Plot top 15 features
    top = importance_df.head(15).sort_values("importance_mean")
    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color='steelblue')
    plt.xlabel("Mean F1 Decrease")
    plt.title("Permutation Feature Importance (Top 15)")
    plt.tight_layout()
    plt.savefig(DATA_DIR / "feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ feature_importance.png created")
    
    print("\nTop 10 Most Important Features:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:<40} {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")
    
except Exception as e:
    print(f"⚠ Could not create feature importance plot: {e}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)

print(f"""
SUMMARY:
  Best Model: {best_name}
  Accuracy: {results.iloc[0]['accuracy']:.2%}
  Precision: {results.iloc[0]['precision']:.2%}
  Recall: {results.iloc[0]['recall']:.2%}
  F1-Score: {results.iloc[0]['f1']:.4f}
  ROC-AUC: {results.iloc[0]['roc_auc']:.4f}
  
FILES CREATED:
  ✓ best_model.joblib - Trained model
  ✓ model_info.json - Model metadata
  ✓ model_comparison.csv - Metrics comparison
  ✓ model_comparison.png - Visualization
  ✓ confusion_matrix.png - Confusion matrix
  ✓ feature_importance.png - Feature importance
  ✓ feature_importance.csv - Feature importance data
  
""")

print("="*70)