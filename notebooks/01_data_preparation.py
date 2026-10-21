
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

# Find the project folder automatically.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Load the original Kaggle dataset.
csv_path = DATA_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv"

if not csv_path.exists():
    raise FileNotFoundError(
        f"Dataset not found: {csv_path}\n"
        "Place the downloaded CSV inside the project's data folder."
    )

df = pd.read_csv(csv_path)

print("\n--- DATASET OVERVIEW ---")
print("Shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nMissing values:")
print(df.isnull().sum())
print("\nAttrition distribution:")
print(df["Attrition"].value_counts())

# Preserve employee identifiers separately.
# They will NOT be used as model features.
if "EmployeeNumber" in df.columns:
    employee_ids = df["EmployeeNumber"].copy()
else:
    raise ValueError("EmployeeNumber column not found in the original dataset.")

# Remove non-predictive or constant columns.
columns_to_drop = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

df = df.drop(columns=columns_to_drop, errors="ignore")

# Convert target: No = 0, Yes = 1.
df["Attrition"] = df["Attrition"].map({
    "No": 0,
    "Yes": 1
})

# Separate predictive features and target.
X = df.drop(columns=["Attrition"])
y = df["Attrition"]

# Split features, target, and employee IDs together.
# This guarantees that each ID stays aligned with its employee.
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X,
    y,
    employee_ids,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Save model features only.
X_train.to_csv(DATA_DIR / "X_train.csv", index=False)
X_test.to_csv(DATA_DIR / "X_test.csv", index=False)

# Save targets.
y_train.to_frame("Attrition").to_csv(
    DATA_DIR / "y_train.csv", index=False
)
y_test.to_frame("Attrition").to_csv(
    DATA_DIR / "y_test.csv", index=False
)

# Save employee IDs separately.
ids_train.rename("EmployeeNumber").to_frame().reset_index(
    drop=True
).to_csv(DATA_DIR / "employee_numbers_train.csv", index=False)

ids_test.rename("EmployeeNumber").to_frame().reset_index(
    drop=True
).to_csv(DATA_DIR / "employee_numbers_test.csv", index=False)

# Create a convenient CSV for batch predictions.
# EmployeeNumber is included for display, not as a model feature.
X_test_with_id = X_test.reset_index(drop=True).copy()
X_test_with_id.insert(
    0,
    "EmployeeNumber",
    ids_test.reset_index(drop=True).values
)

X_test_with_id.to_csv(
    DATA_DIR / "X_test_with_employee_number.csv",
    index=False
)

# Plot the original attrition distribution.
sns.set_theme(style="whitegrid")
plt.figure(figsize=(7, 5))
sns.countplot(data=pd.read_csv(csv_path), x="Attrition")
plt.title("Employee Attrition Distribution")
plt.tight_layout()
plt.savefig(DATA_DIR / "eda_analysis.png", dpi=150)
plt.close()

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Predictive features:", len(X_train.columns))
print("Employee IDs preserved:", len(ids_train) + len(ids_test))
print("\nData preparation completed successfully!")
print("Next: test batch prediction using X_test_with_employee_number.csv")
