from pathlib import Path
import pandas as pd
import numpy as np
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

print("\n" + "="*70)
print("STEP 1: DATA PREPARATION WITH FEATURE ENGINEERING")
print("="*70)

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

# ============================================================================
# FEATURE ENGINEERING - ADD BEFORE DROPPING COLUMNS
# ============================================================================

print("\n" + "="*70)
print("FEATURE ENGINEERING - CREATING 15 NEW FEATURES")
print("="*70)

# 1. WORK ENGAGEMENT FEATURES
print("\n[1/5] Adding Work Engagement Features...")

df['SatisfactionToSalaryRatio'] = df['JobSatisfaction'] / (df['MonthlyIncome'] + 1)
print("  ✓ SatisfactionToSalaryRatio - Captures satisfaction-to-income ratio")

df['YearsSincePromotionRatio'] = df['YearsSinceLastPromotion'] / (df['TotalWorkingYears'] + 1)
print("  ✓ YearsSincePromotionRatio - Time without promotion relative to total experience")

df['WorkLifeScore'] = df['WorkLifeBalance'] * df['YearsAtCompany']
print("  ✓ WorkLifeScore - Work-life balance impact over tenure")

df['CareerProgressRatio'] = df['JobLevel'] / (df['YearsAtCompany'] + 1)
print("  ✓ CareerProgressRatio - Career progression speed")


# 2. STABILITY & COMMITMENT FEATURES
print("\n[2/5] Adding Stability & Commitment Features...")

df['JobHoppingIndex'] = df['NumCompaniesWorked'] / (df['TotalWorkingYears'] + 1)
print("  ✓ JobHoppingIndex - Frequency of job changes indicator")

df['IsNewToCompany'] = (df['YearsAtCompany'] < 2).astype(int)
print("  ✓ IsNewToCompany - Binary flag for employees < 2 years tenure")

df['LongTermEmployee'] = (df['YearsAtCompany'] > 10).astype(int)
print("  ✓ LongTermEmployee - Binary flag for employees > 10 years tenure")

df['ManagerTenureRatio'] = df['YearsWithCurrManager'] / (df['YearsAtCompany'] + 1)
print("  ✓ ManagerTenureRatio - Stability with current manager")


# 3. COMPENSATION & VALUE FEATURES
print("\n[3/5] Adding Compensation & Value Features...")

df['IsHighEarner'] = (df['MonthlyIncome'] > df['MonthlyIncome'].median()).astype(int)
print("  ✓ IsHighEarner - Binary flag for above-median earners")

df['IncomeBelowRoleAverage'] = (
    df['MonthlyIncome'] < df.groupby('JobRole')['MonthlyIncome'].transform('mean')
).astype(int)
print("  ✓ IncomeBelowRoleAverage - Underpaid relative to job role peers")


# 4. SATISFACTION COMPOSITE FEATURES
print("\n[4/5] Adding Satisfaction Composite Features...")

df['OverallSatisfaction'] = (
    df['JobSatisfaction'] +
    df['EnvironmentSatisfaction'] +
    df['WorkLifeBalance'] +
    df['JobInvolvement']
) / 4
print("  ✓ OverallSatisfaction - Composite satisfaction metric")

df['SatisfactionVariance'] = df[[
    'JobSatisfaction',
    'EnvironmentSatisfaction',
    'WorkLifeBalance'
]].std(axis=1)
print("  ✓ SatisfactionVariance - Consistency of satisfaction metrics")

df['IsInDangerZone'] = (
    (df['JobSatisfaction'] <= 2) &
    (df['WorkLifeBalance'] <= 2)
).astype(int)
print("  ✓ IsInDangerZone - Critical risk indicator (low satisfaction + bad work-life)")


# 5. DEMOGRAPHIC INTERACTION FEATURES
print("\n[5/5] Adding Demographic Interaction Features...")

df['IsUnderpaidInDepartment'] = (
    df['MonthlyIncome'] < df.groupby('Department')['MonthlyIncome'].transform('mean')
).astype(int)
print("  ✓ IsUnderpaidInDepartment - Underpaid relative to department")

df['DistanceAboveMedian'] = (df['DistanceFromHome'] > df['DistanceFromHome'].median()).astype(int)
print("  ✓ DistanceAboveMedian - Long commute indicator")

# ============================================================================
# FEATURE SUMMARY
# ============================================================================

print("\n" + "="*70)
print("FEATURE ENGINEERING COMPLETE")
print("="*70)

original_feature_count = len(df.columns) - 2
new_features_added = 15
total_features = original_feature_count + new_features_added

print(f"\nOriginal features: {original_feature_count}")
print(f"New features added: {new_features_added}")
print(f"Total features now: {total_features}")

print("\nNew engineered features:")
new_features = [
    'SatisfactionToSalaryRatio',
    'YearsSincePromotionRatio',
    'WorkLifeScore',
    'CareerProgressRatio',
    'JobHoppingIndex',
    'IsNewToCompany',
    'LongTermEmployee',
    'ManagerTenureRatio',
    'IsHighEarner',
    'IncomeBelowRoleAverage',
    'OverallSatisfaction',
    'SatisfactionVariance',
    'IsInDangerZone',
    'IsUnderpaidInDepartment',
    'DistanceAboveMedian'
]

for i, feature in enumerate(new_features, 1):
    print(f"  {i:2d}. {feature}")

# ============================================================================
# DATA PREPARATION
# ============================================================================

print("\n" + "="*70)
print("DATA PREPARATION")
print("="*70)

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

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

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

print(f"\nTraining rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"Predictive features: {X_train.shape[1]}")
print(f"Employee IDs preserved: {len(ids_train) + len(ids_test)}")

# ============================================================================
# SAVE FILES
# ============================================================================

print("\n" + "="*70)
print("SAVING FILES")
print("="*70)

# Save model features only.
X_train.to_csv(DATA_DIR / "X_train.csv", index=False)
X_test.to_csv(DATA_DIR / "X_test.csv", index=False)
print("\n✓ X_train.csv - Training features")
print("✓ X_test.csv - Test features")

# Save targets.
y_train.to_frame("Attrition").to_csv(DATA_DIR / "y_train.csv", index=False)
y_test.to_frame("Attrition").to_csv(DATA_DIR / "y_test.csv", index=False)
print("✓ y_train.csv - Training target")
print("✓ y_test.csv - Test target")

# Save employee IDs separately.
ids_train.rename("EmployeeNumber").to_frame().reset_index(drop=True).to_csv(
    DATA_DIR / "employee_numbers_train.csv", index=False
)
ids_test.rename("EmployeeNumber").to_frame().reset_index(drop=True).to_csv(
    DATA_DIR / "employee_numbers_test.csv", index=False
)
print("✓ employee_numbers_train.csv")
print("✓ employee_numbers_test.csv")

# Create a convenient CSV for batch predictions.
X_test_with_id = X_test.reset_index(drop=True).copy()
X_test_with_id.insert(0, "EmployeeNumber", ids_test.reset_index(drop=True).values)
X_test_with_id.to_csv(DATA_DIR / "X_test_with_employee_number.csv", index=False)
print("✓ X_test_with_employee_number.csv - For batch predictions")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# Load original data for visualization
original_df = pd.read_csv(csv_path)

# Plot 1: Attrition Distribution
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Attrition counts
sns.countplot(data=original_df, x="Attrition", ax=axes[0, 0], palette="Set2")
axes[0, 0].set_title("Employee Attrition Distribution")
axes[0, 0].set_ylabel("Count")

# Attrition by Department
sns.countplot(data=original_df, x="Department", hue="Attrition", ax=axes[0, 1], palette="Set2")
axes[0, 1].set_title("Attrition by Department")
axes[0, 1].set_ylabel("Count")

# Income distribution by Attrition
original_df.boxplot(column='MonthlyIncome', by='Attrition', ax=axes[1, 0])
axes[1, 0].set_title("Monthly Income by Attrition Status")
axes[1, 0].set_xlabel("Attrition")
axes[1, 0].set_ylabel("Monthly Income")

# Years at Company by Attrition
original_df.boxplot(column='YearsAtCompany', by='Attrition', ax=axes[1, 1])
axes[1, 1].set_title("Years at Company by Attrition Status")
axes[1, 1].set_xlabel("Attrition")
axes[1, 1].set_ylabel("Years at Company")

plt.tight_layout()
plt.savefig(DATA_DIR / "eda_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ eda_analysis.png - EDA visualizations created")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("DATA PREPARATION COMPLETE!")
print("="*70)

print(f"""
SUMMARY:
  Dataset: {len(original_df)} employees
  Features: {total_features} (original: {original_feature_count}, engineered: {new_features_added})
  Training set: {len(X_train)} samples (70%)
  Test set: {len(X_test)} samples (30%)

FILES CREATED:
  ✓ X_train.csv - Training features ({len(X_train)} rows, {X_train.shape[1]} columns)
  ✓ X_test.csv - Test features ({len(X_test)} rows, {X_test.shape[1]} columns)
  ✓ y_train.csv - Training labels
  ✓ y_test.csv - Test labels
  ✓ X_test_with_employee_number.csv - For batch predictions
  ✓ eda_analysis.png - Visualizations
""")

print("="*70)