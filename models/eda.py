# FRESHCHAIN: COMPLETE EXPLORATORY DATA ANALYSIS (EDA)
# Fruit Spoilage Prediction | Single-Cell Implementation


import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency, kruskal
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# -------------------- CONFIGURATION --------------------------

DATA_PATH = Path("data/Dataset.csv")

if not DATA_PATH.exists():
    DATA_PATH = Path("Dataset (1)(1).csv")

OUTPUT_DIR = Path("eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 110

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

FEATURES = [
    "Fruit", "Temp", "Humid (%)",
    "Light (Fux)", "CO2 (pmm)"
]
NUMERIC = ["Temp", "Humid (%)", "Light (Fux)", "CO2 (pmm)"]
TARGET = "Class"

# -------------------- LOAD DATA -----------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "CSV not found. Update DATA_PATH to your dataset location."
    )

raw_df = pd.read_csv(DATA_PATH)
df = raw_df.copy()

print("=" * 75)
print("FRESHCHAIN | FRUIT SPOILAGE PREDICTION | COMPLETE EDA")
print("=" * 75)
print(f"\nDataset: {DATA_PATH.resolve()}")
print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")

required = FEATURES + [TARGET]
missing_columns = set(required) - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

print("\n1. DATASET PREVIEW")
display(df.head(10))

print("\n2. DATA TYPES AND INFORMATION")
print(df.dtypes)
df.info()

print("\n3. STATISTICAL SUMMARY")
display(df.describe(include="all").T)

# -------------------- DATA QUALITY --------------------------

print("\n4. DATA QUALITY REPORT")

quality = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing_count": df.isna().sum(),
    "missing_percent": (df.isna().mean() * 100).round(2),
    "unique_values": df.nunique(dropna=False)
})

display(quality)
quality.to_csv(OUTPUT_DIR / "data_quality.csv")

print("Missing cells:", df.isna().sum().sum())
print("Exact duplicate rows:", df.duplicated().sum())

print("\n5. DUPLICATE RECORD INSPECTION")
duplicates = df[df.duplicated(keep=False)]
display(duplicates.head(20))

# -------------------- CLEAN LABELS FOR ANALYSIS ------------

print("\n6. ORIGINAL TARGET LABELS")
display(df[TARGET].value_counts(dropna=False).to_frame("count"))

df[TARGET] = (
    df[TARGET].astype("string").str.strip().str.casefold()
)

print("\nNORMALIZED TARGET LABELS")
display(df[TARGET].value_counts(dropna=False).to_frame("count"))

valid_df = df.dropna(subset=required).copy()

# Normalize fruit names for analysis only
valid_df["Fruit"] = valid_df["Fruit"].astype(str).str.strip()

# -------------------- CLASS DISTRIBUTION --------------------

print("\n7. CLASS BALANCE")

class_summary = pd.DataFrame({
    "count": valid_df[TARGET].value_counts(),
    "percentage": (
        valid_df[TARGET].value_counts(normalize=True) * 100
    ).round(2)
})

display(class_summary)
class_summary.to_csv(OUTPUT_DIR / "class_distribution.csv")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.countplot(
    data=valid_df, x=TARGET,
    order=valid_df[TARGET].value_counts().index,
    hue=TARGET, legend=False, ax=axes[0]
)
axes[0].set_title("Fruit Quality Class Counts")

axes[1].pie(
    class_summary["count"],
    labels=class_summary.index,
    autopct="%1.1f%%",
    startangle=90
)
axes[1].set_title("Target Class Proportions")

plt.tight_layout()
plt.show()

# -------------------- FRUIT ANALYSIS ------------------------

print("\n8. FRUIT CATEGORY ANALYSIS")

fruit_summary = valid_df.groupby("Fruit").agg(
    records=(TARGET, "size"),
    good_count=(TARGET, lambda s: (s == "good").sum()),
    bad_count=(TARGET, lambda s: (s == "bad").sum())
)

fruit_summary["bad_rate_percent"] = (
    fruit_summary["bad_count"] /
    fruit_summary["records"] * 100
).round(2)

display(fruit_summary.sort_values("records", ascending=False))
fruit_summary.to_csv(OUTPUT_DIR / "fruit_summary.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.countplot(
    data=valid_df, y="Fruit",
    order=valid_df["Fruit"].value_counts().index,
    hue="Fruit", legend=False, ax=axes[0]
)
axes[0].set_title("Records by Fruit")

fruit_summary["bad_rate_percent"].sort_values().plot(
    kind="barh", ax=axes[1]
)
axes[1].set_title("Observed Bad-Class Rate by Fruit")
axes[1].set_xlabel("Bad-class rate (%)")

plt.tight_layout()
plt.show()

fruit_class_percent = pd.crosstab(
    valid_df["Fruit"], valid_df[TARGET], normalize="index"
) * 100

fruit_class_percent.plot(kind="bar", stacked=True, figsize=(10, 5))
plt.title("Quality Class Proportions by Fruit")
plt.ylabel("Percentage")
plt.tight_layout()
plt.show()

# -------------------- NUMERICAL DISTRIBUTIONS ---------------

print("\n9. SENSOR FEATURE STATISTICS")
display(valid_df[NUMERIC].describe().T)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

for col, ax in zip(NUMERIC, axes.flat):
    sns.histplot(valid_df[col], kde=True, ax=ax)
    ax.set_title(f"Distribution of {col}")

plt.tight_layout()
plt.show()

# -------------------- OUTLIERS ------------------------------

print("\n10. OUTLIER ANALYSIS")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
outlier_rows = []

for col, ax in zip(NUMERIC, axes.flat):
    sns.boxplot(data=valid_df, x=col, ax=ax)
    ax.set_title(f"Boxplot: {col}")

    values = pd.to_numeric(valid_df[col], errors="coerce")
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((values < lower) | (values > upper)).sum())

    outlier_rows.append({
        "feature": col,
        "lower_bound": lower,
        "upper_bound": upper,
        "potential_outliers": count,
        "outlier_percent": round(count / len(values) * 100, 2)
    })

plt.tight_layout()
plt.show()

outlier_report = pd.DataFrame(outlier_rows)
display(outlier_report)
outlier_report.to_csv(OUTPUT_DIR / "outlier_report.csv", index=False)

# -------------------- CLASS VS SENSOR FEATURES --------------

print("\n11. SENSOR READINGS BY QUALITY CLASS")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

for col, ax in zip(NUMERIC, axes.flat):
    sns.boxplot(
        data=valid_df, x=TARGET, y=col,
        hue=TARGET, legend=False, ax=ax
    )
    ax.set_title(f"{col} by Class")

plt.tight_layout()
plt.show()

display(
    valid_df.groupby(TARGET)[NUMERIC]
    .agg(["mean", "median", "std", "min", "max"])
    .round(3)
)

# -------------------- CORRELATION ---------------------------

print("\n12. SENSOR CORRELATION")

corr = valid_df[NUMERIC].corr()

display(corr.round(3))

plt.figure(figsize=(9, 6))
sns.heatmap(
    corr, annot=True, cmap="RdBu_r",
    center=0, vmin=-1, vmax=1, fmt=".2f"
)
plt.title("Environmental Feature Correlation")
plt.tight_layout()
plt.show()

# -------------------- PAIRWISE RELATIONSHIPS ----------------

print("\n13. PAIRWISE SENSOR RELATIONSHIPS")

sns.pairplot(
    valid_df.sample(min(2000, len(valid_df)), random_state=42),
    vars=NUMERIC,
    hue=TARGET,
    diag_kind="hist",
    corner=True,
    plot_kws={"alpha": 0.5, "s": 18}
)
plt.show()

# -------------------- FRUIT VS SENSOR READINGS ---------------

print("\n14. SENSOR VARIATION BY FRUIT")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for col, ax in zip(NUMERIC, axes.flat):
    sns.boxplot(
        data=valid_df, x="Fruit", y=col,
        hue="Fruit", legend=False, ax=ax
    )
    ax.set_title(f"{col} by Fruit")
    ax.tick_params(axis="x", rotation=25)

plt.tight_layout()
plt.show()

# -------------------- STATISTICAL TESTS ---------------------

print("\n15. STATISTICAL ASSOCIATION TESTS")

contingency = pd.crosstab(valid_df["Fruit"], valid_df[TARGET])

if contingency.shape[0] > 1 and contingency.shape[1] > 1:
    chi2, p, dof, expected = chi2_contingency(contingency)
    n = contingency.to_numpy().sum()
    denominator = n * min(contingency.shape[0] - 1,
                          contingency.shape[1] - 1)
    cramers_v = np.sqrt(chi2 / denominator) if denominator else np.nan

    print("Fruit vs Class")
    print("Chi-square:", round(chi2, 4))
    print("P-value:", p)
    print("Cramer's V:", round(cramers_v, 4))

for col in NUMERIC:
    groups = [
        group[col].dropna().values
        for _, group in valid_df.groupby(TARGET)
        if len(group[col].dropna()) > 0
    ]

    if len(groups) >= 2:
        statistic, p = kruskal(*groups)
        print(f"{col}: H={statistic:.4f}, p-value={p:.6g}")

# -------------------- MUTUAL INFORMATION --------------------

print("\n16. MUTUAL INFORMATION FEATURE ANALYSIS")

X_mi = valid_df[FEATURES].copy()
y_mi = valid_df[TARGET].copy()

X_mi["Fruit"] = X_mi["Fruit"].astype(str)

for col in NUMERIC:
    X_mi[col] = pd.to_numeric(X_mi[col], errors="coerce")

X_mi[NUMERIC] = X_mi[NUMERIC].fillna(
    X_mi[NUMERIC].median()
)

X_encoded = pd.get_dummies(
    X_mi, columns=["Fruit"], dtype=int
)

y_encoded = LabelEncoder().fit_transform(y_mi)

mi_scores = mutual_info_classif(
    X_encoded, y_encoded, random_state=42
)

mi = pd.DataFrame({
    "feature": X_encoded.columns,
    "mutual_information": mi_scores
}).sort_values("mutual_information", ascending=False)

display(mi)
mi.to_csv(OUTPUT_DIR / "mutual_information.csv", index=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=mi.head(15), x="mutual_information", y="feature")
plt.title("Top 15 Encoded Features by Mutual Information")
plt.tight_layout()
plt.show()

# -------------------- DUPLICATE / LABEL CONFLICT CHECK ------

print("\n17. CONFLICTING LABELS FOR IDENTICAL INPUTS")

conflicts = (
    valid_df.groupby(FEATURES, dropna=False)[TARGET]
    .nunique()
)

conflicts = conflicts[conflicts > 1]

print("Conflicting input combinations:", len(conflicts))
display(conflicts.head(20))

# -------------------- FINAL SUMMARY -------------------------

print("\n18. FINAL EDA SUMMARY")
print("=" * 75)

print("Original rows:", len(raw_df))
print("Rows with required fields present:", len(valid_df))
print("Original duplicate rows:", int(raw_df.duplicated().sum()))
print("Missing cells in original data:", int(raw_df.isna().sum().sum()))
print("Fruit categories:", valid_df["Fruit"].nunique())
print("Normalized classes:", valid_df[TARGET].value_counts().to_dict())

print("\nClass proportions:")
print((valid_df[TARGET].value_counts(normalize=True) * 100).round(2))

print("\nEDA reports saved to:", OUTPUT_DIR.resolve())
print("Files generated:")
for file in sorted(OUTPUT_DIR.iterdir()):
    print(" -", file.name)

print("\nEDA COMPLETED.")
print("Review duplicate records, label consistency, and possible")
print("batch/session leakage before selecting a model.")
print("No rows were deleted from the original CSV.")
