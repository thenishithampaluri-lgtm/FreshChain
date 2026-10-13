from pathlib import Path
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Dataset.csv"

REQUIRED_COLUMNS = [
    "Fruit",
    "Temp",
    "Humid (%)",
    "Light (Fux)",
    "CO2 (pmm)",
    "Class",
]


def load_dataset(path=DATA_PATH):
    """Load and validate the FreshChain fruit dataset."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place Dataset.csv inside the data folder."
        )

    df = pd.read_csv(path)

    # Clean column names
    df.columns = df.columns.astype(str).str.strip()

    # Validate columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Clean categorical columns
    df["Fruit"] = df["Fruit"].astype("string").str.strip()
    df["Class"] = (
        df["Class"]
        .astype("string")
        .str.strip()
        .str.casefold()
    )

    # Convert sensor columns to numeric
    numeric_columns = [
        "Temp",
        "Humid (%)",
        "Light (Fux)",
        "CO2 (pmm)",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    # Remove rows without valid fruit or target labels
    df = df.dropna(subset=["Fruit", "Class"]).copy()
    df = df[
        df["Fruit"].ne("") & df["Class"].ne("")
    ].copy()

    return df


if __name__ == "__main__":
    data = load_dataset()

    print("=" * 50)
    print("FRESHCHAIN DATA LOADER")
    print("=" * 50)

    print("\nDataset shape:", data.shape)

    print("\nFirst five rows:")
    print(data.head())

    print("\nColumn data types:")
    print(data.dtypes)

    print("\nMissing values:")
    print(data.isnull().sum())

    print("\nClass distribution:")
    print(data["Class"].value_counts())

    print("\nDataset loaded successfully!")
