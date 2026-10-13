from pathlib import Path
import pandas as pd

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


def load_data(path=DATA_PATH):
    """Load and validate the FreshChain dataset."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df["Fruit"] = df["Fruit"].astype("string").str.strip()
    df["Class"] = (
        df["Class"].astype("string").str.strip().str.casefold()
    )

    df = df.dropna(subset=["Fruit", "Class"]).copy()
    df = df[
        df["Fruit"].ne("") & df["Class"].ne("")
    ].copy()

    return df


if __name__ == "__main__":
    data = load_data()
    print("Dataset loaded successfully!")
    print("Shape:", data.shape)
    print("\nColumns:")
    print(data.columns.tolist())
    print("\nTarget distribution:")
    print(data["Class"].value_counts())
