import pandas as pd
from pathlib import Path

RAW = Path("data/raw")

FILES = {
    "attack": RAW / "SWaT_Dataset_Attack_v0.xlsx",
    "normal_v0": RAW / "SWaT_Dataset_Normal_v0.xlsx",
    "normal_v1": RAW / "SWaT_Dataset_Normal_v1.xlsx",
}

def load_swat(path, skiprows=1):
    print(f"\nLoading: {path.name}")

    df = pd.read_excel(
        path,
        sheet_name=0,
        header=0,
        skiprows=skiprows,
        engine="openpyxl"
    )

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    # Clean column names
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    # Remove completely empty rows
    df = df.dropna(axis=0, how="all")

    # Convert timestamp
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"],
            errors="coerce",
            dayfirst=True
        )

    return df


for name, path in FILES.items():

    print("\n" + "=" * 90)
    print(name.upper())
    print("=" * 90)

    df = load_swat(path)

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nTimestamp:")
    if "Timestamp" in df.columns:
        print("Missing timestamps:", df["Timestamp"].isna().sum())
        print("Minimum:", df["Timestamp"].min())
        print("Maximum:", df["Timestamp"].max())
        print("Duplicate timestamps:",
              df["Timestamp"].duplicated().sum())

    print("\nDuplicate complete rows:",
          df.duplicated().sum())

    print("\nMissing values:")
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if len(missing) == 0:
        print("No missing values.")
    else:
        print(missing.sort_values(ascending=False))

    print("\nConstant columns:")

    numeric = df.select_dtypes(include="number")

    constant = numeric.columns[
        numeric.nunique(dropna=False) <= 1
    ]

    print(list(constant))

    print("\nData types:")
    print(df.dtypes.value_counts())

    # Search for possible label columns
    possible_labels = [
        c for c in df.columns
        if any(
            key in c.lower()
            for key in ["label", "attack", "class", "normal"]
        )
    ]

    print("\nPossible label columns:")
    print(possible_labels)

    for col in possible_labels:
        print(f"\nValue counts for {col}:")
        print(df[col].value_counts(dropna=False).head(20))

print("\n" + "=" * 90)
print("AUDIT COMPLETE")
print("=" * 90)
