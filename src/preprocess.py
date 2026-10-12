import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_swat(filepath):
    """
    Load official SWaT Excel file while handling
    the two-row header structure.
    """

    df = pd.read_excel(
        filepath,
        sheet_name=0,
        header=0,
        skiprows=1,
        engine="openpyxl"
    )

    # Remove completely empty rows/columns
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    # Clean column names
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Remove accidental unnamed columns
    df = df.loc[
        :,
        ~df.columns.str.lower().str.startswith("unnamed")
    ]

    # Parse timestamp
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"],
            errors="coerce",
            dayfirst=True
        )

    return df


def clean_data(df, name):

    print(f"\n{name}")
    print("-" * 60)

    original_rows = len(df)

    # Remove rows without timestamp
    if "Timestamp" in df.columns:
        df = df.dropna(subset=["Timestamp"])

    # Sort chronologically
    if "Timestamp" in df.columns:
        df = df.sort_values("Timestamp")

    # Remove duplicate rows
    duplicate_rows = df.duplicated().sum()
    df = df.drop_duplicates()

    # Set timestamp as index temporarily
    df = df.set_index("Timestamp")

    # Convert non-numeric values to NaN
    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Linear interpolation
    missing_before = int(df.isna().sum().sum())

    df = df.interpolate(
        method="linear",
        limit_direction="both"
    )

    # Remaining missing values
    df = df.ffill().bfill()

    missing_after = int(df.isna().sum().sum())

    # Return timestamp to column
    df = df.reset_index()

    print("Original rows:", original_rows)
    print("Rows after cleaning:", len(df))
    print("Duplicate rows removed:", duplicate_rows)
    print("Missing values before:", missing_before)
    print("Missing values after:", missing_after)

    return df


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

normal_path = RAW_DIR / "SWaT_Dataset_Normal_v1.xlsx"
attack_path = RAW_DIR / "SWaT_Dataset_Attack_v0.xlsx"

print("=" * 70)
print("CTF-RCA SWaT PREPROCESSING")
print("=" * 70)

normal = load_swat(normal_path)
attack = load_swat(attack_path)

print("\nRaw Normal shape:", normal.shape)
print("Raw Attack shape:", attack.shape)


# ---------------------------------------------------------
# Clean
# ---------------------------------------------------------

normal = clean_data(normal, "NORMAL")
attack = clean_data(attack, "ATTACK")


# ---------------------------------------------------------
# Add labels
# ---------------------------------------------------------

normal["Label"] = 0
attack["Label"] = 1

normal["Dataset"] = "Normal"
attack["Dataset"] = "Attack"


# ---------------------------------------------------------
# Combine
# ---------------------------------------------------------

combined = pd.concat(
    [normal, attack],
    ignore_index=True
)

combined = combined.sort_values(
    "Timestamp"
).reset_index(drop=True)


# ---------------------------------------------------------
# Identify numeric features
# ---------------------------------------------------------

exclude = [
    "Timestamp",
    "Label",
    "Dataset"
]

features = [
    c for c in combined.columns
    if c not in exclude
    and pd.api.types.is_numeric_dtype(
        combined[c]
    )
]

print("\nNumber of numerical features:", len(features))


# ---------------------------------------------------------
# Min-Max normalization
# ---------------------------------------------------------

scaler = MinMaxScaler()

# IMPORTANT:
# Fit only on normal training/reference data.
scaler.fit(normal[features])

combined[features] = scaler.transform(
    combined[features]
)


# ---------------------------------------------------------
# Save scaler parameters
# ---------------------------------------------------------

scaler_data = pd.DataFrame({
    "feature": features,
    "min": scaler.data_min_,
    "max": scaler.data_max_,
    "scale": scaler.scale_
})

scaler_data.to_csv(
    OUT_DIR / "scaler_parameters.csv",
    index=False
)


# ---------------------------------------------------------
# Save processed datasets
# ---------------------------------------------------------

normal_processed = combined[
    combined["Dataset"] == "Normal"
].copy()

attack_processed = combined[
    combined["Dataset"] == "Attack"
].copy()

normal_processed.to_csv(
    OUT_DIR / "normal_processed.csv",
    index=False
)

attack_processed.to_csv(
    OUT_DIR / "attack_processed.csv",
    index=False
)

combined.to_csv(
    OUT_DIR / "swat_combined_processed.csv",
    index=False
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

summary = {
    "normal_rows": len(normal_processed),
    "attack_rows": len(attack_processed),
    "total_rows": len(combined),
    "numeric_features": len(features),
    "missing_after_processing": int(
        combined[features].isna().sum().sum()
    )
}

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

for key, value in summary.items():
    print(f"{key}: {value}")

print("\nSaved:")
print("data/processed/normal_processed.csv")
print("data/processed/attack_processed.csv")
print("data/processed/swat_combined_processed.csv")
print("data/processed/scaler_parameters.csv")
