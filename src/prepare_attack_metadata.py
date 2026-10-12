import pandas as pd
from pathlib import Path

INPUT = Path("data/raw/List_of_attacks_Final.xlsx")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading official iTRUST attack metadata...")

df = pd.read_excel(INPUT, engine="openpyxl")

# Clean column names
df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)

print("\nOriginal columns:")
print(df.columns.tolist())

# ---------------------------------------------------------
# Convert Start Time
# ---------------------------------------------------------

df["Start Time"] = pd.to_datetime(
    df["Start Time"],
    errors="coerce"
)

# ---------------------------------------------------------
# Reconstruct End Time
# ---------------------------------------------------------

def combine_end_time(row):

    start = row["Start Time"]
    end = row["End Time"]

    if pd.isna(start) or pd.isna(end):
        return pd.NaT

    # End Time is usually stored as datetime.time
    if hasattr(end, "hour"):
        return pd.Timestamp(
            year=start.year,
            month=start.month,
            day=start.day,
            hour=end.hour,
            minute=end.minute,
            second=end.second
        )

    # Fallback if stored as string
    try:
        parsed = pd.to_datetime(str(end))

        return pd.Timestamp(
            year=start.year,
            month=start.month,
            day=start.day,
            hour=parsed.hour,
            minute=parsed.minute,
            second=parsed.second
        )

    except Exception:
        return pd.NaT


df["End Time Reconstructed"] = df.apply(
    combine_end_time,
    axis=1
)

# ---------------------------------------------------------
# Calculate duration
# ---------------------------------------------------------

df["Duration_seconds"] = (
    df["End Time Reconstructed"]
    - df["Start Time"]
).dt.total_seconds()

# ---------------------------------------------------------
# Identify physical-impact attacks
# ---------------------------------------------------------

df["Physical_Impact"] = (
    ~df["Attack Point"]
    .astype(str)
    .str.contains(
        "No Physical Impact",
        case=False,
        na=False
    )
)

# ---------------------------------------------------------
# Clean attack-point names
# ---------------------------------------------------------

df["Attack Point"] = (
    df["Attack Point"]
    .astype(str)
    .str.strip()
)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

output = OUTPUT_DIR / "attack_metadata_processed.csv"

df.to_csv(
    output,
    index=False
)

print("\n" + "=" * 70)
print("ATTACK METADATA PROCESSING COMPLETE")
print("=" * 70)

print("Number of attack records:", len(df))
print("Physical-impact records:",
      df["Physical_Impact"].sum())

print("No-physical-impact records:",
      (~df["Physical_Impact"]).sum())

print("\nAttack-point distribution:")
print(
    df["Attack Point"]
    .value_counts(dropna=False)
)

print("\nFirst records:")
print(
    df[
        [
            "Attack #",
            "Start Time",
            "End Time Reconstructed",
            "Attack Point",
            "Duration_seconds",
            "Physical_Impact"
        ]
    ].head(10).to_string(index=False)
)

print("\nSaved:")
print(output)
