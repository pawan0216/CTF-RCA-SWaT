import pandas as pd
import re
from pathlib import Path

META = Path("data/processed/attack_metadata_processed.csv")
OUT = Path("data/processed/attack_ground_truth.csv")

df = pd.read_csv(META)

df["Start Time"] = pd.to_datetime(df["Start Time"], errors="coerce")
df["End Time Reconstructed"] = pd.to_datetime(
    df["End Time Reconstructed"], errors="coerce"
)

def split_components(value):
    if pd.isna(value):
        return []

    value = str(value)

    if "No Physical Impact" in value:
        return []

    # Handle both comma and semicolon separators
    parts = re.split(r"[,;]", value)

    return [
        p.strip().upper()
        for p in parts
        if p.strip()
    ]

df["Affected_Components"] = df["Attack Point"].apply(
    split_components
)

df["Number_Affected_Components"] = (
    df["Affected_Components"].str.len()
)

df["Attack_Type"] = df["Physical_Impact"].map({
    True: "Physical-impact",
    False: "Cyber-only"
})

df["Valid_Time_Window"] = (
    df["Start Time"].notna()
    & df["End Time Reconstructed"].notna()
)

df["Duration_seconds"] = (
    df["End Time Reconstructed"]
    - df["Start Time"]
).dt.total_seconds()

# Create a readable ground-truth representation
df["Ground_Truth_Root_Cause"] = df[
    "Affected_Components"
].apply(
    lambda x: "|".join(x) if x else "CYBER_ONLY_OR_UNSPECIFIED"
)

columns = [
    "Attack #",
    "Start Time",
    "End Time Reconstructed",
    "Duration_seconds",
    "Attack Point",
    "Affected_Components",
    "Number_Affected_Components",
    "Ground_Truth_Root_Cause",
    "Attack_Type",
    "Valid_Time_Window",
    "Attack",
    "Actual Change",
    "Expected Impact or attacker intent",
    "Unexpected Outcome"
]

df[columns].to_csv(
    OUT,
    index=False
)

print("=" * 80)
print("ATTACK GROUND-TRUTH TABLE CREATED")
print("=" * 80)

print("Total attack records:", len(df))
print("Physical-impact:", (df["Attack_Type"] == "Physical-impact").sum())
print("Cyber-only:", (df["Attack_Type"] == "Cyber-only").sum())
print("Valid time windows:", df["Valid_Time_Window"].sum())
print(
    "Invalid/missing time windows:",
    (~df["Valid_Time_Window"]).sum()
)

print("\nMulti-component attacks:")
print(
    df[df["Number_Affected_Components"] > 1][
        [
            "Attack #",
            "Attack Point",
            "Affected_Components"
        ]
    ].to_string(index=False)
)

print("\nGround-truth table:")
print(
    df[
        [
            "Attack #",
            "Start Time",
            "End Time Reconstructed",
            "Ground_Truth_Root_Cause",
            "Attack_Type"
        ]
    ].to_string(index=False)
)

print("\nSaved:", OUT)
