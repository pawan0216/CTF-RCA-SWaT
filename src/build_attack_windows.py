from pathlib import Path
import pandas as pd

BASE = Path("data/processed")

GROUND_TRUTH = BASE / "attack_ground_truth.csv"
OUTPUT = BASE / "rca_attack_windows.csv"

gt = pd.read_csv(
    GROUND_TRUTH,
    low_memory=False
)

gt["Attack_num"] = pd.to_numeric(
    gt["Attack #"],
    errors="coerce"
)

gt["Start Time"] = pd.to_datetime(
    gt["Start Time"],
    errors="coerce"
)

gt["End Time Reconstructed"] = pd.to_datetime(
    gt["End Time Reconstructed"],
    errors="coerce"
)

valid = gt[
    gt["Valid_Time_Window"].astype(str).str.lower() == "true"
].copy()

valid = valid.sort_values(
    "Attack_num"
).reset_index(drop=True)

out = valid[
    [
        "Attack_num",
        "Start Time",
        "End Time Reconstructed",
        "Duration_seconds",
        "Attack Point",
        "Ground_Truth_Root_Cause",
        "Attack_Type",
        "Valid_Time_Window"
    ]
].copy()

out = out.rename(
    columns={
        "Attack_num": "Attack_ID",
        "Start Time": "Start_Timestamp",
        "End Time Reconstructed": "End_Timestamp",
        "Attack Point": "Attack_Point",
        "Ground_Truth_Root_Cause": "Ground_Truth_Root_Cause",
        "Attack_Type": "Attack_Type"
    }
)

out["Ground_Truth_Components"] = (
    out["Ground_Truth_Root_Cause"]
    .fillna("")
    .astype(str)
    .str.split("|")
)

out.to_csv(
    OUTPUT,
    index=False
)

print("=" * 80)
print("ATTACK WINDOW TABLE")
print("=" * 80)

print("Valid attacks:", len(out))
print("Output:", OUTPUT)

print("\nAttack windows:")
print(
    out[
        [
            "Attack_ID",
            "Start_Timestamp",
            "End_Timestamp",
            "Duration_seconds",
            "Ground_Truth_Root_Cause"
        ]
    ].to_string(index=False)
)
