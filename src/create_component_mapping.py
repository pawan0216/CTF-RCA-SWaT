from pathlib import Path
import pandas as pd

OUTPUT = Path(
    "data/processed/rca_component_mapping.csv"
)

mapping = [
    # Ground truth notation, dataset notation, status
    ("AIT-202", "AIT202", "direct"),
    ("AIT-402", "AIT402", "direct"),
    ("AIT-502", "AIT502", "direct"),
    ("AIT-504", "AIT504", "direct"),

    ("DPIT-301", "DPIT301", "direct"),
    ("DIT-301", "DPIT301", "alias"),

    ("FIT-401", "FIT401", "direct"),
    ("FIT-502", "FIT502", "direct"),

    ("LIT-101", "LIT101", "direct"),
    ("LIT-301", "LIT301", "direct"),
    ("LIT-401", "LIT401", "direct"),

    ("MV-101", "MV101", "direct"),
    ("MV-201", "MV201", "direct"),
    ("MV-302", "MV302", "direct"),
    ("MV-303", "MV303", "direct"),
    ("MV-304", "MV304", "direct"),
    ("MV-504", "", "unavailable"),

    ("P-101", "P101", "direct"),
    ("P-102", "P102", "direct"),
    ("P-201", "P201", "direct"),
    ("P-203", "P203", "direct"),
    ("P-205", "P205", "direct"),
    ("P-302", "P302", "direct"),
    ("P-401", "P401", "direct"),
    ("P-501", "P501", "direct"),
    ("P-602", "P602", "direct"),

    ("UV-401", "UV401", "direct"),
]

df = pd.DataFrame(
    mapping,
    columns=[
        "Ground_Truth_Name",
        "Dataset_Variable",
        "Mapping_Type"
    ]
)

df.to_csv(
    OUTPUT,
    index=False
)

print("=" * 80)
print("RCA COMPONENT MAPPING")
print("=" * 80)

print(df.to_string(index=False))

print("\nSaved:")
print(OUTPUT)

print("\nDirect mappings:",
      (df["Mapping_Type"] == "direct").sum())

print("Aliases:",
      (df["Mapping_Type"] == "alias").sum())

print("Unavailable:",
      (df["Mapping_Type"] == "unavailable").sum())
