import pandas as pd
from pathlib import Path

NETWORK_DIR = Path("data/raw/network")

files = sorted(
    NETWORK_DIR.glob("*.csv")
)

for path in files:

    print("\n" + "=" * 80)
    print(path.name)
    print("=" * 80)

    # Read only columns needed for the first audit
    df = pd.read_csv(
        path,
        usecols=[
            "num",
            "date",
            "time",
            "src",
            "dst",
            "proto",
            "appi_name",
            "Modbus_Function_Code",
            "Modbus_Function_Description",
            "Modbus_Transaction_ID",
            "SCADA_Tag",
            "service",
            "s_port",
            "Tag"
        ]
    )

    print("Rows:", len(df))

    print("\nRecord number:")
    print("Minimum:", df["num"].min())
    print("Maximum:", df["num"].max())

    print("\nDate range:")
    print(df["date"].min(), "to", df["date"].max())

    print("\nUnique protocols:")
    print(df["proto"].value_counts(dropna=False).head(20))

    print("\nTop application names:")
    print(df["appi_name"].value_counts(dropna=False).head(20))

    print("\nTop SCADA tags:")
    print(df["SCADA_Tag"].value_counts(dropna=False).head(20))

    print("\nTag values:")
    print(df["Tag"].value_counts(dropna=False).head(20))

    print("\nMissing values:")
    missing = df.isna().sum()
    print(
        missing[missing > 0]
        .sort_values(ascending=False)
        .head(20)
    )

    print("\nDuplicate rows:", df.duplicated().sum())

    # Construct timestamp
    timestamp = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        errors="coerce",
        dayfirst=True
    )

    print("\nTimestamp:")
    print("Missing:", timestamp.isna().sum())
    print("Minimum:", timestamp.min())
    print("Maximum:", timestamp.max())

    print("\nMost common timestamps:")
    print(timestamp.value_counts().head(10))

print("\n" + "=" * 80)
print("NETWORK AUDIT COMPLETE")
print("=" * 80)
