from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("data")

PHYSICAL_FILE = BASE / "processed" / "swat_combined_processed.csv"
GROUND_TRUTH_FILE = BASE / "processed" / "attack_ground_truth.csv"
NETWORK_DIR = BASE / "processed" / "network_features"

OUTPUT_FILE = BASE / "processed" / "ctf_rca_synchronized.csv"


def load_network_features():

    files = sorted(NETWORK_DIR.glob("*_1s.csv"))

    print("=" * 80)
    print("LOADING NETWORK FEATURES")
    print("=" * 80)
    print("Network feature files:", len(files))

    frames = []

    for i, p in enumerate(files, 1):

        df = pd.read_csv(
            p,
            parse_dates=["Timestamp"]
        )

        if df.empty:
            continue

        frames.append(df)

        if i % 50 == 0 or i == len(files):
            print(
                f"Loaded {i}/{len(files)} network files"
            )

    if not frames:
        raise RuntimeError("No Network feature files found.")

    network = pd.concat(
        frames,
        ignore_index=True
    )

    print("\nRaw network feature rows:", f"{len(network):,}")

    # Remove source-file column before aggregation
    if "source_file" in network.columns:
        network = network.drop(columns=["source_file"])

    # Aggregate duplicate timestamps across capture partitions
    numeric_cols = [
        c for c in network.columns
        if c != "Timestamp"
    ]

    network = (
        network
        .groupby("Timestamp", as_index=False)[numeric_cols]
        .sum()
    )

    print(
        "Unique network timestamps:",
        f"{len(network):,}"
    )

    print(
        "Network range:",
        network["Timestamp"].min(),
        "->",
        network["Timestamp"].max()
    )

    return network


def load_physical():

    print("\n" + "=" * 80)
    print("LOADING PHYSICAL SWaT DATA")
    print("=" * 80)

    physical = pd.read_csv(
        PHYSICAL_FILE,
        parse_dates=["Timestamp"]
    )

    physical = physical.sort_values(
        "Timestamp"
    ).reset_index(drop=True)

    print(
        "Physical rows:",
        f"{len(physical):,}"
    )

    print(
        "Physical range:",
        physical["Timestamp"].min(),
        "->",
        physical["Timestamp"].max()
    )

    return physical


def load_ground_truth():

    print("\n" + "=" * 80)
    print("LOADING ATTACK GROUND TRUTH")
    print("=" * 80)

    gt = pd.read_csv(
        GROUND_TRUTH_FILE,
        parse_dates=[
            "Start Time",
            "End Time Reconstructed"
        ]
    )

    print(
        "Ground-truth records:",
        len(gt)
    )

    return gt


def attach_attack_labels(df, gt):

    print("\n" + "=" * 80)
    print("ATTACHING ATTACK GROUND TRUTH")
    print("=" * 80)

    df["Attack_ID"] = 0
    df["Attack_Point_GT"] = pd.NA
    df["Ground_Truth_Root_Cause"] = pd.NA
    df["Attack_Type_GT"] = pd.NA

    valid_attacks = gt[
        gt["Valid_Time_Window"] == True
    ].copy()

    print(
        "Valid time-window attacks:",
        len(valid_attacks)
    )

    for _, attack in valid_attacks.iterrows():

        start = attack["Start Time"]
        end = attack["End Time Reconstructed"]

        if pd.isna(start) or pd.isna(end):
            continue

        mask = (
            (df["Timestamp"] >= start) &
            (df["Timestamp"] <= end)
        )

        df.loc[mask, "Attack_ID"] = int(
            attack["Attack #"]
        )

        df.loc[mask, "Attack_Point_GT"] = (
            attack["Attack Point"]
        )

        df.loc[mask, "Ground_Truth_Root_Cause"] = (
            attack["Ground_Truth_Root_Cause"]
        )

        df.loc[mask, "Attack_Type_GT"] = (
            attack["Attack_Type"]
        )

    df["Is_Attack"] = (
        df["Attack_ID"] > 0
    ).astype("int8")

    print(
        "Attack seconds:",
        int(df["Is_Attack"].sum())
    )

    print(
        "Normal seconds:",
        int((df["Is_Attack"] == 0).sum())
    )

    return df


def main():

    print("=" * 80)
    print("CTF-RCA CYBER-PHYSICAL SYNCHRONIZATION")
    print("=" * 80)

    physical = load_physical()

    network = load_network_features()

    # Physical timeline is the master timeline.
    print("\nMerging Network evidence onto physical timeline...")

    df = physical.merge(
        network,
        on="Timestamp",
        how="left"
    )

    network_columns = [
        c for c in network.columns
        if c != "Timestamp"
    ]

    # No network event in a physical second = zero cyber activity.
    for c in network_columns:
        df[c] = df[c].fillna(0)

    print(
        "Rows after physical-network merge:",
        f"{len(df):,}"
    )

    # Attach attack intervals.
    gt = load_ground_truth()

    df = attach_attack_labels(
        df,
        gt
    )

    # Sort and save.
    df = df.sort_values(
        "Timestamp"
    ).reset_index(drop=True)

    print("\nSaving:")
    print(OUTPUT_FILE)

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 80)
    print("SYNCHRONIZATION COMPLETE")
    print("=" * 80)

    print("Final rows:", f"{len(df):,}")
    print("Final columns:", len(df.columns))

    print(
        "Attack rows:",
        int(df["Is_Attack"].sum())
    )

    print(
        "Normal rows:",
        int((df["Is_Attack"] == 0).sum())
    )

    print(
        "Network event rows:",
        int(
            (df["network_event_count"] > 0).sum()
        )
    )

    print("\nOutput:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
