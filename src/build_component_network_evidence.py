from pathlib import Path
import pandas as pd
import numpy as np
import re

BASE = Path("data/processed")
RAW_NETWORK = Path("data/raw/Network")

WINDOW_FILE = BASE / "rca_attack_windows.csv"
MAPPING_FILE = BASE / "rca_component_mapping.csv"
OUTPUT_FILE = BASE / "rca_component_network_evidence.csv"

CANDIDATES = [
    "FIT101","LIT101","MV101","P101","P102",
    "AIT201","AIT202","AIT203","FIT201",
    "MV201","P201","P202","P203","P204","P205","P206",
    "DPIT301","FIT301","LIT301",
    "MV301","MV302","MV303","MV304",
    "P301","P302",
    "AIT401","AIT402","FIT401","LIT401",
    "P401","P402","P403","P404","UV401",
    "AIT501","AIT502","AIT503","AIT504",
    "FIT501","FIT502","FIT503","FIT504",
    "P501","P502","PIT501","PIT502","PIT503",
    "FIT601","P601","P602","P603"
]

RAW_COLUMNS = [
    "date",
    "time",
    "src",
    "dst",
    "proto",
    "Modbus_Function_Code",
    "Modbus_Transaction_ID",
    "SCADA_Tag",
    "Tag"
]


def normalize_component(x):

    if pd.isna(x):
        return None

    x = str(x).strip().upper()

    if x == "DIT-301":
        return "DPIT301"

    x = x.replace("-", "")

    return x


def component_from_tag(tag):

    if pd.isna(tag):
        return None

    tag = str(tag).strip().upper()

    # Remove common prefixes
    prefixes = [
        "HMI_",
        "PLC_",
        "SCADA_",
        "RTU_"
    ]

    for prefix in prefixes:
        if tag.startswith(prefix):
            tag = tag[len(prefix):]

    tag = normalize_component(tag)

    if tag in CANDIDATES:
        return tag

    return None


def load_windows():

    df = pd.read_csv(
        WINDOW_FILE,
        low_memory=False
    )

    df["Start_Timestamp"] = pd.to_datetime(
        df["Start_Timestamp"],
        errors="coerce"
    )

    df["End_Timestamp"] = pd.to_datetime(
        df["End_Timestamp"],
        errors="coerce"
    )

    df["Attack_ID"] = pd.to_numeric(
        df["Attack_ID"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "Attack_ID",
            "Start_Timestamp",
            "End_Timestamp"
        ]
    )

    df["Attack_ID"] = (
        df["Attack_ID"].astype(int)
    )

    return df


def load_mapping():

    mapping = pd.read_csv(
        MAPPING_FILE,
        low_memory=False
    )

    result = {}

    for _, row in mapping.iterrows():

        gt = normalize_component(
            row["Ground_Truth_Name"]
        )

        dataset = str(
            row["Dataset_Variable"]
        ).strip()

        mapping_type = str(
            row["Mapping_Type"]
        ).strip()

        if (
            mapping_type != "unavailable"
            and dataset
            and dataset != "nan"
        ):
            result[gt] = dataset

    return result


def get_truth_components(
    truth,
    mapping
):

    if pd.isna(truth):
        return set()

    result = set()

    for item in str(truth).split("|"):

        item = item.strip()

        if not item:
            continue

        if item.upper() == "CYBER_ONLY_OR_UNSPECIFIED":
            continue

        normalized = normalize_component(
            item
        )

        if normalized in mapping:
            result.add(
                mapping[normalized]
            )

        elif normalized in CANDIDATES:
            result.add(normalized)

    return result


def get_network_files_for_attack(
    start,
    end
):

    dates = pd.date_range(
        start.normalize(),
        end.normalize(),
        freq="D"
    )

    selected = []

    for d in dates:

        prefix = d.strftime(
            "%Y-%m-%d"
        )

        selected.extend(
            sorted(
                RAW_NETWORK.glob(
                    f"{prefix}_*.csv"
                )
            )
        )

    return selected


def process_attack(
    attack_id,
    start,
    end,
    truth,
    mapping
):

    files = get_network_files_for_attack(
        start,
        end
    )

    print(
        f"  Attack {attack_id}: "
        f"{len(files)} raw network files"
    )

    component_counts = {
        c: {
            "events": 0,
            "modbus": 0,
            "src": set(),
            "dst": set(),
            "transactions": set()
        }
        for c in CANDIDATES
    }

    total_events = 0

    files_used = 0

    for file_index, path in enumerate(files, 1):

        try:

            reader = pd.read_csv(
                path,
                usecols=lambda c: c in RAW_COLUMNS,
                chunksize=100_000,
                low_memory=False
            )

            file_had_events = False

            for chunk in reader:

                if "date" not in chunk.columns:
                    continue

                if "time" not in chunk.columns:
                    continue

                timestamps = pd.to_datetime(
                    chunk["date"].astype(str)
                    + " "
                    + chunk["time"].astype(str),
                    errors="coerce",
                    dayfirst=True
                )

                chunk["Timestamp"] = timestamps

                mask = (
                    (chunk["Timestamp"] >= start)
                    &
                    (chunk["Timestamp"] <= end)
                )

                x = chunk.loc[mask]

                if x.empty:
                    continue

                file_had_events = True

                total_events += len(x)

                x["Mapped_Component"] = (
                    x["SCADA_Tag"]
                    .apply(component_from_tag)
                )

                for component, g in x.groupby(
                    "Mapped_Component"
                ):

                    if component not in component_counts:
                        continue

                    info = component_counts[
                        component
                    ]

                    info["events"] += len(g)

                    if "Modbus_Function_Code" in g.columns:

                        info["modbus"] += int(
                            g[
                                "Modbus_Function_Code"
                            ].notna().sum()
                        )

                    if "src" in g.columns:

                        info["src"].update(
                            g["src"]
                            .dropna()
                            .astype(str)
                            .tolist()
                        )

                    if "dst" in g.columns:

                        info["dst"].update(
                            g["dst"]
                            .dropna()
                            .astype(str)
                            .tolist()
                        )

                    if (
                        "Modbus_Transaction_ID"
                        in g.columns
                    ):

                        info["transactions"].update(
                            g[
                                "Modbus_Transaction_ID"
                            ]
                            .dropna()
                            .astype(str)
                            .tolist()
                        )

            if file_had_events:
                files_used += 1

        except Exception as e:

            print(
                f"    WARNING: "
                f"{path.name}: {e}"
            )

    truth_components = get_truth_components(
        truth,
        mapping
    )

    rows = []

    for component in CANDIDATES:

        info = component_counts[
            component
        ]

        events = info["events"]

        if total_events > 0:
            share = (
                events /
                total_events
            )
        else:
            share = 0.0

        rows.append({

            "Attack_ID": attack_id,

            "Component": component,

            "Network_Events":
                events,

            "Total_Attack_Network_Events":
                total_events,

            "Network_Event_Share":
                share,

            "Unique_Sources":
                len(info["src"]),

            "Unique_Destinations":
                len(info["dst"]),

            "Unique_Modbus_Transactions":
                len(info["transactions"]),

            "Modbus_Events":
                info["modbus"],

            "Files_Used":
                files_used,

            "Ground_Truth_Component":
                int(
                    component in truth_components
                )
        })

    return rows


def main():

    print("=" * 80)
    print("COMPONENT-SPECIFIC RAW NETWORK EVIDENCE")
    print("=" * 80)

    print(
        "Raw network directory:",
        RAW_NETWORK
    )

    windows = load_windows()

    mapping = load_mapping()

    print(
        "Attack episodes:",
        len(windows)
    )

    print(
        "Candidate components:",
        len(CANDIDATES)
    )

    print(
        "Ground-truth mappings:",
        len(mapping)
    )

    all_rows = []

    for i, row in windows.iterrows():

        attack_id = int(
            row["Attack_ID"]
        )

        print(
            f"\n[{i + 1}/{len(windows)}] "
            f"Attack {attack_id}"
        )

        result = process_attack(
            attack_id,
            row["Start_Timestamp"],
            row["End_Timestamp"],
            row["Ground_Truth_Root_Cause"],
            mapping
        )

        all_rows.extend(result)

    df = pd.DataFrame(
        all_rows
    )

    df = df.sort_values(
        [
            "Attack_ID",
            "Network_Event_Share"
        ],
        ascending=[
            True,
            False
        ]
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 80)
    print("COMPONENT NETWORK EVIDENCE COMPLETE")
    print("=" * 80)

    print(
        "Rows:",
        f"{len(df):,}"
    )

    print(
        "Attacks:",
        df["Attack_ID"].nunique()
    )

    print(
        "Components:",
        df["Component"].nunique()
    )

    print(
        "Positive ground-truth rows:",
        int(
            df[
                "Ground_Truth_Component"
            ].sum()
        )
    )

    print(
        "\nOutput:",
        OUTPUT_FILE
    )

    print("\nTop network associations:")

    print(
        df[
            [
                "Attack_ID",
                "Component",
                "Network_Events",
                "Network_Event_Share",
                "Modbus_Events",
                "Ground_Truth_Component"
            ]
        ]
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
