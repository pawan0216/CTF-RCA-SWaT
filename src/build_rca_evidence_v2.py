from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("data/processed")

DATA_FILE = BASE / "ctf_rca_synchronized.csv"
WINDOW_FILE = BASE / "rca_attack_windows.csv"
MAPPING_FILE = BASE / "rca_component_mapping.csv"
OUTPUT_FILE = BASE / "rca_evidence_matrix_v2.csv"

CANDIDATES = [
    "FIT101", "LIT101", "MV101", "P101", "P102",
    "AIT201", "AIT202", "AIT203", "FIT201",
    "MV201", "P201", "P202", "P203", "P204", "P205", "P206",
    "DPIT301", "FIT301", "LIT301",
    "MV301", "MV302", "MV303", "MV304",
    "P301", "P302",
    "AIT401", "AIT402", "FIT401", "LIT401",
    "P401", "P402", "P403", "P404", "UV401",
    "AIT501", "AIT502", "AIT503", "AIT504",
    "FIT501", "FIT502", "FIT503", "FIT504",
    "P501", "P502", "PIT501", "PIT502", "PIT503",
    "FIT601", "P601", "P602", "P603"
]

NETWORK_COLUMNS = [
    "network_event_count",
    "unique_src_count",
    "unique_dst_count",
    "unique_scada_tags",
    "unique_modbus_functions",
    "unique_transactions",
    "tcp_count",
    "udp_count",
    "modbus_event_count"
]

PRE_ATTACK_SECONDS = 300


def load_windows():
    windows = pd.read_csv(
        WINDOW_FILE,
        low_memory=False
    )

    windows["Start_Timestamp"] = pd.to_datetime(
        windows["Start_Timestamp"],
        errors="coerce"
    )

    windows["End_Timestamp"] = pd.to_datetime(
        windows["End_Timestamp"],
        errors="coerce"
    )

    windows["Attack_ID"] = pd.to_numeric(
        windows["Attack_ID"],
        errors="coerce"
    )

    windows = windows.dropna(
        subset=["Attack_ID"]
    )

    windows["Attack_ID"] = (
        windows["Attack_ID"]
        .astype(int)
    )

    return windows


def load_mapping():

    mapping = pd.read_csv(
        MAPPING_FILE,
        low_memory=False
    )

    truth_to_dataset = {}

    for _, row in mapping.iterrows():

        gt_name = str(
            row["Ground_Truth_Name"]
        ).strip().upper()

        dataset_name = str(
            row["Dataset_Variable"]
        ).strip()

        mapping_type = str(
            row["Mapping_Type"]
        ).strip()

        if mapping_type != "unavailable" and dataset_name:
            truth_to_dataset[gt_name] = dataset_name

    return truth_to_dataset


def calculate_global_baseline():

    required = [
        "Timestamp",
        "Is_Attack"
    ] + CANDIDATES

    sums = {
        c: 0.0 for c in CANDIDATES
    }

    sums_sq = {
        c: 0.0 for c in CANDIDATES
    }

    counts = {
        c: 0 for c in CANDIDATES
    }

    reader = pd.read_csv(
        DATA_FILE,
        usecols=required,
        chunksize=100_000,
        low_memory=False
    )

    for chunk in reader:

        normal = chunk[
            chunk["Is_Attack"] == 0
        ]

        if normal.empty:
            continue

        for c in CANDIDATES:

            x = pd.to_numeric(
                normal[c],
                errors="coerce"
            ).dropna()

            if x.empty:
                continue

            sums[c] += float(x.sum())
            sums_sq[c] += float(
                (x * x).sum()
            )
            counts[c] += len(x)

    baseline = {}

    for c in CANDIDATES:

        n = counts[c]

        if n < 2:
            baseline[c] = (0.0, 1.0)
            continue

        mean = sums[c] / n

        variance = (
            sums_sq[c] / n
            - mean * mean
        )

        std = np.sqrt(
            max(variance, 1e-12)
        )

        if std < 1e-8:
            std = 1.0

        baseline[c] = (
            mean,
            std
        )

    return baseline


def collect_attack_data(windows):

    required = (
        ["Timestamp"]
        + CANDIDATES
        + NETWORK_COLUMNS
    )

    attack_data = {}

    for _, w in windows.iterrows():

        attack_id = int(
            w["Attack_ID"]
        )

        start = w["Start_Timestamp"]
        end = w["End_Timestamp"]

        pre_start = (
            start -
            pd.Timedelta(
                seconds=PRE_ATTACK_SECONDS
            )
        )

        pieces = []

        reader = pd.read_csv(
            DATA_FILE,
            usecols=required,
            parse_dates=["Timestamp"],
            chunksize=100_000,
            low_memory=False
        )

        for chunk in reader:

            mask = (
                (chunk["Timestamp"] >= pre_start)
                &
                (chunk["Timestamp"] <= end)
            )

            selected = chunk.loc[mask]

            if not selected.empty:
                pieces.append(
                    selected.copy()
                )

        if pieces:

            attack_data[attack_id] = pd.concat(
                pieces,
                ignore_index=True
            )

    return attack_data


def make_truth_components(
    truth,
    truth_to_dataset
):

    if pd.isna(truth):
        return set()

    components = set()

    for item in str(truth).split("|"):

        item = item.strip().upper()

        if not item:
            continue

        if item == "CYBER_ONLY_OR_UNSPECIFIED":
            continue

        mapped = truth_to_dataset.get(
            item
        )

        if mapped:
            components.add(mapped)

    return components


def extract_evidence(
    windows,
    attack_data,
    baseline,
    truth_to_dataset
):

    rows = []

    for _, w in windows.iterrows():

        attack_id = int(
            w["Attack_ID"]
        )

        data = attack_data.get(
            attack_id
        )

        if data is None or data.empty:
            continue

        start = w["Start_Timestamp"]
        end = w["End_Timestamp"]

        pre_start = (
            start -
            pd.Timedelta(
                seconds=PRE_ATTACK_SECONDS
            )
        )

        pre = data[
            (data["Timestamp"] >= pre_start)
            &
            (data["Timestamp"] < start)
        ]

        attack = data[
            (data["Timestamp"] >= start)
            &
            (data["Timestamp"] <= end)
        ]

        if attack.empty:
            continue

        truth_components = make_truth_components(
            w["Ground_Truth_Root_Cause"],
            truth_to_dataset
        )

        network_total = float(
            attack[
                "network_event_count"
            ].sum()
        )

        modbus_total = float(
            attack[
                "modbus_event_count"
            ].sum()
        )

        for component in CANDIDATES:

            attack_x = pd.to_numeric(
                attack[component],
                errors="coerce"
            ).dropna()

            pre_x = pd.to_numeric(
                pre[component],
                errors="coerce"
            ).dropna()

            if attack_x.empty:
                continue

            baseline_mean, baseline_std = (
                baseline[component]
            )

            # Attack-period anomaly.
            attack_mean = float(
                attack_x.mean()
            )

            anomaly_z = abs(
                attack_mean - baseline_mean
            ) / max(
                baseline_std,
                1e-8
            )

            # Attack persistence.
            persistence = float(
                (
                    abs(
                        attack_x - baseline_mean
                    )
                    > baseline_std
                ).mean()
            )

            # Pre-attack deviation.
            if not pre_x.empty:

                pre_mean = float(
                    pre_x.mean()
                )

                pre_z = abs(
                    pre_mean - baseline_mean
                ) / max(
                    baseline_std,
                    1e-8
                )

            else:

                pre_mean = np.nan
                pre_z = 0.0

            # Change from pre-attack to attack.
            if not pre_x.empty:

                pre_attack_mean = float(
                    pre_x.mean()
                )

                transition = abs(
                    attack_mean -
                    pre_attack_mean
                ) / max(
                    baseline_std,
                    1e-8
                )

            else:

                transition = 0.0

            # Within-attack temporal trend.
            if len(attack_x) >= 2:

                temporal_trend = abs(
                    float(
                        attack_x.iloc[-1]
                    )
                    -
                    float(
                        attack_x.iloc[0]
                    )
                ) / max(
                    baseline_std,
                    1e-8
                )

            else:

                temporal_trend = 0.0

            # Candidate-specific onset score.
            #
            # A component that becomes abnormal
            # during the attack after being relatively
            # normal immediately beforehand receives
            # stronger evidence.
            onset_score = max(
                0.0,
                anomaly_z - pre_z
            )

            truth_flag = int(
                component in truth_components
            )

            rows.append({

                "Attack_ID": attack_id,

                "Component": component,

                "Start_Timestamp": start,

                "End_Timestamp": end,

                "Ground_Truth_Root_Cause":
                    str(
                        w[
                            "Ground_Truth_Root_Cause"
                        ]
                    ),

                "Ground_Truth_Component":
                    truth_flag,

                "Baseline_Mean":
                    baseline_mean,

                "Baseline_Std":
                    baseline_std,

                "PreAttack_Mean":
                    pre_mean,

                "PreAttack_Z":
                    pre_z,

                "Attack_Mean":
                    attack_mean,

                "Attack_Z":
                    anomaly_z,

                "Onset_Score":
                    onset_score,

                "Persistence":
                    persistence,

                "Transition_Score":
                    transition,

                "Temporal_Trend":
                    temporal_trend,

                "Network_Events":
                    network_total,

                "Modbus_Events":
                    modbus_total

            })

    return pd.DataFrame(rows)


def main():

    print("=" * 80)
    print("CTF-RCA EVIDENCE EXTRACTION V2")
    print("=" * 80)

    windows = load_windows()

    print(
        "Attack episodes:",
        len(windows)
    )

    truth_to_dataset = load_mapping()

    print(
        "Ground-truth mappings:",
        len(truth_to_dataset)
    )

    print(
        "\nCalculating normal baseline..."
    )

    baseline = calculate_global_baseline()

    print(
        "Baseline variables:",
        len(baseline)
    )

    print(
        "\nCollecting pre-attack + attack data..."
    )

    attack_data = collect_attack_data(
        windows
    )

    print(
        "Attack datasets collected:",
        len(attack_data)
    )

    print(
        "\nCalculating evidence..."
    )

    evidence = extract_evidence(
        windows,
        attack_data,
        baseline,
        truth_to_dataset
    )

    evidence = evidence.sort_values(
        [
            "Attack_ID",
            "Onset_Score"
        ],
        ascending=[
            True,
            False
        ]
    )

    evidence.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 80)
    print("EVIDENCE V2 COMPLETE")
    print("=" * 80)

    print(
        "Rows:",
        f"{len(evidence):,}"
    )

    print(
        "Attacks:",
        evidence["Attack_ID"].nunique()
    )

    print(
        "Components:",
        evidence["Component"].nunique()
    )

    print(
        "Ground-truth candidate rows:",
        int(
            evidence[
                "Ground_Truth_Component"
            ].sum()
        )
    )

    print(
        "\nOutput:",
        OUTPUT_FILE
    )

    print("\nSample:")
    print(
        evidence[
            [
                "Attack_ID",
                "Component",
                "PreAttack_Z",
                "Attack_Z",
                "Onset_Score",
                "Persistence",
                "Transition_Score",
                "Ground_Truth_Component"
            ]
        ].head(20).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
