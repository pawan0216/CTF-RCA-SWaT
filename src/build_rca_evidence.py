from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("data/processed")

DATA_FILE = BASE / "ctf_rca_synchronized.csv"
WINDOW_FILE = BASE / "rca_attack_windows.csv"
OUTPUT_FILE = BASE / "rca_evidence_matrix.csv"

# 51 candidate process variables identified previously.
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
    ).astype(int)

    return windows


def load_normal_baseline():
    """
    Calculate mean/std for every physical variable
    using ONLY normal observations.

    This is descriptive baseline information.
    Attack labels are not used to construct RCA scores.
    """

    columns = ["Timestamp", "Is_Attack"] + CANDIDATES

    sums = {c: 0.0 for c in CANDIDATES}
    sums_sq = {c: 0.0 for c in CANDIDATES}
    counts = {c: 0 for c in CANDIDATES}

    reader = pd.read_csv(
        DATA_FILE,
        usecols=columns,
        parse_dates=["Timestamp"],
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

            if len(x) == 0:
                continue

            sums[c] += float(x.sum())
            sums_sq[c] += float((x * x).sum())
            counts[c] += int(len(x))

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

        # Avoid unstable z-scores for near-constant signals.
        if std < 1e-8:
            std = 1.0

        baseline[c] = (
            mean,
            std
        )

    return baseline


def calculate_evidence(windows, baseline):

    required = (
        ["Timestamp", "Is_Attack"]
        + CANDIDATES
        + NETWORK_COLUMNS
    )

    # Storage for rows belonging to each attack.
    attack_rows = {
        int(row.Attack_ID): []
        for _, row in windows.iterrows()
    }

    reader = pd.read_csv(
        DATA_FILE,
        usecols=required,
        parse_dates=["Timestamp"],
        chunksize=100_000,
        low_memory=False
    )

    for chunk in reader:

        # Keep only timestamps falling inside at least
        # one attack window.
        for _, w in windows.iterrows():

            attack_id = int(w.Attack_ID)

            mask = (
                (chunk["Timestamp"] >= w.Start_Timestamp)
                &
                (chunk["Timestamp"] <= w.End_Timestamp)
            )

            selected = chunk.loc[mask]

            if not selected.empty:
                attack_rows[attack_id].append(
                    selected.copy()
                )

    results = []

    for _, w in windows.iterrows():

        attack_id = int(w.Attack_ID)

        if not attack_rows[attack_id]:
            continue

        data = pd.concat(
            attack_rows[attack_id],
            ignore_index=True
        )

        # Network activity during this attack.
        network = data[NETWORK_COLUMNS].sum()

        network_events = float(
            network["network_event_count"]
        )

        modbus_events = float(
            network["modbus_event_count"]
        )

        tcp_events = float(
            network["tcp_count"]
        )

        udp_events = float(
            network["udp_count"]
        )

        # Attack ground truth is retained ONLY as evaluation metadata.
        truth = str(
            w["Ground_Truth_Root_Cause"]
        )

        truth_components = set(
            x.strip().upper()
            for x in truth.split("|")
            if x.strip()
        )

        for component in CANDIDATES:

            x = pd.to_numeric(
                data[component],
                errors="coerce"
            ).dropna()

            if len(x) == 0:
                continue

            mean = float(x.mean())
            std = float(x.std())

            baseline_mean, baseline_std = baseline[
                component
            ]

            # Standardized deviation from normal baseline.
            z = (
                abs(mean - baseline_mean)
                / max(baseline_std, 1e-8)
            )

            # Persistence of deviation.
            deviations = (
                abs(x - baseline_mean)
                > baseline_std
            )

            persistence = float(
                deviations.mean()
            )

            # Temporal trend within attack.
            if len(x) >= 2:

                first_value = float(x.iloc[0])
                last_value = float(x.iloc[-1])

                trend = abs(
                    last_value - first_value
                ) / max(
                    abs(baseline_std),
                    1e-8
                )

            else:
                trend = 0.0

            is_truth = int(
                component.upper()
                in truth_components
            )

            results.append({
                "Attack_ID": attack_id,
                "Component": component,
                "Start_Timestamp": w.Start_Timestamp,
                "End_Timestamp": w.End_Timestamp,

                "Ground_Truth_Root_Cause": truth,

                "Baseline_Mean": baseline_mean,
                "Baseline_Std": baseline_std,

                "Attack_Mean": mean,
                "Attack_Std": std,

                "Anomaly_Z": z,
                "Persistence": persistence,
                "Temporal_Trend": trend,

                "Network_Events": network_events,
                "Modbus_Events": modbus_events,
                "TCP_Events": tcp_events,
                "UDP_Events": udp_events,

                # Evaluation-only field.
                "Ground_Truth_Component": is_truth
            })

    return pd.DataFrame(results)


def main():

    print("=" * 80)
    print("CTF-RCA PHYSICAL + NETWORK EVIDENCE EXTRACTION")
    print("=" * 80)

    print("\nLoading attack windows...")
    windows = load_windows()

    print(
        "Attack episodes:",
        len(windows)
    )

    print("\nCalculating normal physical baseline...")

    baseline = load_normal_baseline()

    print(
        "Baseline components:",
        len(baseline)
    )

    print("\nExtracting attack evidence...")

    evidence = calculate_evidence(
        windows,
        baseline
    )

    if evidence.empty:
        raise RuntimeError(
            "No attack evidence was generated."
        )

    evidence = evidence.sort_values(
        [
            "Attack_ID",
            "Anomaly_Z"
        ],
        ascending=[
            True,
            False
        ]
    ).reset_index(drop=True)

    evidence.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 80)
    print("EVIDENCE MATRIX COMPLETE")
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
        "Output:",
        OUTPUT_FILE
    )

    print("\nSample:")
    print(
        evidence[
            [
                "Attack_ID",
                "Component",
                "Anomaly_Z",
                "Persistence",
                "Temporal_Trend",
                "Network_Events",
                "Ground_Truth_Component"
            ]
        ].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()
