from pathlib import Path
import pandas as pd

INPUT_DIR = Path("data/raw/Network")
OUTPUT_DIR = Path("data/processed/network_features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORE_COLUMNS = [
    "num",
    "date",
    "time",
    "orig",
    "type",
    "i/f_name",
    "i/f_dir",
    "src",
    "dst",
    "proto",
    "appi_name",
    "proxy_src_ip",
    "Modbus_Function_Code",
    "Modbus_Function_Description",
    "Modbus_Transaction_ID",
    "SCADA_Tag",
    "Modbus_Value",
    "service",
    "s_port",
    "Tag",
]


def parse_timestamp(df):
    date = df["date"].astype("string").str.strip()
    time = df["time"].astype("string").str.strip()

    raw = date + " " + time

    return pd.to_datetime(
        raw,
        errors="coerce",
        dayfirst=True
    )


def process_file(path):

    # Empty partition
    if path.stat().st_size == 0:
        return None, "empty"

    try:
        df = pd.read_csv(
            path,
            usecols=lambda c: c in CORE_COLUMNS,
            low_memory=False
        )
    except pd.errors.EmptyDataError:
        return None, "empty"

    except Exception as e:
        return None, f"read_error: {e}"

    # Make sure all expected columns exist
    for col in CORE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    # Timestamp
    df["Timestamp"] = parse_timestamp(df)

    # Remove invalid timestamps
    df = df.dropna(subset=["Timestamp"])

    if df.empty:
        return None, "no_valid_timestamp"

    # Normalize categorical fields first
    proto = df["proto"].astype("string").fillna("").str.lower()

    # IMPORTANT:
    # fillna(False) prevents the nullable-boolean -> int8 error.
    df["is_tcp"] = (
        proto.eq("tcp")
        .fillna(False)
        .astype("int8")
    )

    df["is_udp"] = (
        proto.eq("udp")
        .fillna(False)
        .astype("int8")
    )

    df["is_modbus"] = (
        df["Modbus_Function_Code"].notna()
        .fillna(False)
        .astype("int8")
    )

    df["has_scada_tag"] = (
        df["SCADA_Tag"].notna()
        .fillna(False)
        .astype("int8")
    )

    df["has_src"] = (
        df["src"].notna()
        .fillna(False)
        .astype("int8")
    )

    df["has_dst"] = (
        df["dst"].notna()
        .fillna(False)
        .astype("int8")
    )

    # 1-second cyber-event aggregation
    grouped = (
        df.set_index("Timestamp")
        .groupby(pd.Grouper(freq="1s"))
        .agg(
            network_event_count=("num", "count"),

            unique_src_count=("src", "nunique"),
            unique_dst_count=("dst", "nunique"),

            unique_scada_tags=("SCADA_Tag", "nunique"),

            unique_modbus_functions=(
                "Modbus_Function_Code",
                "nunique"
            ),

            unique_transactions=(
                "Modbus_Transaction_ID",
                "nunique"
            ),

            tcp_count=("is_tcp", "sum"),
            udp_count=("is_udp", "sum"),

            modbus_event_count=("is_modbus", "sum"),
            scada_event_count=("has_scada_tag", "sum"),

            valid_src_count=("has_src", "sum"),
            valid_dst_count=("has_dst", "sum"),
        )
        .reset_index()
    )

    grouped["source_file"] = path.name

    return grouped, "ok"


def main():

    files = sorted(INPUT_DIR.glob("*.csv"))

    print("=" * 80)
    print("SWaT NETWORK PREPROCESSING")
    print("=" * 80)
    print("Input files:", len(files))
    print("Input directory:", INPUT_DIR)
    print("Output directory:", OUTPUT_DIR)

    counters = {
        "ok": 0,
        "empty": 0,
        "read_error": 0,
        "no_valid_timestamp": 0,
    }

    error_log = []

    for i, path in enumerate(files, 1):

        result, status = process_file(path)

        if result is not None:

            output_name = path.stem + "_1s.csv"

            result.to_csv(
                OUTPUT_DIR / output_name,
                index=False
            )

            counters["ok"] += 1

        elif status == "empty":

            counters["empty"] += 1

        elif status.startswith("read_error"):

            counters["read_error"] += 1

            error_log.append(
                f"{path.name}\t{status}"
            )

        elif status == "no_valid_timestamp":

            counters["no_valid_timestamp"] += 1

            error_log.append(
                f"{path.name}\tno_valid_timestamp"
            )

        if i % 25 == 0 or i == len(files):

            print(
                f"Processed {i}/{len(files)} | "
                f"OK={counters['ok']} | "
                f"Empty={counters['empty']} | "
                f"Errors={counters['read_error']} | "
                f"No timestamp={counters['no_valid_timestamp']}"
            )

    # Save error log
    if error_log:
        with open(
            OUTPUT_DIR / "network_preprocessing_errors.txt",
            "w",
            encoding="utf-8"
        ) as f:
            f.write("\n".join(error_log))

    print("\n" + "=" * 80)
    print("NETWORK PREPROCESSING COMPLETE")
    print("=" * 80)

    for key, value in counters.items():
        print(f"{key}: {value}")

    print("\nOutput directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
