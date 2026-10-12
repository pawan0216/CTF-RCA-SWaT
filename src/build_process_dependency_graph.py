from pathlib import Path
import pandas as pd

BASE = Path("data/processed")

# SWaT process-dependency relationships.
# These represent process-flow/control relationships used as
# supporting evidence during root-cause ranking.

EDGES = [
    # Stage 1
    ("FIT101", "LIT101", "process_flow"),
    ("LIT101", "MV101", "level_control"),
    ("MV101", "P101", "control"),
    ("MV101", "P102", "control"),

    # Stage 2
    ("P201", "FIT201", "process_flow"),
    ("P202", "FIT201", "process_flow"),
    ("FIT201", "AIT201", "process_flow"),
    ("AIT201", "AIT202", "process_dependency"),
    ("AIT202", "AIT203", "process_dependency"),
    ("MV201", "P201", "control"),
    ("P203", "P204", "control"),
    ("P205", "P206", "control"),

    # Stage 3
    ("FIT301", "LIT301", "process_flow"),
    ("LIT301", "DPIT301", "process_dependency"),
    ("DPIT301", "MV301", "control"),
    ("MV301", "MV302", "control"),
    ("MV302", "MV303", "control"),
    ("MV303", "MV304", "control"),
    ("MV304", "P301", "control"),
    ("P301", "P302", "process_flow"),

    # Stage 4
    ("P401", "FIT401", "process_flow"),
    ("FIT401", "LIT401", "process_flow"),
    ("LIT401", "AIT401", "process_dependency"),
    ("AIT401", "AIT402", "process_dependency"),
    ("AIT402", "UV401", "process_dependency"),
    ("UV401", "AIT401", "control"),

    # Stage 5
    ("P501", "FIT501", "process_flow"),
    ("P501", "FIT502", "process_flow"),
    ("P502", "FIT503", "process_flow"),
    ("FIT501", "AIT501", "process_dependency"),
    ("FIT502", "AIT502", "process_dependency"),
    ("FIT503", "AIT503", "process_dependency"),
    ("FIT504", "AIT504", "process_dependency"),
    ("AIT501", "AIT502", "process_dependency"),
    ("AIT502", "AIT503", "process_dependency"),
    ("AIT503", "AIT504", "process_dependency"),
    ("AIT504", "P501", "control"),

    # Stage 6
    ("FIT601", "P601", "process_flow"),
    ("P601", "P602", "control"),
    ("P602", "P603", "control"),
]

edges = pd.DataFrame(
    EDGES,
    columns=["Source", "Target", "Relationship"]
)

# Candidate variables actually available in the synchronized dataset.
sync = pd.read_csv(
    BASE / "ctf_rca_synchronized.csv",
    nrows=1
)

candidates = set(sync.columns)

edges["Source_Available"] = edges["Source"].isin(candidates)
edges["Target_Available"] = edges["Target"].isin(candidates)

edges["Usable"] = (
    edges["Source_Available"]
    & edges["Target_Available"]
)

edges.to_csv(
    BASE / "rca_process_dependency_graph.csv",
    index=False
)

print("=" * 80)
print("RCA PROCESS-DEPENDENCY GRAPH")
print("=" * 80)
print(f"Total proposed edges : {len(edges)}")
print(f"Usable edges          : {edges['Usable'].sum()}")
print(f"Unavailable edges     : {(~edges['Usable']).sum()}")

print("\nUSABLE PROCESS DEPENDENCIES")
print("-" * 80)
print(
    edges[edges["Usable"]]
    [["Source", "Target", "Relationship"]]
    .to_string(index=False)
)

print("\nUnavailable edges:")
print(
    edges[~edges["Usable"]]
    [["Source", "Target", "Relationship"]]
    .to_string(index=False)
)

print("\nSaved:")
print(BASE / "rca_process_dependency_graph.csv")
