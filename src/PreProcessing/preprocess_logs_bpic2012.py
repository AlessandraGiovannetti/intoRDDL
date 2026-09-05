"""
Preprocessing BPIC12 for generic MDP discovery.

Adapted from the data preprocessing code of I. Teinemaa et al.,
"Outcome-Oriented Predictive Process Monitoring: Review and Benchmark".

This version is intended to produce a single event log suitable for
generic MDP discovery.

It does NOT:
    - distinguish between accepted / declined / cancelled outcomes
    - create deviant / regular labels
    - create three separate datasets

It DOES:
    - convert the XES log to CSV
    - keep all cases and all activities
    - forward-fill resource values within each case
    - extract temporal features
    - extract event number within each case
    - extract number of concurrently open cases
    - handle missing categorical/numerical values
    - group infrequent categorical values into "other"
"""

import pandas as pd
import numpy as np
import os
import sys
import pm4py


# ============================================================
# INPUT / OUTPUT
# ============================================================

input_file = "./logs/split/bpi12/train.xes.gz"

input_data_folder = "./logs/split/bpi12"
output_data_folder = "./src/input"

csv_filename = "bpi12.csv"
out_filename = "bpi12_preprocessed.csv"


# ============================================================
# CONVERT XES -> CSV
# ============================================================

print("Loading XES log...")

log = pm4py.read_xes(input_file)

print("Converting XES to DataFrame...")

df_xes = pm4py.convert_to_dataframe(log)

os.makedirs(input_data_folder, exist_ok=True)

csv_path = os.path.join(
    output_data_folder,
    csv_filename
)

df_xes.to_csv(
    csv_path,
    sep=";",
    index=False
)

print(f"Saved CSV to {csv_path}")


# ============================================================
# CONFIGURATION
# ============================================================


case_id_col = "case:concept:name"
activity_col = "concept:name"
timestamp_col = "time:timestamp"
resource_col = "org:resource"

# Static attributes
static_cat_cols = []
static_num_cols = [
    "AMOUNT_REQ"
]

# Dynamic attributes
dynamic_cat_cols = [
    activity_col,
    resource_col,
    "lifecycle:transition"
]

dynamic_num_cols = [
    "timesincemidnight",
    "timesincelastevent",
    "timesincecasestart",
    "event_nr",
    "month",
    "weekday",
    "hour",
    "open_cases"
]

static_cols = (
    static_cat_cols
    + static_num_cols
    + [case_id_col]
)

dynamic_cols = (
    dynamic_cat_cols
    + dynamic_num_cols
    + [timestamp_col]
)

cat_cols = dynamic_cat_cols + static_cat_cols


# Minimum frequency for categorical values
freq_threshold = 10


# ============================================================
# TIMESTAMP FEATURES
# ============================================================

def extract_timestamp_features(group):
    """
    Extract temporal features for a single case.

    Features:
        - timesincelastevent
        - timesincecasestart
        - event_nr
    """

    # Sort chronologically
    group = group.sort_values(
        timestamp_col,
        ascending=True,
        kind="mergesort"
    ).copy()

    # --------------------------------------------------------
    # Time since previous event
    # --------------------------------------------------------

    group["timesincelastevent"] = (
        group[timestamp_col]
        .diff()
        .dt.total_seconds()
        .div(60)
        .fillna(0)
    )

    # --------------------------------------------------------
    # Time since case start
    # --------------------------------------------------------

    case_start = group[timestamp_col].iloc[0]

    group["timesincecasestart"] = (
        group[timestamp_col] - case_start
    ).dt.total_seconds().div(60).fillna(0)

    # --------------------------------------------------------
    # Event number
    # --------------------------------------------------------

    group["event_nr"] = np.arange(
        1,
        len(group) + 1
    )

    return group


# ============================================================
# OPEN CASES
# ============================================================

def get_open_cases(date):
    """
    Return the number of cases that are open at a given timestamp.

    A case is considered open when:

        start_time <= date < end_time
    """

    return (
        (dt_first_last_timestamps["start_time"] <= date)
        &
        (dt_first_last_timestamps["end_time"] > date)
    ).sum()


# ============================================================
# LOAD CSV
# ============================================================

print("\nLoading generated CSV...")

data = pd.read_csv(
    os.path.join(output_data_folder, csv_filename),
    sep=";"
)

print("Columns loaded:")
print(data.columns.tolist())


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    case_id_col,
    activity_col,
    timestamp_col
]

missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if missing_columns:
    raise ValueError(
        "The following required columns are missing from the log: "
        + str(missing_columns)
    )


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

data.rename(
    columns=lambda x: x.replace("(case) ", ""),
    inplace=True
)


# ============================================================
# TIMESTAMP CONVERSION
# ============================================================

print("\nConverting timestamps...")

data[timestamp_col] = pd.to_datetime(
    data[timestamp_col],
    format="mixed",
    errors="coerce"
)

# Check invalid timestamps
invalid_timestamps = data[timestamp_col].isna().sum()

if invalid_timestamps > 0:
    print(
        f"WARNING: {invalid_timestamps} rows have invalid timestamps."
    )

    data = data.dropna(
        subset=[timestamp_col]
    )


# ============================================================
# SORT EVENTS
# ============================================================

data = data.sort_values(
    [case_id_col, timestamp_col],
    ascending=[True, True],
    kind="mergesort"
).reset_index(drop=True)


# ============================================================
# FORWARD FILL RESOURCE
# ============================================================

if resource_col in data.columns:

    print("Forward-filling resource values...")

    data[resource_col] = (
        data.sort_values(
            timestamp_col,
            ascending=True,
            kind="mergesort"
        )
        .groupby(case_id_col)[resource_col]
        .transform(lambda grp: grp.ffill())
    )


# ============================================================
# BASIC TIMESTAMP FEATURES
# ============================================================

print("Extracting timestamp features...")

data["timesincemidnight"] = (
    data[timestamp_col].dt.hour * 60
    + data[timestamp_col].dt.minute
)

data["month"] = data[timestamp_col].dt.month

data["weekday"] = data[timestamp_col].dt.weekday

data["hour"] = data[timestamp_col].dt.hour


# ============================================================
# CASE-LEVEL TIMESTAMP FEATURES
# ============================================================

print("Extracting case-level timestamp features...")

data = (
    data.groupby(
        case_id_col,
        group_keys=False
    )
    .apply(
        extract_timestamp_features
    )
    .reset_index(drop=True)
)


# ============================================================
# OPEN CASES
# ============================================================

print("Extracting open cases...")

data = data.sort_values(
    timestamp_col,
    ascending=True,
    kind="mergesort"
).reset_index(drop=True)


dt_first_last_timestamps = (
    data.groupby(case_id_col)[timestamp_col]
    .agg(["min", "max"])
)

dt_first_last_timestamps.columns = [
    "start_time",
    "end_time"
]

data["open_cases"] = data[timestamp_col].apply(
    get_open_cases
)


# ============================================================
# SELECT RELEVANT COLUMNS
# ============================================================

# Keep only columns that actually exist in the log.
# This makes the preprocessing robust if, for example,
# lifecycle:transition is not present.

available_static_cols = [
    col for col in static_cols
    if col in data.columns
]

available_dynamic_cols = [
    col for col in dynamic_cols
    if col in data.columns
]

selected_cols = list(dict.fromkeys(
    available_static_cols + available_dynamic_cols
))

data = data[selected_cols].copy()


# ============================================================
# MISSING VALUES
# ============================================================

print("Handling missing values...")

# Forward-fill values within each case
grouped = (
    data.sort_values(
        timestamp_col,
        ascending=True,
        kind="mergesort"
    )
    .groupby(case_id_col)
)

for col in selected_cols:

    if col == case_id_col:
        continue

    data[col] = grouped[col].transform(
        lambda grp: grp.ffill()
    )


# ------------------------------------------------------------
# Categorical missing values
# ------------------------------------------------------------

existing_cat_cols = [
    col for col in cat_cols
    if col in data.columns
]

if existing_cat_cols:

    data[existing_cat_cols] = (
        data[existing_cat_cols]
        .fillna("missing")
    )


# ------------------------------------------------------------
# Numerical missing values
# ------------------------------------------------------------

existing_num_cols = [
    col for col in static_num_cols + dynamic_num_cols
    if col in data.columns
]

if existing_num_cols:

    data[existing_num_cols] = (
        data[existing_num_cols]
        .fillna(0)
    )


# ============================================================
# RARE CATEGORICAL VALUES
# ============================================================

print(
    f"Grouping categorical values occurring fewer than "
    f"{freq_threshold} times..."
)

for col in existing_cat_cols:

    counts = data[col].value_counts()

    frequent_values = counts[
        counts >= freq_threshold
    ].index

    mask = data[col].isin(
        frequent_values
    )

    data.loc[
        ~mask,
        col
    ] = "other"


# ============================================================
# FINAL SORTING
# ============================================================

data = data.sort_values(
    [case_id_col, timestamp_col],
    ascending=[True, True],
    kind="mergesort"
).reset_index(drop=True)

# ============================================================
# CASE EXECUTION TIME
# ============================================================

print("Calculating case execution time...")

case_execution_time = (
    data.groupby(case_id_col)[timestamp_col]
    .agg(["min", "max"])
)

case_execution_time["execution_time_minutes"] = (
    case_execution_time["max"]
    - case_execution_time["min"]
).dt.total_seconds() / 60.0

data = data.merge(
    case_execution_time["execution_time_minutes"],
    left_on=case_id_col,
    right_index=True,
    how="left"
)
# ============================================================
# SAVE
# ============================================================

output_path = os.path.join(
    output_data_folder,
    out_filename
)

print(
    f"\nSaving processed dataset to: "
    f"{output_path}"
)

data.to_csv(
    output_path,
    sep=";",
    index=False
)

# ============================================================
# REMOVE TEMPORARY CSV
# ============================================================

if os.path.exists(csv_path):
    os.remove(csv_path)
    print(f"Temporary CSV removed: {csv_path}")


# ============================================================
# SAVE FINAL DATASET
# ============================================================

final_output = os.path.join(
    output_data_folder,
    out_filename
)

data.to_csv(
    final_output,
    sep=";",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("PREPROCESSING COMPLETED")
print("========================================")

print(f"Output file: {final_output}")
print(f"Number of events: {len(data)}")
print(f"Number of cases: {data[case_id_col].nunique()}")
print(f"Number of columns: {len(data.columns)}")

print("\nFinal columns:")
for col in data.columns:
    print(f"  - {col}")

print("\nActivity distribution:")
print(data[activity_col].value_counts())

print("\nFirst rows:")
print(data.head())