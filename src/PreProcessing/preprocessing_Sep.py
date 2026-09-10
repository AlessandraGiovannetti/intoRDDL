"""
Preprocessing Sepsis Cases for generic MDP discovery.

Adapted from the data preprocessing code of I. Teinemaa et al.,
"Outcome-Oriented Predictive Process Monitoring: Review and Benchmark".

This version creates ONE dataset for generic MDP discovery.

It does NOT:
    - create sepsis_cases_1.csv
    - create sepsis_cases_2.csv
    - create deviant/regular labels
    - distinguish between predictive outcomes

It DOES:
    - keep all activities
    - keep case-level attributes
    - extract temporal features
    - extract release-related features
    - handle missing values
    - group rare categorical values into "other"
"""

import pm4py
import pandas as pd
import numpy as np
import os

import time

# ============================================================
# INPUT / OUTPUT
# ============================================================



preprocessing_start = time.perf_counter()

input_file = "./logs/split/sepsis/train.xes.gz"

input_data_folder = "./logs/split/sepsis"
output_data_folder = "./src/input"

csv_filename = "sepsis.csv"
out_filename = "sepsis_preprocessed.csv"


# ============================================================
# OPTIONAL: XES -> CSV
# ============================================================

CONVERT_XES_TO_CSV = True


if CONVERT_XES_TO_CSV:

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
# COLUMN CONFIGURATION
# ============================================================

case_id_col = "Case ID"
activity_col = "Activity"
timestamp_col = "timestamp"

category_freq_threshold = 10


# ============================================================
# FEATURES
# ============================================================

dynamic_cat_cols = [
    "Activity",
    "org:group"
]

static_cat_cols = [
    "Diagnose",
    "DiagnosticArtAstrup",
    "DiagnosticBlood",
    "DiagnosticECG",
    "DiagnosticIC",
    "DiagnosticLacticAcid",
    "DiagnosticLiquor",
    "DiagnosticOther",
    "DiagnosticSputum",
    "DiagnosticUrinaryCulture",
    "DiagnosticUrinarySediment",
    "DiagnosticXthorax",
    "DisfuncOrg",
    "Hypotensie",
    "Hypoxie",
    "InfectionSuspected",
    "Infusion",
    "Oligurie",
    "SIRSCritHeartRate",
    "SIRSCritLeucos",
    "SIRSCritTachypnea",
    "SIRSCritTemperature",
    "SIRSCriteria2OrMore"
]

dynamic_num_cols = [
    "CRP",
    "LacticAcid",
    "Leucocytes"
]

static_num_cols = [
    "Age"
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

cat_cols = (
    dynamic_cat_cols
    + static_cat_cols
)


# ============================================================
# TIMESTAMP FEATURES
# ============================================================

def extract_timestamp_features(group):

    group = group.sort_values(
        timestamp_col,
        ascending=False,
        kind="mergesort"
    ).copy()

    # Time since previous event
    tmp = (
        group[timestamp_col]
        - group[timestamp_col].shift(-1)
    )

    tmp = tmp.fillna(pd.Timedelta(0))

    group["timesincelastevent"] = (
        tmp / pd.Timedelta(minutes=1)
    )

    # Time since case start
    case_start = group[timestamp_col].iloc[-1]

    tmp = (
        group[timestamp_col]
        - case_start
    )

    tmp = tmp.fillna(pd.Timedelta(0))

    group["timesincecasestart"] = (
        tmp / pd.Timedelta(minutes=1)
    )

    # Restore chronological order
    group = group.sort_values(
        timestamp_col,
        ascending=True,
        kind="mergesort"
    )

    # Event number
    group["event_nr"] = (
        np.arange(len(group)) + 1
    )

    return group
# ============================================================
# LOAD DATA
# ============================================================

print("Loading Sepsis dataset...")

data = pd.read_csv(
    os.path.join(
        output_data_folder,
        csv_filename,
    ),
    sep=";"
)

print("Original columns:")
print(data.columns.tolist())


# ============================================================
# RENAME COLUMNS
# ============================================================

data = data.rename(
    columns={
        "case:concept:name": case_id_col,
        "concept:name": activity_col,
        "org:resource": "Resource",
        "time:timestamp": timestamp_col,
    }
)

data[case_id_col] = (
    data[case_id_col]
    .fillna("missing_caseid")
)


# ============================================================
# REMOVE INCOMPLETE CASES
# ============================================================

print("\nRemoving incomplete cases...")

releases = [
    "Release A",
    "Release B",
    "Release C",
    "Release D",
    "Release E"
]

cases_with_release = (
    data.groupby(case_id_col)[activity_col]
    .apply(
        lambda x: x.isin(releases).any()
    )
)

incomplete_cases = (
    cases_with_release[
        ~cases_with_release
    ].index
)

print(
    f"Removing {len(incomplete_cases)} "
    "incomplete cases."
)

data = data[
    ~data[case_id_col].isin(incomplete_cases)
].copy()


# ============================================================
# SELECT COLUMNS
# ============================================================

available_static_cols = [
    col
    for col in static_cols
    if col in data.columns
]

available_dynamic_cols = [
    col
    for col in dynamic_cols
    if col in data.columns
]

selected_cols = list(
    dict.fromkeys(
        available_static_cols
        + available_dynamic_cols
    )
)

data = data[selected_cols].copy()


# ============================================================
# TIMESTAMP CONVERSION
# ============================================================

print("\nConverting timestamps...")

data[timestamp_col] = pd.to_datetime(
    data[timestamp_col],
    format="mixed",
    errors="coerce"
)

invalid_timestamps = (
    data[timestamp_col].isna().sum()
)

if invalid_timestamps > 0:

    print(
        f"WARNING: {invalid_timestamps} "
        "invalid timestamps."
    )

    data = data.dropna(
        subset=[timestamp_col]
    )


# ============================================================
# BASIC TIMESTAMP FEATURES
# ============================================================

print("Extracting timestamp features...")

data["timesincemidnight"] = (
    data[timestamp_col].dt.hour * 60
    + data[timestamp_col].dt.minute
)

data["month"] = (
    data[timestamp_col].dt.month
)

data["weekday"] = (
    data[timestamp_col].dt.weekday
)

data["hour"] = (
    data[timestamp_col].dt.hour
)


# ============================================================
# CASE-LEVEL TIMESTAMP FEATURES
# ============================================================

print(
    "Extracting case-level timestamp features..."
)

data = (
    data
    .groupby(
        case_id_col,
        group_keys=False
    )
    .apply(
        extract_timestamp_features
    )
    .reset_index(drop=True)
)


# ============================================================
# SORT
# ============================================================

data = data.sort_values(
    [
        case_id_col,
        timestamp_col
    ],
    ascending=[
        True,
        True
    ],
    kind="mergesort"
).reset_index(drop=True)


# ============================================================
# FORWARD FILL
# ============================================================

print(
    "Forward-filling missing values "
    "within each case..."
)

grouped = data.groupby(
    case_id_col
)

for col in static_cols + dynamic_cols:

    if col in data.columns:

        data[col] = (
            grouped[col]
            .transform(
                lambda grp: grp.ffill()
            )
        )


# ============================================================
# MISSING VALUES
# ============================================================

print("Handling missing values...")

existing_cat_cols = [
    col
    for col in cat_cols
    if col in data.columns
]

if existing_cat_cols:

    data[existing_cat_cols] = (
        data[existing_cat_cols]
        .fillna("missing")
    )


existing_num_cols = [
    col
    for col in (
        static_num_cols
        + dynamic_num_cols
        + [
            "timesincemidnight",
            "month",
            "weekday",
            "hour",
            "timesincelastrelease",
            "recent_release",
            "event_nr"
        ]
    )
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
    "Grouping rare categorical values..."
)

for col in existing_cat_cols:

    # Activity deve rimanere invariata
    if col == activity_col:
        continue

    counts = data[col].value_counts()

    frequent_values = (
        counts[
            counts >= category_freq_threshold
        ]
        .index
    )

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
    [
        case_id_col,
        timestamp_col
    ],
    ascending=[
        True,
        True
    ],
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


preprocessing_end = time.perf_counter()

execution_time_seconds = (
    preprocessing_end - preprocessing_start
)

execution_time_minutes = (
    execution_time_seconds / 60
)

print("\n========================================")
print("PREPROCESSING EXECUTION TIME")
print("========================================")

print(f"Execution time: {execution_time_seconds:.2f} seconds")
print(f"Execution time: {execution_time_minutes:.2f} minutes")

# ============================================================
# REMOVE TEMPORARY CSV
# ============================================================

if CONVERT_XES_TO_CSV and os.path.exists(csv_path):
    os.remove(csv_path)
    print(f"Temporary CSV removed: {csv_path}")


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("PREPROCESSING COMPLETED")
print("========================================")

print(
    f"Output file: {output_path}"
)

print(
    f"Number of events: {len(data)}"
)

print(
    f"Number of cases: "
    f"{data[case_id_col].nunique()}"
)

print(
    f"Number of columns: "
    f"{len(data.columns)}"
)

print("\nFinal columns:")

for col in data.columns:
    print(f"  - {col}")

print("\nActivity distribution:")

print(
    data[activity_col]
    .value_counts()
)

print("\nFirst rows:")

print(data.head())