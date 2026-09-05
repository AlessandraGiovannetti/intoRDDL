"""
Preprocessing BPI Challenge 2020 - Travel Permit Log
for generic MDP discovery.

Input:
    PermitLog.csv

Output:
    PermitLog_preprocessed.csv

The preprocessing:
    - keeps all activities
    - keeps relevant case-level attributes
    - keeps Resource and Role
    - extracts temporal features
    - handles missing values
    - groups rare categorical values into "other"
    - creates event_nr
    - creates timesincelastevent
    - creates timesincecasestart

The dataset is prepared for generic MDP discovery.
No outcome-specific filtering is performed.
"""

import pm4py
import pandas as pd
import numpy as np
import os


# ============================================================
# FILE CONFIGURATION
# ============================================================

input_file = "../logs/PermitLog.xes.gz"

input_data_folder = "./input"
output_data_folder = "./input"

csv_filename = "PermitLog.csv"
out_filename = "PermitLog_preprocessed.csv"


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
        input_data_folder,
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
resource_col = "Resource"
role_col = "Role"

category_freq_threshold = 10


# ------------------------------------------------------------
# Raw column names in THIS Permit log
# ------------------------------------------------------------

raw_case_id_col = "id"
raw_activity_col = "concept:name"
raw_timestamp_col = "time:timestamp"
raw_resource_col = "org:resource"
raw_role_col = "org:role"


# ============================================================
# FEATURES
# ============================================================

# ------------------------------------------------------------
# Dynamic attributes
# ------------------------------------------------------------

dynamic_cat_cols = [
    "Activity",
    "Resource",
    "Role"
]


# ------------------------------------------------------------
# Static case-level categorical attributes
#
# These are the meaningful Permit-level attributes.
# The many dec_id_*, Rfp_id_*, DeclarationNumber_*,
# Project_*, Task_*, etc. columns are intentionally not
# included because they are mostly identifiers / duplicated
# linked-object information and can create very high
# cardinality in the MDP/RDDL encoding.
# ------------------------------------------------------------

static_cat_cols = [

    "OrganizationalEntity",

    "ProjectNumber",
    "TaskNumber",

    "ActivityNumber",

    "Permit",

    "travel permit number",

    "BudgetNumber",

    "Overspent",

    "DeclarationNumber_0",

    "RfpNumber_0",

    "Cost Type_0",

    "Task_0",

    "Project_0",

]


# ------------------------------------------------------------
# Static numeric attributes
# ------------------------------------------------------------

static_num_cols = [

    "TotalDeclared",

    "RequestedAmount_0",

    "RequestedBudget",

    "OverspentAmount",

]


# ------------------------------------------------------------
# Dynamic numeric attributes
#
# These are included only if they actually exist in the CSV.
# ------------------------------------------------------------

dynamic_num_cols = [

    "Amount",

    "RequestedAmount",

]


# ============================================================
# COMPLETE COLUMN LISTS
# ============================================================

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

    # --------------------------------------------------------
    # Sort newest -> oldest
    # --------------------------------------------------------

    group = group.sort_values(
        timestamp_col,
        ascending=False,
        kind="mergesort"
    ).copy()


    # --------------------------------------------------------
    # Time since previous event
    # --------------------------------------------------------

    tmp = (
        group[timestamp_col]
        - group[timestamp_col].shift(-1)
    )

    tmp = tmp.fillna(
        pd.Timedelta(0)
    )

    group["timesincelastevent"] = (
        tmp / pd.Timedelta(minutes=1)
    )


    # --------------------------------------------------------
    # Time since case start
    # --------------------------------------------------------

    case_start = (
        group[timestamp_col].iloc[-1]
    )

    tmp = (
        group[timestamp_col]
        - case_start
    )

    tmp = tmp.fillna(
        pd.Timedelta(0)
    )

    group["timesincecasestart"] = (
        tmp / pd.Timedelta(minutes=1)
    )


    # --------------------------------------------------------
    # Restore chronological order
    # --------------------------------------------------------

    group = group.sort_values(
        timestamp_col,
        ascending=True,
        kind="mergesort"
    )


    # --------------------------------------------------------
    # Event number
    # --------------------------------------------------------

    group["event_nr"] = (
        np.arange(len(group)) + 1
    )

    return group


# ============================================================
# LOAD CSV
# ============================================================

print("\nLoading Permit CSV dataset...")

csv_path = os.path.join(
    input_data_folder,
    csv_filename
)

data = pd.read_csv(
    csv_path,
    sep=";"
)


print("\nOriginal columns:")
print(data.columns.tolist())

print(
    f"\nOriginal number of columns: "
    f"{len(data.columns)}"
)


# ============================================================
# RENAME CORE COLUMNS
# ============================================================

rename_map = {

    raw_case_id_col:
        case_id_col,

    raw_activity_col:
        activity_col,

    raw_timestamp_col:
        timestamp_col,

    raw_resource_col:
        resource_col,

    raw_role_col:
        role_col,
}


# Rename only columns that actually exist

rename_map = {
    old: new
    for old, new in rename_map.items()
    if old in data.columns
}


data = data.rename(
    columns=rename_map
)


# ============================================================
# REMOVE "case:" PREFIX
# ============================================================

data.columns = [

    col[len("case:"):]
    if col.startswith("case:")
    else col

    for col in data.columns
]


# ============================================================
# CHECK CORE COLUMNS
# ============================================================

required_columns = [

    case_id_col,
    activity_col,
    timestamp_col
]


missing_required = [

    col
    for col in required_columns
    if col not in data.columns
]


if missing_required:

    raise ValueError(
        "Missing required columns: "
        + str(missing_required)
    )


# ============================================================
# CASE ID
# ============================================================

data[case_id_col] = (
    data[case_id_col]
    .fillna("missing_caseid")
    .astype(str)
)


# ============================================================
# SHOW AVAILABLE PERMIT ATTRIBUTES
# ============================================================

print("\nChecking configured attributes...")

print("\nCategorical attributes found:")

for col in static_cat_cols:

    if col in data.columns:

        print(
            f"  [OK] {col}"
        )

    else:

        print(
            f"  [--] {col}"
        )


print("\nNumeric attributes found:")

for col in static_num_cols:

    if col in data.columns:

        print(
            f"  [OK] {col}"
        )

    else:

        print(
            f"  [--] {col}"
        )


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


print("\nSelected columns:")

for col in selected_cols:

    print(
        f"  - {col}"
    )


data = data[selected_cols].copy()


# ============================================================
# TIMESTAMP CONVERSION
# ============================================================

print("\nConverting timestamps...")

data[timestamp_col] = pd.to_datetime(
    data[timestamp_col],
    format="mixed",
    errors="coerce",
    utc=True
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

print(
    "Extracting timestamp features..."
)


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

data = (

    data

    .sort_values(
        [
            case_id_col,
            timestamp_col
        ],

        ascending=[
            True,
            True
        ],

        kind="mergesort"
    )

    .reset_index(drop=True)

)


# ============================================================
# FORWARD FILL
# ============================================================

print(
    "\nForward-filling missing values "
    "within each case..."
)


grouped = data.groupby(
    case_id_col
)


# Only actual columns are processed

for col in (

    static_cols
    + dynamic_cols

):

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

print(
    "Handling missing values..."
)


existing_cat_cols = [

    col
    for col in cat_cols
    if col in data.columns
]


if existing_cat_cols:

    # --------------------------------------------------------
    # Convert categorical values to string
    # --------------------------------------------------------

    for col in existing_cat_cols:

        data[col] = (
            data[col]
            .astype("object")
        )


    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    data[existing_cat_cols] = (

        data[existing_cat_cols]

        .fillna("missing")

    )


    # --------------------------------------------------------
    # Normalize common missing values
    # --------------------------------------------------------

    data[existing_cat_cols] = (

        data[existing_cat_cols]

        .replace(
            {
                "UNKNOWN": "missing",
                "Unknown": "missing",
                "unknown": "missing",

                "MISSING": "missing",
                "Missing": "missing",
                "missing": "missing",

                "NaN": "missing",
                "nan": "missing"
            }
        )

    )


# ============================================================
# NUMERIC MISSING VALUES
# ============================================================

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
            "event_nr"
        ]

    )

    if col in data.columns
]


if existing_num_cols:

    data[existing_num_cols] = (

        data[existing_num_cols]

        .apply(
            pd.to_numeric,
            errors="coerce"
        )

        .fillna(0)

    )


# ============================================================
# SPECIAL HANDLING FOR OVERSPENT
# ============================================================

if "Overspent" in data.columns:

    print(
        "\nOverspent distribution before "
        "normalization:"
    )

    print(
        data["Overspent"]
        .value_counts(dropna=False)
    )


    # Keep it categorical.
    #
    # True / False are NOT converted to 1 / 0.
    #
    # This is useful for the RDDL encoder because
    # the attribute can later be represented as bool.

    data["Overspent"] = (

        data["Overspent"]

        .astype(str)

        .replace(
            {
                "nan": "missing",
                "NaN": "missing"
            }
        )

    )


# ============================================================
# RARE CATEGORICAL VALUES
# ============================================================

print(
    "\nGrouping rare categorical values..."
)


for col in existing_cat_cols:

    # --------------------------------------------------------
    # Activity must remain unchanged
    # --------------------------------------------------------

    if col == activity_col:

        continue


    counts = (
        data[col]
        .value_counts()
    )


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

data = (

    data

    .sort_values(
        [
            case_id_col,
            timestamp_col
        ],

        ascending=[
            True,
            True
        ],

        kind="mergesort"
    )

    .reset_index(drop=True)

)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    output_data_folder,
    exist_ok=True
)


output_path = os.path.join(
    output_data_folder,
    out_filename
)


print(
    f"\nSaving processed dataset to:"
    f"\n{output_path}"
)


data.to_csv(
    output_path,
    sep=";",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n========================================"
)

print(
    "PREPROCESSING COMPLETED"
)

print(
    "========================================"
)


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


print(
    "\nFinal columns:"
)


for col in data.columns:

    print(
        f"  - {col}"
    )


print(
    "\nActivity distribution:"
)


print(
    data[activity_col]
    .value_counts()
)


if "Overspent" in data.columns:

    print(
        "\nOverspent distribution:"
    )

    print(
        data["Overspent"]
        .value_counts()
    )


print(
    "\nFirst rows:"
)


print(
    data.head()
)