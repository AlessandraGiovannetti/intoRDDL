"""
Preprocessing BPI Challenge 2020 - International Declarations Log
for generic MDP discovery.

Input:
    InternationalDeclarations.csv
    oppure InternationalDeclarations.xes.gz

Output:
    InternationalDeclarations_preprocessed.csv

The preprocessing:
    - keeps all activities
    - keeps relevant declaration-level attributes
    - keeps relevant permit-level attributes
    - keeps Resource and Role
    - extracts temporal features
    - handles missing values
    - groups rare categorical values into "other"
    - creates event_nr
    - creates timesincelastevent
    - creates timesincecasestart

No outcome-specific filtering is performed.
"""

import pm4py
import pandas as pd
import numpy as np
import os


# ============================================================
# FILE CONFIGURATION
# ============================================================

input_file = "./logs/split/internationalDeclarations/train.xes.gz"

input_data_folder = "./logs/split/internationalDeclarations"
output_data_folder = "./src/input"

csv_filename = "InternationalDeclarations.csv"
out_filename = "intDecl_preprocessed.csv"

CONVERT_XES_TO_CSV = True


# ============================================================
# XES -> CSV
# ============================================================

if CONVERT_XES_TO_CSV:

    print("Loading XES log...")

    log = pm4py.read_xes(input_file)

    print("Converting XES to DataFrame...")

    df_xes = pm4py.convert_to_dataframe(log)

    os.makedirs(
        input_data_folder,
        exist_ok=True
    )

    csv_path = os.path.join(
        input_data_folder,
        csv_filename
    )

    df_xes.to_csv(
        csv_path,
        sep=";",
        index=False
    )

    print(
        f"Saved CSV to {csv_path}"
    )


# ============================================================
# COLUMN CONFIGURATION
# ============================================================

case_id_col = "Case ID"
activity_col = "Activity"
timestamp_col = "timestamp"
resource_col = "Resource"
role_col = "Role"

category_freq_threshold = 10


# ============================================================
# RAW COLUMN NAMES
# ============================================================

raw_case_id_col = "id"
raw_activity_col = "concept:name"
raw_timestamp_col = "time:timestamp"
raw_resource_col = "org:resource"
raw_role_col = "org:role"


# ============================================================
# FEATURES
# ============================================================

# ------------------------------------------------------------
# Dynamic categorical attributes
# ------------------------------------------------------------

dynamic_cat_cols = [
    "Activity",
    "Resource",
    "Role"
]


# ------------------------------------------------------------
# Static categorical attributes
# ------------------------------------------------------------
#
# These are case-level attributes.
#
# We keep the attributes that identify the declaration/permit
# context but avoid unnecessary duplicated identifiers.
# ------------------------------------------------------------

static_cat_cols = [

    # Declaration
    "DeclarationNumber",

    # Permit
    "Permit travel permit number",
    "travel permit number",

    "Permit TaskNumber",
    "Permit BudgetNumber",
    "Permit ProjectNumber",

    "Permit OrganizationalEntity",

    "Permit ID",
    "Permit id",

    "BudgetNumber",

]


# ------------------------------------------------------------
# Static numeric attributes
# ------------------------------------------------------------
#
# These are the main monetary attributes of the declaration.
# ------------------------------------------------------------

static_num_cols = [

    "Amount",

    "RequestedAmount",

    "OriginalAmount",

    "Permit RequestedBudget",

    "AdjustedAmount",

]


# ------------------------------------------------------------
# Dynamic numeric attributes
# ------------------------------------------------------------
#
# The provided log does not contain native event-level
# numerical attributes.
#
# Keep this list empty unless the extracted log contains
# additional event-level numeric attributes.
# ------------------------------------------------------------

dynamic_num_cols = []


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

        tmp
        / pd.Timedelta(minutes=1)

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

        tmp
        / pd.Timedelta(minutes=1)

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
# LOAD DATA
# ============================================================

print(
    "\nLoading International Declarations dataset..."
)


csv_path = os.path.join(
    input_data_folder,
    csv_filename
)


data = pd.read_csv(
    csv_path,
    sep=";"
)


print("\nOriginal columns:")

print(
    data.columns.tolist()
)


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
# CHECK REQUIRED COLUMNS
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
# AVAILABLE ATTRIBUTES
# ============================================================

print(
    "\nChecking configured attributes..."
)


print(
    "\nCategorical attributes found:"
)


for col in static_cat_cols:

    if col in data.columns:

        print(
            f"  [OK] {col}"
        )

    else:

        print(
            f"  [--] {col}"
        )


print(
    "\nNumeric attributes found:"
)


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


print(
    "\nSelected columns:"
)


for col in selected_cols:

    print(
        f"  - {col}"
    )


data = data[
    selected_cols
].copy()


# ============================================================
# TIMESTAMP CONVERSION
# ============================================================

print(
    "\nConverting timestamps..."
)


data[timestamp_col] = pd.to_datetime(

    data[timestamp_col],

    format="mixed",

    errors="coerce",

    utc=True

)


invalid_timestamps = (

    data[timestamp_col]
    .isna()
    .sum()

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

        ascending=[True, True],

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

    for col in existing_cat_cols:

        data[col] = (
            data[col]
            .astype("object")
        )


    data[existing_cat_cols] = (

        data[existing_cat_cols]

        .fillna("missing")

    )


    data[existing_cat_cols] = (

        data[existing_cat_cols]

        .replace(

            {

                "UNKNOWN": "missing",
                "Unknown": "missing",
                "unknown": "missing",

                "MISSING": "missing",
                "Missing": "missing",

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
# RARE CATEGORICAL VALUES
# ============================================================

print(
    "\nGrouping rare categorical values..."
)


for col in existing_cat_cols:

    # Activity must remain unchanged

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

        ascending=[True, True],

        kind="mergesort"

    )

    .reset_index(drop=True)

)

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

if CONVERT_XES_TO_CSV and os.path.exists(csv_path):
    os.remove(csv_path)
    print(f"Temporary CSV removed: {csv_path}")

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


print(
    "\nFirst rows:"
)


print(
    data.head()
)