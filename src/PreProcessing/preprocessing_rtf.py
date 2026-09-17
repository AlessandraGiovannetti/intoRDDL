"""
Adapted from the data preprocessing code of I. Teinemaa et al., 
"Outcome-Oriented Predictive Process Monitoring: Review and Benchmark".
"""
import pandas as pd
import os
import numpy as np
import sys
import pm4py
import os
import sys
import pandas as pd
import numpy as np
import time


preprocessing_start = time.perf_counter()

input_file = "./logs/split/rtf/train.xes.gz"

input_data_folder = "./logs/split/rtf"
output_data_folder = "./src/input"

csv_filename = "rtf.csv"
out_filename = "rtf_preprocessed.csv"

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


# changed the column names
case_id_col = "Case ID"
activity_col = "Activity"
resource_col = "Resource"
timestamp_col = "Complete Timestamp"
label_col = "label"
pos_label = "deviant"
neg_label = "regular"

freq_threshold = 10

# features for classifier
dynamic_cat_cols = ["Activity", "Resource", "lastSent", "notificationType", "dismissal"]
static_cat_cols = ["article", "vehicleClass"]
dynamic_num_cols = ["expense", "timesincecasestart", "remaining_time_minutes"]
static_num_cols = ["amount", "points"]

static_cols = static_cat_cols + static_num_cols + [case_id_col]
dynamic_cols = dynamic_cat_cols + dynamic_num_cols + [timestamp_col]
cat_cols = dynamic_cat_cols + static_cat_cols


def extract_timestamp_features(group):
    """
    Extract temporal features for a single case.

    Features:
        - timesincelastevent
        - timesincecasestart
        - remaining_time_minutes
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
    # Remaining time until case completion
    # --------------------------------------------------------

    case_end = group[timestamp_col].iloc[-1]

    group["remaining_time_minutes"] = (
        case_end - group[timestamp_col]
    ).dt.total_seconds().div(60).fillna(0)

    # --------------------------------------------------------
    # Event number
    # --------------------------------------------------------

    group["event_nr"] = np.arange(
        1,
        len(group) + 1
    )

    return group


def check_if_activity_exists(group, activity, cut_from_idx=True):
    relevant_activity_idxs = np.where(group[activity_col] == activity)[0]
    if len(relevant_activity_idxs) > 0:
        idx = relevant_activity_idxs[0]
        group[label_col] = pos_label
        if cut_from_idx:
            return group[:idx]
        else:
            return group
    else:
        group[label_col] = neg_label
        return group



data = pd.read_csv(
    os.path.join(
        output_data_folder,
        csv_filename,
    ),
    sep=";"
)


data.rename(columns=lambda x: x.replace('(case) ', ''), inplace=True)
data = data.rename(columns={
    "case:concept:name": "Case ID",
    "concept:name": "Activity",
    "org:resource": "Resource",
    "time:timestamp": "Complete Timestamp"
})

# discard cases that never have "Payment" or "Send for Credit Collection"
valid_end_activities = ["Payment", "Send for Credit Collection"]
cases_with_terminal = data.groupby(case_id_col)[activity_col].apply(lambda acts: acts.isin(valid_end_activities).any())
incomplete_cases = cases_with_terminal.index[~cases_with_terminal]
data = data[~data[case_id_col].isin(incomplete_cases)]

# add event duration
data[timestamp_col] = pd.to_datetime(data[timestamp_col])
data["timesincemidnight"] = data[timestamp_col].dt.hour * 60 + data[timestamp_col].dt.minute
data["month"] = data[timestamp_col].dt.month
data["weekday"] = data[timestamp_col].dt.weekday
data["hour"] = data[timestamp_col].dt.hour

# add features extracted from timestamp
print("Extracting timestamp features...")
sys.stdout.flush()
data = data.groupby(case_id_col, group_keys=False).apply(extract_timestamp_features)

data = data.sort_values([case_id_col, timestamp_col], ascending=[True, True], kind="mergesort").reset_index(drop=True)

# Fill forward missing values within each case
grouped = data.groupby(case_id_col)
for col in static_cols + dynamic_cols:
    data[col] = grouped[col].transform(lambda grp: grp.ffill())

# Handle categorical and remaining missing values
data[cat_cols] = data[cat_cols].fillna('missing')
data = data.fillna(0)
# set infrequent factor levels to "other"
for col in cat_cols:
    if col != activity_col:
        counts = data[col].value_counts()
        mask = data[col].isin(counts[counts >= freq_threshold].index)
        data.loc[~mask, col] = "other"

# keep only final columns
data = data[static_cols + dynamic_cols]

# assign class labels
print("Assigning class labels...")
sys.stdout.flush()
data = data.sort_values([case_id_col, timestamp_col], ascending=[True, True], kind="mergesort").reset_index(drop=True)

# Apply labeling
dt_labeled = data.groupby(case_id_col, group_keys=False).apply(
    check_if_activity_exists,
    activity="Send for Credit Collection",
    cut_from_idx=False
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
