"""
Uso:
    python src/discretize.py \
        input.csv \
        output.csv \
        --quantiles 3 \
        --exclude col1 col2
        -- dataset (traffic_fines, sepsis)
"""

import pandas as pd
import argparse
import os


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Discretize numeric columns of a CSV using quantiles."
)

parser.add_argument(
    "input_csv",
    help="Path to the input CSV file"
)

parser.add_argument(
    "output_csv",
    help="Path to the output CSV file"
)

parser.add_argument(
    "--dataset",
    choices=["rtf", "sepsis", "bpi12", "permit", "intDecl"],
    required=True,
    help="Dataset type: sepsis, road traffic fines (rtf), BPI challenge 2012 (bpi12), Permit Log (permit), or international Declarations (intDecl)"
)

parser.add_argument(
    "--quantiles",
    type=int,
    default=3,
    help="Number of quantile intervals (default: 3)"
)

parser.add_argument(
    "--exclude",
    nargs="*",
    default=[],
    help="Numeric columns to exclude from discretization"
)

args = parser.parse_args()
# ============================================================
# FIXED DATASET OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = "./src/output"

dataset_dir = os.path.join(
    OUTPUT_DIR,
    args.dataset
)

os.makedirs(
    dataset_dir,
    exist_ok=True
)

input_path = os.path.join(
    dataset_dir,
    args.input_csv
)

output_path = os.path.join(
    dataset_dir,
    args.output_csv
)

# ============================================================
# READ CSV
# ============================================================

df = pd.read_csv(input_path)


# ============================================================
# DATASET-SPECIFIC CONFIGURATION
# ============================================================

if args.dataset == "rtf":

    # Columns to remove completely
    columns_to_remove = [
        "timesincecasestart",
        "Resource",
    ]

    # Values to interpret as missing
    missing_values = [
        "missing",
        "NIL"
    ]

    # Numeric columns that are actually categorical
    categorical_columns = [
        "article",
        "vehicleClass"
    ]

    if "article" in df.columns:
        df["article"] = df["article"].astype("string")

elif args.dataset == "sepsis":

    columns_to_remove = ["org:group", "Age"]

    missing_values = []

    categorical_columns = []

elif args.dataset == "bpi12":

    columns_to_remove = ["org:resource"]

    missing_values = []

    categorical_columns = []

elif args.dataset == "permit":

    columns_to_remove = ["Resource", "Role", "Task_0", "RfpNumber_0"]

    missing_values = ["missing"]

    categorical_columns = []

elif args.dataset == "intDecl":

    columns_to_remove = ["Resource", "Role"]

    missing_values = []

    categorical_columns = []



# ============================================================
# REMOVE DATASET-SPECIFIC COLUMNS
# ============================================================

for col in columns_to_remove:

    if col in df.columns:

        df = df.drop(columns=col)

        print(f"Removed column: {col}")

    else:

        print(
            f"WARNING: column '{col}' "
            f"not found in the input file."
        )


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

if missing_values:

    df = df.replace(
        missing_values,
        pd.NA
    )

    print(
        "\nReplaced the following values with missing values:"
    )

    for value in missing_values:
        print(f"  - {value}")


# ============================================================
# AUTOMATICALLY IDENTIFY NUMERIC COLUMNS
# ============================================================

numeric_columns = df.select_dtypes(
    include=["number"]
).columns.tolist()


# Remove categorical columns
numeric_columns = [
    col
    for col in numeric_columns
    if col not in categorical_columns
]


# Remove excluded columns
numeric_columns = [
    col
    for col in numeric_columns
    if col not in args.exclude
]


# ============================================================
# PRINT COLUMN INFORMATION
# ============================================================

print("\nDataset:")
print(f"  {args.dataset}")


print("\nCategorical columns:")
if categorical_columns:

    for col in categorical_columns:
        if col in df.columns:
            print(f"  - {col}")

else:

    print("  None")


print("\nNumeric columns to discretize:")

if numeric_columns:

    for col in numeric_columns:
        print(f"  - {col}")

else:

    print("  None")


if args.exclude:

    print("\nExcluded columns:")

    for col in args.exclude:
        print(f"  - {col}")


# ============================================================
# DISCRETIZATION
# ============================================================

if args.quantiles == 3:

    labels = [
        "low",
        "medium",
        "high"
    ]

else:

    labels = None


for col in numeric_columns:

    # --------------------------------------------------------
    # Number of unique non-missing values
    # --------------------------------------------------------

    n_unique = df[col].nunique(
        dropna=True
    )

    # --------------------------------------------------------
    # Not enough unique values
    # --------------------------------------------------------

    if n_unique < args.quantiles:

        print(
            f"\nColumn '{col}' has only "
            f"{n_unique} unique value(s)."
        )

        print(
            f"Treating '{col}' as categorical."
        )

        # Use pandas StringDtype so that NA remains missing
        df[col] = df[col].astype("string")

        continue


    # --------------------------------------------------------
    # Quantile discretization
    # --------------------------------------------------------

    try:

        discretized = pd.qcut(
            df[col],
            q=args.quantiles,
            labels=labels,
            duplicates="drop"
        )


        # ----------------------------------------------------
        # Check how many bins were actually created
        # ----------------------------------------------------

        n_bins = discretized.cat.categories.size


        # ----------------------------------------------------
        # qcut created fewer bins than requested
        # ----------------------------------------------------

        if n_bins < args.quantiles:

            print(
                f"\nColumn '{col}': qcut could only create "
                f"{n_bins} bin(s) instead of "
                f"{args.quantiles}."
            )

            print(
                f"Treating '{col}' as categorical."
            )

            df[col] = df[col].astype("string")


        # ----------------------------------------------------
        # Successful discretization
        # ----------------------------------------------------

        else:

            df[col] = discretized

            print(
                f"\nColumn '{col}' discretized into "
                f"{n_bins} quantile bins."
            )


    # --------------------------------------------------------
    # qcut error
    # --------------------------------------------------------

    except ValueError as e:

        print(
            f"\nWARNING: could not discretize "
            f"column '{col}': {e}"
        )

        print(
            f"Treating '{col}' as categorical."
        )

        df[col] = df[col].astype("string")


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    output_path,
    index=False,
    na_rep=""
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n----------------------------------------")
print("Processing completed.")
print("----------------------------------------")

print("\nOutput file:")
print(output_path)

print("\nFinal columns:")
for col in df.columns:
    print(f"  - {col}")