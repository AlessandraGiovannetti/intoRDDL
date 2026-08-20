import pandas as pd
import argparse

"""
Uso:
    python discretize.py \
        input.csv \
        output.csv \
        --quantiles 3 \
        --exclude col1 col2
"""

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


df = pd.read_csv(args.input_csv)


numeric_columns = df.select_dtypes(
    include=["number"]
).columns.tolist()


# Remove excluded columns
numeric_columns = [
    col for col in numeric_columns
    if col not in args.exclude
]


print("Numeric columns to discretize:")
for col in numeric_columns:
    print(f"  - {col}")

if args.exclude:
    print("\nExcluded columns:")
    for col in args.exclude:
        print(f"  - {col}")


if args.quantiles == 3:
    labels = ["low", "medium", "high"]
else:
    labels = None


for col in numeric_columns:

    try:
        df[col] = pd.qcut(
            df[col],
            q=args.quantiles,
            labels=labels,
            duplicates="drop"
        )

    except ValueError as e:
        print(
            f"WARNING: could not discretize "
            f"column '{col}': {e}"
        )


df.to_csv(args.output_csv, index=False)

print("\nDiscretized file saved to:")
print(args.output_csv)