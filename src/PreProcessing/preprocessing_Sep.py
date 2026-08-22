import pandas as pd
import numpy as np
import pm4py


# =============================================================================
# CONFIGURAZIONE
# =============================================================================

INPUT_XES = "./src/logs/train.xes"
OUTPUT_CSV = "./src/logs/sepsis_cases_preprocessed.csv"

CASE_ID = "case:concept:name"
ACTIVITY = "concept:name"
TIMESTAMP = "time:timestamp"


# =============================================================================
# CARICAMENTO LOG
# =============================================================================

log = pm4py.read_xes(INPUT_XES)

df = pm4py.convert_to_dataframe(log)

df = df.rename(columns={
    CASE_ID: "Case ID",
    ACTIVITY: "Activity",
    TIMESTAMP: "timestamp"
})


# =============================================================================
# ORDINAMENTO
# =============================================================================

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values(
    ["Case ID", "timestamp"],
    kind="mergesort"
).reset_index(drop=True)


# =============================================================================
# MANTIENI SOLO I CASI COMPLETI
# =============================================================================

release_events = {
    "Release A",
    "Release B",
    "Release C",
    "Release D",
    "Release E"
}


def has_release(trace):
    return trace["Activity"].isin(release_events).any()


valid_cases = (
    df.groupby("Case ID")
      .filter(has_release)
)


df = valid_cases.reset_index(drop=True)


# =============================================================================
# FORWARD FILL DEGLI ATTRIBUTI
# =============================================================================

df = (
    df.groupby("Case ID", group_keys=False)
      .apply(lambda g: g.ffill())
      .reset_index(drop=True)
)


# =============================================================================
# RIEMPI I NaN RIMASTI
# =============================================================================

cat_cols = df.select_dtypes(include="object").columns

num_cols = df.select_dtypes(exclude="object").columns

df[cat_cols] = df[cat_cols].fillna("missing")

df[num_cols] = df[num_cols].fillna(0)


# =============================================================================
# COSTRUZIONE recent_release
# =============================================================================

def build_recent_release(trace):

    trace = trace.sort_values("timestamp").copy()

    last_release = None

    recent_release = []

    for _, row in trace.iterrows():

        if last_release is None:
            delta = pd.Timedelta(days=9999)
        else:
            delta = row["timestamp"] - last_release

        value = int(
            row["Activity"] == "Return ER"
            and delta < pd.Timedelta(days=28)
        )

        recent_release.append(value)

        if row["Activity"] in release_events:
            last_release = row["timestamp"]

    trace["recent_release"] = recent_release

    return trace


df = (
    df.groupby("Case ID", group_keys=False)
      .apply(build_recent_release)
      .reset_index(drop=True)
)


# =============================================================================
# SALVATAGGIO
# =============================================================================

df.to_csv(
    OUTPUT_CSV,
    sep=";",
    index=False
)

print(df.head())

print()
print("Numero casi:", df["Case ID"].nunique())
print("Numero eventi:", len(df))
print("Salvato:", OUTPUT_CSV)