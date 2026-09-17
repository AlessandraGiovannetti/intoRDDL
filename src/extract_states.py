"""
Prende i prefissi high + alarm individuati dal predictive monitoring
e determina lo stato dell'MDP corrispondente a ciascun prefisso.

Input:
    alarm_results/bpi12_O_DECLINED_alarm_high_unique_variants.csv

MDP:
    src/output/bpi12/mdp_transitions_total.csv

Output:
    bpi12_alarm_prefixes_mdp_states.csv
"""

import pandas as pd
from functools import lru_cache


# ============================================================
# CONFIGURATION
# ============================================================

ALARM_FILE = (
    "src/output/bpi12/"
    "bpi12_O_DECLINED_alarm_high_unique_variants.csv"
)

MDP_FILE = (
    "src/output/bpi12/"
    "mdp_transitions.csv"
)

OUTPUT_FILE = "bpi12_alarm_prefixes_mdp_states.csv"

START_STATE = 0


# ============================================================
# LOAD MDP
# ============================================================

def load_mdp_transitions(mdp_csv_path):
    """
    Carica le transizioni dell'MDP.

    CSV atteso:
        state
        action_index
        action
        next_state
        probability

    Restituisce:

        (state, action) ->
            [(next_state, probability), ...]
    """

    mdp_df = pd.read_csv(mdp_csv_path)

    transition_dict = {}

    for _, row in mdp_df.iterrows():

        key = (
            row["state"],
            row["action"]
        )

        transition_dict.setdefault(
            key,
            []
        ).append(
            (
                row["next_state"],
                row["probability"]
            )
        )

    return transition_dict


# ============================================================
# FIND MDP STATE FOR PREFIX
# ============================================================

def get_state_for_prefix(
    prefix,
    transition_dict,
    start_state=0
):
    """
    Trova lo stato MDP corrispondente al prefisso.

    Se esistono più percorsi possibili, viene mantenuto
    quello con probabilità cumulata maggiore.

    Restituisce:

        final_state
        cumulative_probability
        valid
    """

    prefix = tuple(prefix)

    @lru_cache(maxsize=None)
    def search(position, current_state):

        # ----------------------------------------------------
        # Prefix completely processed
        # ----------------------------------------------------

        if position == len(prefix):
            return current_state, 1.0

        action = prefix[position]

        key = (
            current_state,
            action
        )

        if key not in transition_dict:
            return None

        transitions = sorted(
            transition_dict[key],
            key=lambda x: x[1],
            reverse=True
        )

        best_result = None
        best_probability = -1.0

        for next_state, probability in transitions:

            result = search(
                position + 1,
                next_state
            )

            if result is None:
                continue

            final_state, future_probability = result

            total_probability = (
                probability *
                future_probability
            )

            if total_probability > best_probability:

                best_probability = total_probability

                best_result = (
                    final_state,
                    total_probability
                )

        return best_result

    result = search(
        0,
        start_state
    )

    if result is None:

        return (
            start_state,
            0.0,
            False
        )

    final_state, probability = result

    return (
        final_state,
        probability,
        True
    )


# ============================================================
# PROCESS ALARM FILE
# ============================================================

def process_alarm_prefixes(
    alarm_file,
    transition_dict,
    output_file,
    start_state=0
):

    # --------------------------------------------------------
    # Load alarm results
    # --------------------------------------------------------

    df = pd.read_csv(
        alarm_file,
        sep=";"
    )

    print(
        f"Prefissi caricati dal file alarm: {len(df)}"
    )

    # --------------------------------------------------------
    # Check required column
    # --------------------------------------------------------

    if "prefix" not in df.columns:

        raise ValueError(
            "Il file alarm non contiene la colonna 'prefix'."
        )

    rows = []

    invalid_count = 0

    # --------------------------------------------------------
    # Process every prefix
    # --------------------------------------------------------

    for _, row in df.iterrows():

        prefix_string = str(
            row["prefix"]
        )

        activities = [
            activity.strip()
            for activity in prefix_string.split(">")
        ]

        # ----------------------------------------------------
        # Find MDP state
        # ----------------------------------------------------

        mdp_state, path_probability, valid = (
            get_state_for_prefix(
                activities,
                transition_dict,
                start_state=start_state
            )
        )

        if not valid:

            invalid_count += 1

            print(
                "\nPrefisso senza percorso MDP:"
            )

            print(
                prefix_string
            )

            continue

        # ----------------------------------------------------
        # Copy original alarm information
        # ----------------------------------------------------

        output_row = row.to_dict()

        output_row[
            "mdp_state"
        ] = mdp_state

        output_row[
            "path_probability"
        ] = path_probability

        output_row[
            "mdp_path_valid"
        ] = True

        rows.append(
            output_row
        )

    # --------------------------------------------------------
    # Create output dataframe
    # --------------------------------------------------------

    result = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result.to_csv(
        output_file,
        sep=";",
        index=False
    )

    print(
        "\n=========================================="
    )

    print(
        f"Prefissi validi: {len(result)}"
    )

    print(
        f"Prefissi senza percorso MDP: {invalid_count}"
    )

    print(
        f"Salvato: {output_file}"
    )

    print(
        "=========================================="
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\nCaricamento transizioni MDP..."
    )

    transition_dict = load_mdp_transitions(
        MDP_FILE
    )

    print(
        f"Transizioni caricate: "
        f"{len(transition_dict)} coppie stato-azione"
    )

    process_alarm_prefixes(
        ALARM_FILE,
        transition_dict,
        OUTPUT_FILE,
        start_state=START_STATE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()