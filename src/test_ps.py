"""
Script pm4py per:
1) Discretizzare la durata di ogni traccia in 3 quantili: low, medium, high
2) Filtrare le VARIANTI UNICHE che hanno durata "high"
3) Estrarre tutti i prefissi possibili di ciascuna variante filtrata

Requisiti:
    pip install pm4py pandas
"""

import pandas as pd
import pm4py
from pm4py.objects.log.obj import EventLog


def discretize_and_label(log: EventLog, n_quantiles: int = 3):
    """
    Calcola la durata (in secondi) di ogni traccia e la discretizza usando
    pd.qcut (stesso criterio dello script discretize.py di riferimento):
    - quantili di uguale frequenza (terzili di default)
    - se ci sono troppi valori duplicati ai bordi, qcut riduce
      automaticamente il numero di bin (duplicates="drop")
    - se non riesce a creare almeno n_quantiles bin distinti, la durata
      viene trattata come categorica "as-is" (nessuna etichetta low/medium/high)

    Assegna il risultato come attributo trace.attributes["time_label"].
    """
    labels = ["low", "medium", "high"] if n_quantiles == 3 else None

    # Prima passata: calcolo le durate di tutte le tracce
    durations = []
    for trace in log:
        start_ts = trace[0]["time:timestamp"]
        end_ts = trace[-1]["time:timestamp"]
        duration = (end_ts - start_ts).total_seconds()
        durations.append(duration)
        trace.attributes["duration_seconds"] = duration

    durations_series = pd.Series(durations)

    n_unique = durations_series.nunique(dropna=True)

    if n_unique < n_quantiles:
        print(
            f"ATTENZIONE: le durate hanno solo {n_unique} valore/i unico/i, "
            f"impossibile creare {n_quantiles} quantili distinti. "
            f"Verrà usata la durata grezza come etichetta."
        )
        discretized = durations_series.astype("string")
    else:
        discretized = pd.qcut(
            durations_series,
            q=n_quantiles,
            labels=labels,
            duplicates="drop",
        )

        n_bins = discretized.cat.categories.size if hasattr(discretized, "cat") else n_unique

        if n_bins < n_quantiles:
            print(
                f"ATTENZIONE: qcut ha creato solo {n_bins} bin invece di "
                f"{n_quantiles} (troppi valori duplicati ai bordi dei quantili). "
                f"Verrà usata la durata grezza come etichetta."
            )
            discretized = durations_series.astype("string")

    # Seconda passata: assegno l'etichetta calcolata a ogni traccia
    for trace, label in zip(log, discretized):
        trace.attributes["time_label"] = label

    return log


def filter_unique_high_variants(
    log: EventLog,
    required_any_of: set = {"A_CANCELLED", "A_DECLINED"}
):
    
    
    variants = pm4py.get_variants(log)  # dict: variante -> lista di tracce

    unique_high_traces = []
    for variant_key, traces_of_variant in variants.items():
        representative = traces_of_variant[0]

        is_high = representative.attributes.get("time_label") == "high"

        activities = set(event["concept:name"] for event in representative)
        has_negative_outcome = bool(activities & required_any_of)

        if is_high and has_negative_outcome:
            unique_high_traces.append(representative)

    return unique_high_traces

"""def filter_unique_high_variants(log: EventLog, excluded_activity: str = "A_APPROVED"):
    
    Restituisce, per ogni variante presente nel log, UNA traccia
    rappresentativa (per evitare i duplicati), mantenendo solo le
    varianti che soddisfano ENTRAMBE le condizioni:
      - la traccia rappresentativa ha time_label == 'high'
      - la variante NON contiene l'attività `excluded_activity`
        (es. 'A_APPROVED', l'outcome desiderato/positivo del processo)

    In questo modo si isolano le varianti "lente" che NON sono andate
    a buon fine (niente A_APPROVED), utili ad es. per capire i pattern
    di comportamento anomalo/indesiderato.
    
    variants = pm4py.get_variants(log)  # dict: variante -> lista di tracce

    unique_high_traces = []
    for variant_key, traces_of_variant in variants.items():
        representative = traces_of_variant[0]

        is_high = representative.attributes.get("time_label") == "high"

        activities = [event["concept:name"] for event in representative]
        has_desired_outcome = excluded_activity in activities

        if is_high and not has_desired_outcome:
            unique_high_traces.append(representative)

    return unique_high_traces"""

def get_all_prefixes(trace, step: int = 2):
    """
    Dato un oggetto Trace (o una lista di nomi di attività),
    restituisce i prefissi di lunghezza 2, 4, 6, ... (multipli di `step`),
    fermandosi al massimo alla penultima attività (mai la traccia intera).

    Es: [a, b, c, d, e, f, g] (len=7) -> [a,b], [a,b,c,d], [a,b,c,d,e,f]
    Es: [a, b, c, d, e, f] (len=6) -> [a,b], [a,b,c,d]  (non [a,b,c,d,e,f], che sarebbe l'intera traccia)
    """
    activities = [event["concept:name"] for event in trace]
    prefixes = [activities[:i] for i in range(step, len(activities), step)]
    return prefixes


def load_mdp_transitions(mdp_csv_path: str):
    """
    Carica le transizioni dell'MDP da CSV (colonne: state, action_index,
    action, next_state, probability) e costruisce un dizionario:
        (state, action) -> lista di (next_state, probability)
    Nota: essendo l'MDP non deterministico, per una stessa coppia
    (state, action) possono esserci più next_state possibili.
    """
    mdp_df = pd.read_csv(mdp_csv_path)

    transition_dict = {}
    for _, row in mdp_df.iterrows():
        key = (row["state"], row["action"])
        transition_dict.setdefault(key, []).append((row["next_state"], row["probability"]))

    return transition_dict


def get_state_for_prefix(prefix, transition_dict, start_state=0):
    """
    Trova un percorso completo per un prefisso nell'MDP.

    Le transizioni vengono considerate in ordine di probabilità decrescente.
    Se la transizione più probabile non permette di completare il prefisso,
    vengono provate le altre.

    Restituisce:
        (stato_finale, probabilita_cumulata, valido)
    """

    from functools import lru_cache

    prefix = tuple(prefix)

    @lru_cache(maxsize=None)
    def search(position, current_state):
        # Prefisso completato
        if position == len(prefix):
            return current_state, 1.0

        action = prefix[position]
        key = (current_state, action)

        if key not in transition_dict:
            return None

        # Prima le transizioni più probabili
        transitions = sorted(
            transition_dict[key],
            key=lambda x: x[1],
            reverse=True
        )

        best_result = None
        best_probability = -1.0

        for next_state, probability in transitions:

            result = search(position + 1, next_state)

            if result is None:
                continue

            final_state, future_probability = result
            total_probability = probability * future_probability

            if total_probability > best_probability:
                best_probability = total_probability
                best_result = (
                    final_state,
                    total_probability
                )

        return best_result

    result = search(0, start_state)

    if result is None:
        return start_state, 0.0, False

    final_state, probability = result

    return final_state, probability, True


def save_prefixes_to_csv(
    high_variant_traces,
    output_path: str,
    transition_dict=None,
    start_state=0
):
    """
    Salva i prefissi validi delle varianti high.

    Per ogni variante il percorso nell'MDP viene costruito
    progressivamente:

        A
        A|B
        A|B|C
        A|B|C|D

    Il prefisso successivo riutilizza i risultati del prefisso precedente,
    evitando di ripartire ogni volta da start_state.

    Se esistono più transizioni possibili, vengono mantenuti gli stati
    alternativi raggiungibili e vengono eliminate solo le diramazioni
    che non permettono di continuare.

    I prefissi per i quali non esiste alcun percorso nell'MDP
    vengono eliminati.
    """

    rows = []
    skipped_prefixes = 0

    for variant_id, trace in enumerate(high_variant_traces):

        activities = [event["concept:name"] for event in trace]
        variant_str = "|".join(activities)

        duration = trace.attributes.get("duration_seconds")
        time_label = trace.attributes.get("time_label")

        # ---------------------------------------------------------
        # Stati raggiungibili per il prefisso corrente.
        #
        # Ogni elemento è:
        #     state -> probability
        #
        # All'inizio siamo nello start_state con probabilità 1.
        # ---------------------------------------------------------
        current_paths = {
            start_state: 1.0
        }

        prefixes = get_all_prefixes(trace)

        for prefix in prefixes:

            action = prefix[-1]

            # Nuovi stati raggiungibili dopo l'ultima attività
            next_paths = {}

            # -----------------------------------------------------
            # Estendiamo tutti i percorsi possibili precedenti
            # con l'azione corrente.
            # -----------------------------------------------------
            for current_state, current_probability in current_paths.items():

                key = (current_state, action)

                if key not in transition_dict:
                    continue

                possible_transitions = transition_dict[key]

                for next_state, probability in possible_transitions:

                    new_probability = (
                        current_probability * probability
                    )

                    # Se lo stesso stato è raggiunto attraverso
                    # più percorsi, teniamo quello con probabilità
                    # maggiore.
                    if (
                        next_state not in next_paths
                        or new_probability > next_paths[next_state]
                    ):
                        next_paths[next_state] = new_probability

            # -----------------------------------------------------
            # Nessun percorso possibile per questo prefisso.
            # -----------------------------------------------------
            if not next_paths:
                skipped_prefixes += 1

                # Da questo punto in poi i prefissi successivi
                # della stessa variante non possono essere validi,
                # perché sono estensioni di questo prefisso.
                break

            # -----------------------------------------------------
            # Il prefisso è valido.
            # -----------------------------------------------------
            current_paths = next_paths

            # Stato finale: quello con probabilità cumulata maggiore
            mdp_state, path_probability = max(
                current_paths.items(),
                key=lambda x: x[1]
            )

            row = {
                "variant_id": variant_id,
                "variant": variant_str,
                "duration_seconds": duration,
                "time_label": time_label,
                "prefix_length": len(prefix),
                "prefix": "|".join(prefix),
                "mdp_state": mdp_state,
                "path_probability": path_probability,
                "mdp_path_valid": True,
            }

            rows.append(row)

    print(
        f"Prefissi eliminati perché senza path nell'MDP: "
        f"{skipped_prefixes}"
    )

    df = pd.DataFrame(rows)

    df.to_csv(output_path, index=False)

    print(
        f"Salvati {len(df)} prefissi in '{output_path}'"
    )

    return df

def save_prefixes_to_csv(high_variant_traces, output_path: str, transition_dict=None, start_state=0):
    """
    Salva tutti i prefissi in un CSV, una riga per prefisso, con colonne:
    - variant_id: id progressivo della variante
    - variant: sequenza completa di attività della variante (separata da '|')
    - duration_seconds / time_label: informazioni sulla traccia rappresentativa
    - prefix_length: lunghezza del prefisso (1, 2, 3, ...)
    - prefix: attività del prefisso separate da '|'
    - mdp_state: stato dell'MDP raggiunto seguendo il prefisso (se transition_dict è fornito)
    - path_probability: probabilità cumulata del percorso più probabile seguito nell'MDP
    - mdp_path_valid: False se il prefisso non è interamente rappresentato nell'MDP
    """
    rows = []
    skipped_prefixes = 0
    for variant_id, trace in enumerate(high_variant_traces):
        activities = [event["concept:name"] for event in trace]
        variant_str = "|".join(activities)
        duration = trace.attributes.get("duration_seconds")
        time_label = trace.attributes.get("time_label")

        prefixes = get_all_prefixes(trace)
        for prefix in prefixes:
            row = {
                "variant_id": variant_id,
                "variant": variant_str,
                "duration_seconds": duration,
                "time_label": time_label,
                "prefix_length": len(prefix),
                "prefix": "|".join(prefix),
            }

            if transition_dict is not None:
                mdp_state, path_probability, valid = get_state_for_prefix(
                    prefix, transition_dict, start_state=start_state
                )
                if not valid:
                    skipped_prefixes += 1
                    continue
                row["mdp_state"] = mdp_state
                row["path_probability"] = path_probability
                row["mdp_path_valid"] = valid

            rows.append(row)
    print(f"Prefissi eliminati perché senza path nell'MDP: {skipped_prefixes}")

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Salvati {len(df)} prefissi in '{output_path}'")
    return df


def main(log_path: str, mdp_csv_path: str, output_csv_path: str = "high_variant_prefixes.csv"):
    # --- Caricamento del log (adatta al tuo formato: xes, csv, ecc.) ---
    # Nota: dalle versioni recenti di pm4py, read_xes restituisce di default
    # un pandas DataFrame. Per usare l'API a tracce (trace[0]["time:timestamp"],
    # ecc.) dobbiamo convertirlo esplicitamente in un EventLog.
    log = pm4py.read_xes(log_path)
    log = pm4py.convert_to_event_log(log)

    # --- 1) Discretizzazione in low/medium/high (stesso criterio di discretize.py: pd.qcut) ---
    log = discretize_and_label(log, n_quantiles=3)

    # --- 2) Filtro varianti uniche con label 'high' ---
    high_variant_traces = filter_unique_high_variants(log, required_any_of={"A_CANCELLED", "A_DECLINED"})
    #print(f"Trovate {len(high_variant_traces)} varianti uniche con durata 'high' e outcome CANCELLED/DECLINED")

    #high_variant_traces = filter_unique_high_variants(log, excluded_activity="A_APPROVED")
    print(f"Trovate {len(high_variant_traces)} varianti uniche con durata 'high' e SENZA A_APPROVED")

    # --- 3) Caricamento MDP e mapping prefisso -> stato ---
    transition_dict = load_mdp_transitions(mdp_csv_path)

    # --- 4) Estrazione di tutti i prefissi (+ stato MDP) e salvataggio su CSV ---
    df = save_prefixes_to_csv(high_variant_traces, output_csv_path, transition_dict=transition_dict, start_state=0)

    invalid_count = (~df["mdp_path_valid"]).sum() if "mdp_path_valid" in df else 0
    if invalid_count > 0:
        print(f"Attenzione: {invalid_count} prefissi non trovano un percorso completo nell'MDP")
    

    return df


if __name__ == "__main__":
    # Sostituisci con i path dei tuoi file
    main("logs/split/bpi12/test.xes.gz", "src/output/bpi12/mdp_transitions_total.csv", "high_variant_prefixes.csv")