"""
encoding.py
=======================

Genera domain.rddl + instance.rddl nello stile minimale compatibile con
PROST:
  - nessun "types" block, nessuna pvariable parametrizzata, nessun
    interm-fluent (PROST li rifiuta con errori di parsing);
  - uno state-fluent bool scalare per ciascuno stato (s0, s1, ...);
  - NESSUN blocco action-preconditions;
  - DI DEFAULT NESSUN ATTRIBUTO: solo control-flow (stati + transizioni).
    Gli attributi (state-fluent real, if/else bilanciato) sono
    disponibili con --include-attributes;
  - reward di default: +terminal_bonus (10.0) se si raggiunge uno stato
    terminale, altrimenti 1.0. 
  - le probabilita' di transizione osservate nel log sono scritte come
    costanti letterali dentro Bernoulli(p);
  - le azioni sono action-fluent bool scalari separate (una per
    attivita' osservata).

Uso:
    python encoding.py \
        --states mdp_states_described_test_discretized.csv \ (usare la versione discretizzata)
        --transitions mdp_transitions_test.csv \
        --outdir out_prost/ \
        --horizon 20

Per reintrodurre gli attributi:
    python encoding.py ... --include-attributes
"""

import argparse
import os
import re
from collections import defaultdict

import pandas as pd
import numpy as np


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def sanitize_pvar_name(name: str) -> str:
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1-\2', str(name).strip())
    s = re.sub(r'[^0-9a-zA-Z]+', '-', s)
    s = re.sub(r'-+', '-', s).strip('-').lower()
    if not s:
        # es. '?' o altri valori senza alcun carattere alfanumerico
        s = 'na'
    elif not s[0].isalpha():
        s = 'v-' + s
    return s


def cat_label(v) -> str:
    """Etichetta leggibile per un valore categorico usato in un nome di
    pvariable one-hot. I float che rappresentano interi (es. 3.0, letti da
    un CSV con colonna numerica) diventano '3' invece di '3-0'; gli altri
    float usano una rappresentazione compatta (%g)."""
    if isinstance(v, (float, np.floating)):
        fv = float(v)
        if fv.is_integer():
            return str(int(fv))
        return f"{fv:g}"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v)


def uniquify(names):
    seen = defaultdict(int)
    out = []
    for n in names:
        seen[n] += 1
        out.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    return out


def or_tree(conditions):
    """OR bilanciato di una lista di espressioni booleane (stringhe), profondita' O(log n)
    invece di O(n) come una catena lineare 'a | b | c | ... '."""
    conditions = list(conditions)
    if len(conditions) == 1:
        return conditions[0]
    mid = len(conditions) // 2
    left = or_tree(conditions[:mid])
    right = or_tree(conditions[mid:])
    return f"({left}) | ({right})"


def sum_tree(terms):
    """Somma bilanciata di una lista di espressioni aritmetiche (stringhe),
    profondita' O(log n) invece di O(n) come 'a + b + c + ... '."""
    terms = list(terms)
    if len(terms) == 1:
        return terms[0]
    mid = len(terms) // 2
    left = sum_tree(terms[:mid])
    right = sum_tree(terms[mid:])
    return f"({left}) + ({right})"


def balanced_chain(pairs, else_expr):
    """
    Equivalente semantico di:
        if (guard0) then value0
        else if (guard1) then value1
        ...
        else else_expr
    ma strutturato come albero bilanciato (profondita' O(log n)) invece di una
    catena lineare (profondita' O(n)): i guard sono a due a due mutuamente
    esclusivi (per costruzione, nel nostro caso), quindi instradare con un OR
    bilanciato dei guard di "meta' sinistra" e ricorrere e' semanticamente
    identico alla catena originale.

    Ottimizzazione: quando un ramo (sinistro o destro) contiene UN SOLO
    guard, il fatto stesso di essere instradati li' garantisce gia' che quel
    guard sia vero (e' esattamente route_cond) - non serve ritestarlo con un
    ulteriore if annidato identico (che genererebbe "if (g) then (if (g)
    then v else e) else ..." con g duplicato). Si usa direttamente il valore.
    """
    pairs = list(pairs)
    if not pairs:
        return else_expr
    if len(pairs) == 1:
        guard, value = pairs[0]
        return f"if ({guard}) then {value} else {else_expr}"
    mid = len(pairs) // 2
    left, right = pairs[:mid], pairs[mid:]
    route_cond = or_tree([g for g, _ in left])
    left_expr = left[0][1] if len(left) == 1 else balanced_chain(left, else_expr)
    right_expr = balanced_chain(right, else_expr)
    return f"if ({route_cond}) then ({left_expr}) else ({right_expr})"


TRI_VALUES = {'True', 'False', 'missing'}
TRI_CODE = {'True': 1.0, 'False': 0.0, 'missing': -1.0}
TRI_BOOL = {'True': True, 'False': False, 'missing': False}  # missing -> false


def classify_columns(states_df, id_col, skip_cols, numeric_cat_max_unique=20):
    """Classifica ciascuna colonna di attributo in:
      - 'tri'  : booleano/tri-state (True/False/missing) -> singolo bool
      - 'cat'  : categorica -> one-hot booleano (UN booleano per valore)

    NESSUNA colonna viene piu' classificata come 'real': anche le colonne
    numeriche (int/float) vengono trattate come categoriche, assumendo che
    se non sono gia' state discretizzate a monte i valori distinti presenti
    siano comunque pochi. Se una colonna numerica supera
    `numeric_cat_max_unique` valori distinti, viene sollevato un errore
    esplicito (probabile colonna continua non discretizzata, non adatta a
    un encoding one-hot).
    """
    attrs = []
    for col in states_df.columns:
        if col == id_col or col in skip_cols:
            continue
        series = states_df[col]
        if pd.api.types.is_bool_dtype(series):
            # pandas converte automaticamente in bool nativo le colonne che
            # contengono SOLO 'True'/'False' in ogni riga (nessun 'missing' da
            # nessuna parte nel file) - senza questo controllo esplicito,
            # finivano classificate come 'cat' invece di 'tri'
            kind = 'tri'
        elif pd.api.types.is_float_dtype(series) or pd.api.types.is_integer_dtype(series):
            # niente 'real': colonna numerica -> categorica one-hot.
            n_unique = series.dropna().nunique()
            if n_unique > numeric_cat_max_unique:
                raise ValueError(
                    f"Colonna '{col}' e' numerica con {n_unique} valori distinti "
                    f"(> --numeric-cat-max-unique={numeric_cat_max_unique}): sembra "
                    f"continua/non discretizzata. Discretizzala a monte (bin/quantili) "
                    f"prima di generare l'RDDL, oppure alza la soglia se e' voluto."
                )
            kind = 'cat'
        else:
            vals = set(series.dropna().unique().tolist())
            kind = 'tri' if vals.issubset(TRI_VALUES) else 'cat'
        attrs.append({'col': col, 'kind': kind, 'pvar': sanitize_pvar_name(col)})
    pvars = uniquify([a['pvar'] for a in attrs])
    for a, p in zip(attrs, pvars):
        a['pvar'] = p
    return attrs


def build_cat_onehot_names(states_df, attrs, reserved_names):
    """Per ogni colonna 'cat' assegna un nome di pvariable bool per ciascun
    valore distinto (one-hot), es. 'crp-low', 'crp-medium', 'crp-high', oppure
    (per colonne numeriche trattate come categoriche) 'age-group-3',
    'age-group-12'. Ritorna {col: {valore_originale: nome_pvar}}. I nomi sono
    resi univoci anche rispetto a `reserved_names` (stati, azioni, altri
    attributi)."""
    names = {}
    all_new = []
    per_col_vals = {}
    for a in attrs:
        if a['kind'] == 'cat':
            col = a['col']
            vals = sorted(states_df[col].dropna().unique().tolist())
            per_col_vals[col] = vals
            for v in vals:
                all_new.append(f"{a['pvar']}-{sanitize_pvar_name(cat_label(v))}")
    uniq = uniquify(list(reserved_names) + all_new)[len(reserved_names):]
    i = 0
    for a in attrs:
        if a['kind'] == 'cat':
            col = a['col']
            names[col] = {}
            for v in per_col_vals[col]:
                names[col][v] = uniq[i]
                i += 1
    return names


def attr_value_code(a, row):
    """Valore 'grezzo' dell'attributo `a` nello stato `row`. Per 'tri' ritorna
    un bool Python (missing -> False, gestendo sia stringhe che bool nativi
    letti da pandas); per 'cat' (incluse le colonne numeriche trattate come
    categoriche) ritorna il valore originale cosi' com'e' (stringa o
    numero), usato solo per il confronto di uguaglianza nel one-hot."""
    col = a['col']
    if a['kind'] == 'tri':
        val = row[col]
        if isinstance(val, (bool, np.bool_)):
            return bool(val)
        if pd.isna(val):
            return False
        return TRI_BOOL[val]
    else:
        return row[col]


# --------------------------------------------------------------------------
# Costruzione delle clausole di transizione (stick-breaking per gruppo sorgente)
# --------------------------------------------------------------------------

def build_target_clauses(trans_df, id_col, action_col, action_fluent_of, n_states):
    """
    Ritorna:
      - target_clauses: {target_state_idx: [(guard_expr, value_expr), ...]}
        usato per le CPF di transizione s{K}' (value_expr e' "true" o "Bernoulli(p)")
      - flat_clauses: [(guard_expr, target_state_idx), ...] con UNA entry per ogni
        (source,action) osservata nel log -> l'esito PIU' PROBABILE per quel
        gruppo. NESSUN riferimento a fluent primati (niente s{k}'): dipende solo
        da (stato corrente, azione). Necessario perche' anche un solo attributo
        che referenzia s{k}' causa segfault in PROST quando la reward referenzia
        anch'essa s{k}' (osservato empiricamente, indipendentemente da dimensione
        o profondita' della formula).
        Compromesso: per i gruppi con piu' di un esito possibile (53 su 338 nel
        dataset di test), l'attributo riflette l'esito piu' probabile anche nei
        rari casi in cui la transizione REALE (quella si', esatta, guidata da
        Bernoulli) e' finita su un esito diverso - la transizione di stato resta
        sempre esatta, solo l'attributo puo' essere "tipico" invece che esatto
        in quel sottoinsieme di step.

    Per target_clauses i guard sono costruiti ordinando i target per indice di
    stato crescente, cosi' che le dipendenze (riferimenti a s{T}' di target con
    indice minore) formino sempre un grafo aciclico.
    """
    target_clauses = defaultdict(list)
    flat_clauses = []

    groups = trans_df.groupby([id_col, action_col])
    for (src, act), grp in groups:
        grp_sorted = grp.sort_values('next_state')
        rows = list(grp_sorted[['next_state', 'probability']].itertuples(index=False, name=None))
        act_lit = action_fluent_of[act]
        cum = 0.0
        m = len(rows)
        for i, (tgt, prob) in enumerate(rows):
            earlier_targets = [t for t, _ in rows[:i]]
            guard_parts = [f"s{src}", act_lit] + [f"~s{t}'" for t in earlier_targets]
            guard = " ^ ".join(guard_parts)
            if m == 1 or i == m - 1:
                value = "true"
            else:
                remaining = max(1.0 - cum, 1e-9)
                cond_p = min(max(prob / remaining, 0.0), 1.0)
                value = f"Bernoulli({cond_p:.6f})"
            target_clauses[tgt].append((guard, value))
            cum += prob

        # per gli attributi: un solo guard per l'intero gruppo (stato,azione),
        # SENZA riferimenti a s{k}' - se ci sono piu' esiti possibili, si usa
        # quello con probabilita' massima
        best_tgt, best_prob = max(rows, key=lambda r: r[1])
        flat_clauses.append((f"s{src} ^ {act_lit}", best_tgt))

    return target_clauses, flat_clauses


# --------------------------------------------------------------------------
# DOMAIN
# --------------------------------------------------------------------------

def build_domain(domain_name, bool_fluents, n_states, action_names,
                  action_fluent_of, target_clauses, flat_clauses, terminal_idx, init_idx,
                  valid_states_of_action, include_attrs=True,
                  terminal_bonus=None):

    L = []
    L.append(f"domain {domain_name} {{")
    L.append("")
    L.append("    requirements = {")
    L.append("        reward-deterministic")
    L.append("    };")
    L.append("")
    L.append("    pvariables {")
    L.append("")
    L.append("        // -----------------------")
    L.append("        // dummy non fluent (richiesto solo perche' il blocco non-fluents non puo'")
    L.append("        // restare vuoto)")
    L.append("        // -----------------------")
    L.append("        DUMMY : { non-fluent, bool, default = false };")
    L.append("")
    L.append("        // -----------------------")
    L.append("        // stato MDP (one-hot): un booleano scalare per ciascuno stato scoperto")
    L.append("        // -----------------------")
    for k in range(n_states):
        default = 'true' if k == init_idx else 'false'
        L.append(f"        s{k} : {{ state-fluent, bool, default={default} }};")
    L.append("")
    L.append("        // -----------------------")
    L.append("        // attributi dello stato, TUTTI booleani nativi (fluents di prima classe,")
    L.append("        // usabili nella reward, nessun 'real'): gli attributi 'tri' (True/False/")
    L.append("        // missing, missing->false) sono un singolo booleano; TUTTE le colonne")
    L.append("        // categoriche - incluse quelle numeriche non discretizzate a monte, se")
    L.append("        // con pochi valori distinti - sono one-hot: un booleano PER CIASCUN")
    L.append("        // VALORE possibile, esattamente uno vero")
    L.append("        // -----------------------")
    if include_attrs:
        for bf in bool_fluents:
            L.append(f"        {bf['pvar']} : {{ state-fluent, bool, default=false }};")
    else:
        L.append("        // (attributi omessi in questa versione, solo control-flow)")
    L.append("")
    L.append("        // -----------------------")
    L.append("        // azioni: una booleana scalare per ciascuna attivita' osservata nel log")
    L.append("        // -----------------------")
    for name in action_names:
        L.append(f"        {action_fluent_of[name]} : {{ action-fluent, bool, default=false }};")
    L.append("    };")
    L.append("")

    # ---- cpfs ----
    L.append("    cpfs {")
    L.append("")
    L.append("        // ---- transizioni: catena di Bernoulli condizionati (stick-breaking) ----")
    L.append("        // quando un (stato,azione) ha piu' esiti osservati, ciascuno tranne")
    L.append("        // l'ultimo pesca con probabilita' condizionata al non essersi gia' verificato")
    L.append("        // uno degli esiti precedenti dello stesso gruppo (riferimento a s{T}')")
    L.append("        // Transizione diretta a singolo tick: niente variabili di staging.")
    for k in range(n_states):
        clauses = target_clauses.get(k, [])
        expr = balanced_chain(clauses, "false")
        L.append(f"        s{k}' = {expr};")
        L.append("")
    L.append("")
    if include_attrs and bool_fluents:
        L.append("        // ---- attributi booleani (tri singoli + cat one-hot, niente real) ----")
        L.append("        // CPF sul NUOVO stato one-hot s{k}': if (OR degli stati in cui vale")
        L.append("        // true) then true else false - state-fluent bool nativo, evita il")
        L.append("        // segfault osservato con gli attributi codificati come real.")
        for bf in bool_fluents:
            pvar = bf['pvar']
            values = bf['values']  # lista di bool, una per stato k
            true_states = [k for k, v in enumerate(values) if v]
            if not true_states:
                L.append(f"        {pvar}' = false;")
            elif len(true_states) == n_states:
                L.append(f"        {pvar}' = true;")
            else:
                cond = or_tree([f"s{k}'" for k in true_states])
                L.append(f"        {pvar}' = if ({cond}) then true else false;")
            L.append("")
    L.append("    };")
    L.append("")
    L.append("    reward =")
    if terminal_bonus is not None and terminal_idx:
        L.append("        // stessa forma if/then/else e stesso albero OR bilanciato gia' usati e")
        L.append("        // validati nel blocco termination, solo spostati nella reward - niente")
        L.append("        // negazioni ne' moltiplicazioni (a differenza del tentativo precedente")
        L.append("        // che aveva causato un crash)")
        or_terminal = or_tree([f"s{k}'" for k in sorted(terminal_idx)])
        L.append(f"        if ( {or_terminal} )")
        L.append(f"        then {terminal_bonus:.6g}")
        L.append("        else 1.0;")
    else:
        L.append("        1.0;")
        L.append("    // PLACEHOLDER: sostituire con la reward vera basata sugli attributi.")
    L.append("")
    L.append("")
    L.append("}")
    return "\n".join(L)


# --------------------------------------------------------------------------
# INSTANCE (solo dati: DUMMY non-fluent + init-state)
# --------------------------------------------------------------------------

def build_instance(domain_name, instance_name, bool_fluents, init_idx,
                    horizon, discount, include_attrs=True):
    L = []
    L.append(f"non-fluents {domain_name}_nf {{")
    L.append(f"    domain = {domain_name};")
    L.append("")
    L.append("    non-fluents {")
    L.append("        DUMMY = true;")
    L.append("    };")
    L.append("}")
    L.append("")
    L.append(f"instance {instance_name} {{")
    L.append(f"    domain = {domain_name};")
    L.append(f"    non-fluents = {domain_name}_nf;")
    L.append("")
    L.append("    init-state {")
    L.append(f"        s{init_idx} = true;")
    L.append("")
    if include_attrs:
        for bf in bool_fluents:
            val = bf['values'][init_idx]
            L.append(f"        {bf['pvar']} = {'true' if val else 'false'};")
    L.append("    };")
    L.append("")
    L.append("    max-nondef-actions = 1;")
    L.append(f"    horizon = {horizon};")
    L.append(f"    discount = {discount};")
    L.append("}")
    return "\n".join(L)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--states', required=True)
    ap.add_argument('--transitions', required=True)
    ap.add_argument('--outdir', default='.')
    ap.add_argument('--domain-name', default='mdp_process_domain')
    ap.add_argument('--instance-name', default='mdp_process_inst')
    ap.add_argument('--id-col', default='state')
    ap.add_argument('--action-col', default='action')
    ap.add_argument('--last-action-col', default='last_action_name')
    ap.add_argument('--init-state', type=int, default=None)
    ap.add_argument('--horizon', type=int, default=40)
    ap.add_argument('--discount', type=float, default=1.0)
    ap.add_argument('--include-attributes', action='store_true',
                     help="includi anche gli attributi (state-fluent bool per ciascuna colonna "
                          "del CSV stati, tri o one-hot). Di default sono ESCLUSI: la "
                          "configurazione senza attributi e' quella confermata funzionante con "
                          "PROST.")
    ap.add_argument('--terminal-bonus', type=float, default=10.0,
                     help="reward = terminal_bonus se si raggiunge uno stato terminale, "
                          "altrimenti 1.0.")
    ap.add_argument('--max-attributes', type=int, default=None,
                     help="se impostato, usa solo le prime N colonne di attributo del CSV "
                          "(utile per isolare via bisection se un sottoinsieme di attributi "
                          "causa segfault e altri no)")
    ap.add_argument('--numeric-cat-max-unique', type=int, default=20,
                     help="numero massimo di valori distinti ammessi per una colonna numerica "
                          "prima di trattarla come categorica one-hot; oltre questa soglia lo "
                          "script si ferma con un errore (probabile colonna continua non "
                          "discretizzata). Default: 20.")
    args = ap.parse_args()

    states_df = pd.read_csv(args.states).sort_values(args.id_col).reset_index(drop=True)
    trans_df = pd.read_csv(args.transitions)

    n_states = len(states_df)
    # indice di stato k = posizione nella tabella ordinata per id (0..n-1);
    # assumiamo che gli id in id_col siano gia' 0..n-1 (come nel dataset di test)
    state_ids = states_df[args.id_col].tolist()
    if state_ids != list(range(n_states)):
        raise ValueError("Questo script assume che gli id di stato siano 0..N-1 contigui; "
                          "rimappa gli id prima di generare (basta un dizionario id->indice).")

    skip_cols = {args.last_action_col}
    attrs = classify_columns(states_df, args.id_col, skip_cols,
                              numeric_cat_max_unique=args.numeric_cat_max_unique)
    if args.max_attributes is not None:
        attrs = attrs[:args.max_attributes]

    # valori "grezzi" di ciascun attributo per ciascuno stato (indice k = stato k):
    # bool per 'tri', valore categorico originale (stringa o numero) per 'cat'
    attr_values_by_state = {}
    for a in attrs:
        vals = []
        for _, row in states_df.iterrows():
            vals.append(attr_value_code(a, row))
        attr_values_by_state[a['pvar']] = vals

    # azioni: nome scalare sanitizzato per ciascuna attivita' osservata (serve
    # gia' qui per riservare i nomi prima di generare i nomi one-hot 'cat')
    action_names = sorted(trans_df[args.action_col].unique().tolist())
    raw = ["action-" + sanitize_pvar_name(a) for a in action_names]
    action_fluent_of = dict(zip(action_names, uniquify(raw)))

    reserved_names = (['DUMMY'] + [f"s{k}" for k in range(n_states)]
                       + list(action_fluent_of.values())
                       + [a['pvar'] for a in attrs if a['kind'] != 'cat'])
    cat_onehot_names = build_cat_onehot_names(states_df, attrs, reserved_names)

    # bool_fluents: un elemento per attributo 'tri' (booleano singolo) + N
    # elementi per ciascun attributo 'cat' (uno per valore, one-hot, incluse le
    # colonne numeriche non discretizzate a monte con pochi valori distinti) -
    # stessa struttura uniforme {'pvar':..., 'values':[bool per stato]}
    bool_fluents = []
    for a in attrs:
        if a['kind'] == 'tri':
            bool_fluents.append({'pvar': a['pvar'], 'values': attr_values_by_state[a['pvar']]})
        elif a['kind'] == 'cat':
            col = a['col']
            raw_vals = attr_values_by_state[a['pvar']]
            for cat_value, subpvar in cat_onehot_names[col].items():
                bool_fluents.append({
                    'pvar': subpvar,
                    'values': [rv == cat_value for rv in raw_vals]
                })

    # stati validi da cui ciascuna azione e' stata osservata (per action-preconditions)
    valid_states_of_action = defaultdict(set)
    for _, row in trans_df.iterrows():
        valid_states_of_action[row[args.action_col]].add(int(row[args.id_col]))

    # stato iniziale
    if args.init_state is None:
        starts = states_df.loc[states_df[args.last_action_col] == 'start', args.id_col].tolist() \
            if args.last_action_col in states_df.columns else []
        init_idx = int(starts[0]) if starts else 0
    else:
        init_idx = args.init_state

    terminal_idx = set(
        int(s) for s in states_df.loc[states_df[args.last_action_col] == 'archive_application', args.id_col]
    ) if args.last_action_col in states_df.columns else set()

    target_clauses, flat_clauses = build_target_clauses(trans_df, args.id_col, args.action_col,
                                                          action_fluent_of, n_states)

    domain_txt = build_domain(
        args.domain_name, bool_fluents, n_states, action_names,
        action_fluent_of, target_clauses, flat_clauses, terminal_idx, init_idx,
        valid_states_of_action,
        include_attrs=args.include_attributes,
        terminal_bonus=args.terminal_bonus
    )
    instance_txt = build_instance(
        args.domain_name, args.instance_name, bool_fluents, init_idx,
        args.horizon, args.discount,
        include_attrs=args.include_attributes
    )

    os.makedirs(args.outdir, exist_ok=True)
    dpath = os.path.join(args.outdir, 'domain.rddl')
    ipath = os.path.join(args.outdir, 'instance.rddl')
    with open(dpath, 'w') as f:
        f.write(domain_txt)
    with open(ipath, 'w') as f:
        f.write(instance_txt)

    print(f"Scritto {dpath}")
    print(f"Scritto {ipath}")
    n_cat_numeric = sum(1 for a in attrs if a['kind'] == 'cat'
                         and pd.api.types.is_numeric_dtype(states_df[a['col']]))
    print(f"Stati: {n_states}  Azioni: {len(action_names)}  Attributi: {len(attrs)}"
          f"  (di cui numerici trattati come categorici: {n_cat_numeric})")
    print(f"Init-state: s{init_idx}   Stati terminali: {sorted(terminal_idx)}")


if __name__ == '__main__':
    main()