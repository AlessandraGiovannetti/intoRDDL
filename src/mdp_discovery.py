"""
mdp_discovery.py
=================

Discovery-only version of the MDP construction pipeline.
Usa le classi/funzioni del submodule ProcessPilot (DatasetMDP, get_real_data,
define_real_state_cols, transition_probabilities_faster*, k_means) e definisce
solo MDPDiscovery, che orchestra la costruzione dell'MDP (astratto e non).
"""

import os
import sys


import random
from collections import defaultdict
import sys
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ProcessPilot"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ProcessPilot", "src"))


from dataset_manager.DatasetManager import *   
from utils import *                             
from MDP_functions import *                    

SEED = 0
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


# ---------------------------------------------------------------------------
# MDPDiscovery: builds the (unabstracted and abstracted) state space and the
# transition probabilities of the MDP from the full event log
# ---------------------------------------------------------------------------

class MDPDiscovery:
    """Discovers the MDP underlying an event log: encodes states, estimates
    transition probabilities, and (optionally) abstracts the state space."""

    def __init__(self, dataset, k=None, state_abstraction=False):
        self.dataset = dataset
        self.k = k
        self.state_abstraction = state_abstraction

        # ---- load & clean the log ----
        self.df, self.dataset_manager = get_real_data(dataset)
        self.config = DatasetMDP(dataset, self.df, self.dataset_manager)
        self.df = self.config.df
        self.all_cases = self.config.all_cases
        self.all_actions = self.config.all_actions
        self.activity_index = self.config.activity_index
        self.n_actions = self.config.n_actions
        self.costs_dic = self.config.costs_dic

        self.df = self.config.cut_at_terminal_states(self.df, self.dataset_manager)
        self.df_readable = self.df.copy()

        if self.dataset == "bpic2012_accepted":
            bins = [-float("inf"), 6000, 15000, float("inf")]
            self.df["AMOUNT_BUCKET"] = pd.cut(self.df["AMOUNT_REQ"], bins=bins,
                                               labels=["low", "medium", "high"], right=True)
        elif self.dataset == "traffic_fines_1":
            bins = [-float("inf"), 50, float("inf")]
            self.df["AMOUNT_BUCKET"] = pd.cut(self.df["amount"], bins=bins,
                                               labels=["low", "high"], right=False)

        self.state_cols_simulation, self.control_flow_var = define_real_state_cols(
            dataset, self.dataset_manager)

        # ---- scale numeric columns ----
        self.numeric_cols = self.dataset_manager.static_num_cols + self.dataset_manager.dynamic_num_cols
        self.scaler = StandardScaler()
        self.df[self.numeric_cols] = self.scaler.fit_transform(self.df[self.numeric_cols])

        # ---- encode categorical columns ----
        self.categorical_cols = self.dataset_manager.static_cat_cols + self.dataset_manager.dynamic_cat_cols
        self.onehot_cols = []
        self.onehot_encoders = {}
        self.embedding_cols = {}
        self.label_encoders = {}

        for col in self.categorical_cols:
            n_unique = self.df[col].nunique()
            if n_unique <= 100:
                ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.float32)
                encoded = ohe.fit_transform(self.df[[col]])
                ohe_cols = [f"{col}_{cat}" for cat in ohe.categories_[0]]
                df_ohe = pd.DataFrame(encoded, columns=ohe_cols, index=self.df.index)
                self.df = pd.concat([self.df, df_ohe], axis=1)
                if col in self.state_cols_simulation:
                    self.state_cols_simulation.remove(col)
                self.state_cols_simulation += ohe_cols
                self.onehot_encoders[col] = ohe
                self.onehot_cols.append(col)
            else:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le
                self.embedding_cols[col] = len(le.classes_)

        # ---- build the unabstracted state space ----
        self.all_states_unabs = self.df[self.state_cols_simulation].drop_duplicates().reset_index(drop=True)
        self.n_states_unabs = len(self.all_states_unabs)
        self.all_state_unabs_index = {tuple(row): idx for idx, row in self.all_states_unabs.iterrows()}
        print(f"Discovered {self.n_states_unabs} original states from the event log "
              f"({self.n_states_unabs / len(self.df):.4f} states/event)")

        # ---- estimate transition probabilities on the unabstracted MDP ----
        self.transition_proba = transition_probabilities_faster(
            self.df, self.state_cols_simulation, self.all_states_unabs,
            self.activity_index, self.n_actions)
        self.all_states_unabs = self.all_states_unabs.drop(columns=["state_index"])

        # ---- abstract the state space (optional) ----
        print("Abstracting the state space")
        self.abstract_state_space()

        self.initial_states = self.get_initial_states()
        print(f"Initial states of the resulting MDP: {self.initial_states}")

        print("Building abstracted transition probabilities / action mask")
        self.build_abs_transition_proba()

    # -- state abstraction -------------------------------------------------
    def abstract_state_space(self):
        if self.state_abstraction in ["full_k_means", "partial_k_means"]:
            self.clustering_state_cols = (
                [c for c in self.state_cols_simulation if c not in (["last_action"] + self.control_flow_var)]
                if self.state_abstraction == "partial_k_means"
                else self.state_cols_simulation
            )
            self.df, self.kmeanModel = k_means(self.df, self.clustering_state_cols, self.k)
            self.state_cols = (
                ["last_action", "cluster"] + self.control_flow_var
                if self.state_abstraction == "partial_k_means"
                else ["cluster"]
            )

        elif self.state_abstraction == "branchi":
            if self.dataset in ["bpic2012_accepted", "traffic_fines_1"]:
                self.state_cols = ["last_action", "AMOUNT_BUCKET"] + self.control_flow_var

        elif self.state_abstraction is False:
            self.state_cols = self.state_cols_simulation

        elif self.state_abstraction == "contextual":
            self.state_cols = ["last_action"]

        elif self.state_abstraction == "structural":
            df_structural = self.df.copy()
            df_structural["state"] = df_structural[self.state_cols_simulation].apply(
                lambda row: self.all_state_unabs_index.get(tuple(row), -1), axis=1)
            df_structural["next_state"] = df_structural.groupby("ID")["state"].shift(-1).fillna(-1).astype(int)

            state_action_map = defaultdict(set)
            for _, row in df_structural.iterrows():
                state_action_map[row["state"]].add(f"{row['action']};{row['next_state']}")

            unique_state_groups = {}
            merged_state_index = 0
            state_to_cluster = {}
            for state, action_next_pairs in state_action_map.items():
                action_next_pairs = tuple(sorted(action_next_pairs))
                if action_next_pairs not in unique_state_groups:
                    unique_state_groups[action_next_pairs] = merged_state_index
                    merged_state_index += 1
                state_to_cluster[state] = unique_state_groups[action_next_pairs]

            df_structural["cluster"] = df_structural["state"].map(state_to_cluster)
            self.df = df_structural.copy()
            self.state_cols = ["cluster"]

        elif self.state_abstraction == "action-set":
            state_to_cluster = {}
            unique_sets = {}
            merged_state_index = 0
            action_sets = []

            for idx, row in self.df.iterrows():
                state = self.all_state_unabs_index[tuple(row[self.state_cols_simulation])]
                next_actions = [
                    a for a in range(self.n_actions)
                    if (state, a) in self.transition_proba and self.transition_proba[(state, a)].sum() > 0
                ]
                action_sets.append(next_actions)
                key = tuple(sorted(next_actions))
                if key not in unique_sets:
                    unique_sets[key] = merged_state_index
                    merged_state_index += 1
                state_to_cluster[state] = unique_sets[key]

            self.df["action_set"] = action_sets
            self.df["cluster"] = self.df[self.state_cols_simulation].apply(
                lambda row: state_to_cluster[self.all_state_unabs_index[tuple(row)]], axis=1)
            self.state_cols = ["cluster"]

        elif self.state_abstraction == "action-count":
            state_to_cluster = {}
            unique_counts = {}
            merged_state_index = 0
            action_counts_list = []

            for idx, row in self.df.iterrows():
                state = self.all_state_unabs_index[tuple(row[self.state_cols_simulation])]
                counts = np.zeros(self.n_actions, dtype=int)
                for (s, a), probs in self.transition_proba.items():
                    if s == state and probs.sum() > 0:
                        counts[a] += 1
                action_counts_list.append(counts)
                key = tuple(counts)
                if key not in unique_counts:
                    unique_counts[key] = merged_state_index
                    merged_state_index += 1
                state_to_cluster[state] = unique_counts[key]

            self.df["action_count"] = [list(c) for c in action_counts_list]
            self.df["cluster"] = self.df[self.state_cols_simulation].apply(
                lambda row: state_to_cluster[self.all_state_unabs_index[tuple(row)]], axis=1)
            self.state_cols = ["cluster"]

        # build the abstracted state space
        self.all_states = self.df[self.state_cols].copy()
        if isinstance(self.all_states, pd.Series):
            self.all_states = self.all_states.to_frame()
        self.all_states = self.all_states.drop_duplicates().reset_index(drop=True)
        self.n_states = len(self.all_states)
        self.all_state_index = {tuple(row): idx for idx, row in self.all_states.iterrows()}

        self.unabs_to_abs_state = {
            tuple(row[self.state_cols_simulation])
            if isinstance(row[self.state_cols_simulation], (list, tuple, np.ndarray, pd.Series))
            else (row[self.state_cols_simulation],):
            self.all_state_index.get(
                tuple(row[self.state_cols])
                if isinstance(row[self.state_cols], (list, tuple, np.ndarray, pd.Series))
                else (row[self.state_cols],),
                None,
            )
            for _, row in self.df.iterrows()
        }
        print(f"Discovered {self.n_states} abstracted states (k={self.k}) "
              f"({self.n_states / len(self.df):.4f} states/event)")

    # -- transition probabilities on the abstracted MDP ---------------------
    def build_abs_transition_proba(self):
        self.abs_transition_proba = transition_probabilities_faster(
            self.df, self.state_cols, self.all_states, self.activity_index, self.n_actions)
        self.all_states = self.all_states.drop(columns=["state_index"])

        self.valid_actions_abs = {
            abs_state: [
                a for a in range(self.n_actions)
                if (abs_state, a) in self.abs_transition_proba
                and self.abs_transition_proba[(abs_state, a)].sum() > 0
            ]
            for abs_state in range(self.n_states)
        }

    # -- simple diagnostics ---------------------------------------------------
    def measure_simplicity(self):
        n_nodes = self.n_states
        abs_transitions_model = set()
        for (abs_state, action), probs in self.abs_transition_proba.items():
            prob_vector = probs.toarray().flatten()
            if prob_vector.sum() <= 0:
                continue
            next_abs_indices = np.where(prob_vector > 0)[0]
            for next_abs in next_abs_indices:
                abs_transitions_model.add((int(abs_state), int(action), int(next_abs)))
        n_edges = len(abs_transitions_model)
        print(f"Simplicity: {n_nodes} nodes, {n_edges} edges")
        return n_nodes, n_edges, abs_transitions_model

    def get_initial_states(self):
        initial_states = set()
        for case_id, group in self.df.groupby("ID"):
            group = group.sort_values(by=self.dataset_manager.timestamp_col)
            first_row = group.iloc[0]
            state_tuple = tuple(first_row[col] for col in self.state_cols)
            if state_tuple in self.all_state_index:
                initial_states.add(self.all_state_index[state_tuple])
        return sorted(initial_states)

    def describe_states(self):
        df_desc = self.df.copy()
        df_desc["state"] = df_desc[self.state_cols].apply(
            lambda row: self.all_state_index[tuple(row)], axis=1
        )

        unscaled = df_desc.copy()
        unscaled[self.numeric_cols] = self.scaler.inverse_transform(df_desc[self.numeric_cols])

        idx_to_action = {v: k for k, v in self.activity_index.items()}
        unscaled["last_action_name"] = unscaled["last_action"].map(idx_to_action)

        for col in self.onehot_cols:
            ohe_cols = [c for c in self.state_cols_simulation if c.startswith(f"{col}_")]
            unscaled[col] = (
                unscaled[ohe_cols].idxmax(axis=1).str.replace(f"{col}_", "", regex=False)
            )

        describe_cols = (
            ["last_action_name"]
            + self.onehot_cols
            + list(self.embedding_cols.keys())
            + self.numeric_cols
        )

        summary = (
            unscaled.groupby("state")[describe_cols]
            .agg(lambda s: s.mode().iloc[0] if s.dtype == object else s.mean())
            .reset_index()
        )
        return summary


if __name__ == "__main__":
    mdp = MDPDiscovery(dataset="sepsis_cases_1", k=10, state_abstraction="partial_k_means")
    mdp.measure_simplicity()

    rows = []
    for (state, action), probs in mdp.abs_transition_proba.items():
        probs = probs.toarray().flatten()
        for next_state, p in enumerate(probs):
            if p > 0:
                rows.append({
                    "state": state,
                    "action_index": action,
                    "action": mdp.all_actions[action],
                    "next_state": next_state,
                    "probability": float(p),
                })

    transitions = pd.DataFrame(rows)
    transitions.to_csv("mdp_transitions_test.csv", index=False)
    print("Salvato in mdp_transitions_test.csv")

    states = mdp.all_states.copy()
    states["state"] = states.index
    states["initial"] = states["state"].isin(mdp.initial_states)
    states.to_csv("mdp_states_test.csv", index=False)
    print("Salvato in mdp_states_test.csv")

    state_descriptions = mdp.describe_states()
    state_descriptions.to_csv("mdp_states_described_test.csv", index=False)
    print("Salvato in mdp_states_described_test.csv")