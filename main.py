#!/usr/bin/env python3
"""
main.py
=======
Complete pipeline for Prescriptive Process Monitoring:

    1. src/mdp_discovery.py -> mdp_transitions.csv + mdp_states_described.csv
    2. src/discretize.py    -> mdp_states_discretized.csv
    3. src/encoding_init.py -> domain + instance RDDL
    4. PROST  -> policy

Usage:
    python main.py --dataset sepsis_preprocessed
    python main.py --dataset rtf_preprocessed --k 20 --quantiles 4

Output (output/<dataset>/):
    mdp/       MDP transitions and states
    encoding/  RDDL domain and instance
    prost/     PROST results
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATASETS = [
    "sepsis_preprocessed",
    "bpi12_preprocessed",
    "rtf_preprocessed",
    "permit_preprocessed",
    "intDecl_preprocessed",
]

# File prodotti nelle varie fasi
TRANSITIONS_FILE = "mdp_transitions.csv"          # discovery
DESCRIBED_FILE = "mdp_states_described.csv"       # discovery
DISCRETIZED_FILE = "mdp_states_discretized.csv"   # discretizzazione

# Cartelle definite dal Dockerfile di PROST
RDDL_DIR = Path(os.environ.get("RDDL", "/RDDL"))
PROST_OUT_DIR = Path(os.environ.get("PROST_OUT", "/OUTPUTS"))
PROST_SH = "/workspace/pyRDDLGym/prost.sh"


def short_name(dataset):
    """sepsis_preprocessed -> sepsis (rtf, sepsis, bpi12, permit, intDecl)."""
    return dataset.replace("_preprocessed", "")


def run(cmd, cwd=ROOT):
    """Lancia un comando mostrando l'output in tempo reale; esce se fallisce."""
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    result = subprocess.run([str(c) for c in cmd], cwd=cwd)
    if result.returncode != 0:
        sys.exit(f"[ERRORE] comando fallito (exit code {result.returncode})")


def step_discovery(args, mdp_dir):
    print("\n=== 1/4  MDP discovery ===")
    run([
        sys.executable, "src/mdp_discovery.py",
        "--dataset", args.dataset,
        "--k", args.k,
        "--state-abstraction", args.state_abstraction,
        "--outdir", mdp_dir,
    ])


def step_discretize(args, mdp_dir):
    print("\n=== 2/4  Discretizzazione ===")
    # 'state' viene sempre escluso; eventuali altre colonne si aggiungono con --exclude-cols
    exclude = ["state"] + args.exclude_cols
    run([
        sys.executable, "src/discretize.py",
        mdp_dir / DESCRIBED_FILE,
        mdp_dir / DISCRETIZED_FILE,
        "--quantiles", args.quantiles,
        "--exclude", *exclude,
        "--dataset", short_name(args.dataset),
    ])


def step_encoding(mdp_dir, enc_dir):
    print("\n=== 3/4  Encoding RDDL ===")
    run([
        sys.executable, "src/encoding_init.py",
        "--states", mdp_dir / DISCRETIZED_FILE,
        "--transitions", mdp_dir / TRANSITIONS_FILE,
        "--include-attributes",
        "--outdir", enc_dir,
    ])


def step_prost(args, enc_dir, prost_res_dir):
    print("\n=== 4/4  PROST ===")

    rddl_files = sorted(enc_dir.glob("*.rddl"))
    if not rddl_files:
        sys.exit(f"[ERRORE] nessun file .rddl trovato in {enc_dir}")

    # Svuota /RDDL e /OUTPUTS (equivalente del 'docker rm -f' + nuovo run)
    for d in (RDDL_DIR, PROST_OUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
        for item in d.iterdir():
            shutil.rmtree(item) if item.is_dir() else item.unlink()

    # Equivalente del bind mount su /RDDL
    for f in rddl_files:
        shutil.copy(f, RDDL_DIR / f.name)
        print(f"  copiato {f.name} -> {RDDL_DIR}")

    # Equivalente di: docker run ... prost 1 "[PROST -s 1 -se [IPC2014]]"
    run(["/bin/bash", PROST_SH, args.prost_instances, args.prost_config],
        cwd="/workspace/pyRDDLGym")

    # Equivalente di: docker cp prost_final:/OUTPUTS/ ...
    prost_res_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(PROST_OUT_DIR, prost_res_dir, dirs_exist_ok=True)
    print(f"  risultati PROST copiati in {prost_res_dir}")


def main():
    p = argparse.ArgumentParser(
        description="MDP discovery -> discretizzazione -> RDDL encoding -> PROST")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--k", type=int, default=10, help="numero di cluster (k-means)")
    p.add_argument("--state-abstraction", default="partial_k_means",
                   help="tipo di astrazione (default: partial_k_means)")
    p.add_argument("--quantiles", type=int, default=3,
                   help="numero di quantili per la discretizzazione (default: 3)")
    p.add_argument("--exclude-cols", nargs="*", default=[],
                   help="colonne extra da escludere dalla discretizzazione "
                        "('state' è sempre escluso)")
    p.add_argument("--outdir", default=str(ROOT / "output"),
                   help="cartella base degli output (default: ./output)")
    p.add_argument("--prost-instances", default="1",
                   help="primo argomento di prost.sh (default: 1)")
    p.add_argument("--prost-config", default="[PROST -s 1 -se [IPC2014]]",
                   help="secondo argomento di prost.sh")
    p.add_argument("--skip-discovery", action="store_true",
                   help="salta la discovery e riusa i csv già presenti")
    p.add_argument("--skip-discretize", action="store_true",
                   help="salta la discretizzazione e riusa il csv già presente")
    p.add_argument("--skip-encoding", action="store_true",
                   help="salta l'encoding e riusa i .rddl già presenti")
    args = p.parse_args()
    args.k = str(args.k)
    args.quantiles = str(args.quantiles)

    base = Path(args.outdir) / short_name(args.dataset)
    mdp_dir, enc_dir, prost_res_dir = base / "mdp", base / "encoding", base / "prost"
    for d in (mdp_dir, enc_dir):
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    if not args.skip_discovery:
        step_discovery(args, mdp_dir)
    if not args.skip_discretize:
        step_discretize(args, mdp_dir)
    if not args.skip_encoding:
        step_encoding(mdp_dir, enc_dir)
    step_prost(args, enc_dir, prost_res_dir)

    print(f"\nFatto in {(time.perf_counter() - t0) / 60:.2f} minuti. Output in {base}")


if __name__ == "__main__":
    main()