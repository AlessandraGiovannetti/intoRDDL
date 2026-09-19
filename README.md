# intoRDDL

This repository provides an **integrated tool** for:

- Discovering an **MDP** from event logs by using [ProcessPilot](https://github.com/vsonnemans15/Process_Pilot)
- Generating **RDDL domains and instances** from the discovered MDP
- Solving them with the [PROST](https://github.com/prost-planner/prost) planner to generate prescriptive interventions

Everything runs inside a single Docker container, so PROST does not have to be launched from a separate container.

## Pipeline

1. **MDP discovery**: states, actions and transition probabilities are discovered from the event log, with optional state abstraction.
2. **Discretization**: the described states are discretized (quantile-based).
3. **RDDL encoding**: the discretized states and the transitions are encoded into an RDDL domain and instance.
4. **Planning**: PROST is run on the generated domain and instance.

## Repository structure

```
intoRDDL/
├── main.py               # pipeline entry point (discovery -> discretization -> encoding -> PROST)
├── Dockerfile            # project image, extends the PROST base image
├── requirements.txt
├── .dockerignore
└── src/
    ├── mdp_discovery.py  # MDP discovery
    ├── discretize.py     # state discretization
    ├── encoding_init.py  # MDP -> RDDL encoding
    ├── Preprocessing/    # preprocessing of the event logs
    ├── input/            # preprocessed event logs: <dataset>.csv
    └── ProcessPilot/     # git submodule
```

## Supported datasets

| Dataset name            | Input file                            |
|-------------------------|---------------------------------------|
| `sepsis_preprocessed`   | `src/input/sepsis_preprocessed.csv`   |
| `bpi12_preprocessed`    | `src/input/bpi12_preprocessed.csv`    |
| `rtf_preprocessed`      | `src/input/rtf_preprocessed.csv`      |
| `permit_preprocessed`   | `src/input/permit_preprocessed.csv`   |
| `intDecl_preprocessed`  | `src/input/intDecl_preprocessed.csv`  |

A dataset must be preprocessed following the procedure in `src/Preprocessing`. The datasets listed above, together with their preprocessing, are already available in the repository.

## Prerequisites

- [Docker](https://www.docker.com/) (Docker Desktop on Windows/macOS)
- Git, to fetch the ProcessPilot submodule

## Setup

### 1. Clone the repository with its submodule

```bash
git clone https://github.com/AlessandraGiovannetti/intoRDDL.git
cd intoRDDL
git submodule update --init
```

### 2. Build the PROST base image

The PROST image (Ubuntu 22.04, Z3, pyRDDLGym, PROST) is built once from its own Dockerfile. The build context must also contain `rddlsim.py` and `prost.sh`, which the Dockerfile copies in.

```bash
docker build -t prost-base -f Dockerfile.prost .
```

### 3. Build the project image

```bash
docker build -t intordd .
```

The project image extends `prost-base`, installs the Python dependencies and copies the code. It overrides PROST's default entrypoint with `python main.py`.

## Usage

Run the whole pipeline on one dataset, mounting a local `output` folder to retrieve the results.

**PowerShell**

```powershell
docker run --rm -it -v "${PWD}/output:/app/output" intordd --dataset <dataset>
```

**Bash**

```bash
docker run --rm -it -v "$(pwd)/output:/app/output" intordd --dataset <dataset>
```

The `-it` flags are needed because the encoding step asks which initial state to start from. To skip the question (and drop `-it`), pass the state directly with `--init-state`:

```bash
docker run --rm -v "$(pwd)/output:/app/output" intordd --dataset <dataset> --init-state <state>
```

You can also run only some of the steps, instead of the entire pipeline, by adding the `--skip-discovery`, `--skip-discretize` and `--skip-encoding` options (see below). Skipped steps reuse the files already present in the output folder.

### Options

| Option                  | Default                        | Description                                                        |
|-------------------------|--------------------------------|--------------------------------------------------------------------|
| `--dataset`             | (required)                     | One of the supported datasets listed above                         |
| `--k`                   | `10`                           | Number of clusters used for state abstraction                      |
| `--state-abstraction`   | `partial_k_means`              | State abstraction strategy                                         |
| `--quantiles`           | `3`                            | Number of quantiles used to discretize the states                  |
| `--exclude-cols`        | (none)                         | Extra columns to exclude from discretization (`state` is always excluded) |
| `--init-state`          | (interactive prompt)           | Initial state used by the encoding. If omitted, the encoding asks for it: run the container with `-it` |
| `--outdir`              | `./output`                     | Base output folder                                                 |
| `--prost-instances`     | `1`                            | First argument passed to `prost.sh`                                |
| `--prost-config`        | `[PROST -s 1 -se [IPC2014]]`   | Second argument passed to `prost.sh` (PROST configuration string)  |
| `--skip-discovery`      | off                            | Reuse the CSV files already in the output folder                   |
| `--skip-discretize`     | off                            | Reuse the discretized states CSV already in the output folder      |
| `--skip-encoding`       | off                            | Reuse the RDDL files already in the output folder                  |

Example: re-run only PROST with a different configuration on an already generated MDP.

```powershell
docker run --rm -v "${PWD}/output:/app/output" intordd `
  --dataset sepsis_preprocessed --skip-discovery --skip-discretize --skip-encoding `
  --prost-config "[PROST -s 1 -se [IPC2014]]"
```

## Output

For each run, results are written under `output/<dataset>/` (dataset name without the `_preprocessed` suffix):

```
output/sepsis/
├── mdp/        # mdp_transitions.csv, mdp_states_described.csv, mdp_states_discretized.csv
├── encoding/   # RDDL domain and instance
└── prost/      # PROST results (copy of the container's /OUTPUTS)
```

## Running the steps manually

Each step can also be run on its own, without Docker, provided the dependencies are installed.

```bash
python src/mdp_discovery.py --dataset sepsis_preprocessed --k 10 \
    --state-abstraction partial_k_means --outdir output/sepsis/mdp

python src/discretize.py \
    output/sepsis/mdp/mdp_states_described.csv \
    output/sepsis/mdp/mdp_states_discretized.csv \
    --quantiles 3 --exclude state --dataset sepsis

python src/encoding_init.py \
    --states output/sepsis/mdp/mdp_states_discretized.csv \
    --transitions output/sepsis/mdp/mdp_transitions.csv \
    --include-attributes --outdir output/sepsis/encoding
```

In `discretize.py` the dataset is given by its short name (`rtf`, `sepsis`, `bpi12`, `permit`, `intDecl`).

PROST itself is only available inside the container.