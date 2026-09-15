# PFlow Tools

Utilities for analysis/plotting of the clusters' energy fractions regression results and HDF5 dataset preparation for SALT.

## What this repository provides

- Python package `pflow-tools` (import as `pflow_tools`)
- Analysis classes:
    - `ClusterRegressionAnalyzer`
    - `ClusterRegressionPlotter`
- CLI tool from package entry points:
    - `h5-prep`
- Local shell scripts for SALT training/testing
- SLURM wrapper scripts for SALT training/testing

## Installation

From repository root:

```bash
pip install -e .
```

If you prefer requirements-based install:

```bash
pip install -r requirements.txt
```

## Declared dependencies

Package metadata in [pyproject.toml](pyproject.toml) declares:

- `numpy>=1.24.0,<3.0.0`
- `awkward>=2.0.0`
- `PyYAML>=6.0`
- `matplotlib>=3.5.0`
- `hist>=2.6.0`

The shell training/testing scripts call `salt`, so SALT must be installed in the
active environment used for those scripts.

## Python API usage

Use `ClusterRegressionAnalyzer` from [tools/cluster_regression_analyzer.py](tools/cluster_regression_analyzer.py):

```python
from pflow_tools import ClusterRegressionAnalyzer

analyzer = ClusterRegressionAnalyzer(
        model_predictions_path="/path/to/predictions.h5",
        config_path="/path/to/config.yaml",
        sample_name="Di-jets",
        output_dir="./output",
        final_activation=True           # Optional, use if final layer stores raw logits
        residual_particle_type="MUONS", # Optional, can be used to infer one particle's energy fraction
)

analyzer.run_all()
```

Notes:

- Prediction prefix is auto-detected from available HDF5 fields.
- If one truth particle type is missing from config targets, residual inference
    can be auto-applied (`auto_residual=True` by default).

## CLI: h5-prep

The command `h5-prep` is exposed by [pyproject.toml](pyproject.toml#L29) and implemented in [tools/utils/h5_prep.py](tools/utils/h5_prep.py).

### Config-driven pipeline

```bash
h5-prep -c /path/to/pipeline.yaml
```

Example config files are available in [tools/utils/configs](tools/utils/configs).

### Subcommands

Remove columns:

```bash
h5-prep remove-cols /path/in.h5 /path/out.h5 -d clusters
```

Split datasets:

```bash
h5-prep split /path/in.h5 /path/train.h5 /path/val.h5 \
    -t /path/test.h5 --fraction_train 0.7 --fraction_val 0.2 --fraction_test 0.1 --shuffle
```

Create standardization/Z-score normalization dictionary:

```bash
h5-prep std-dict /path/train.h5 -o /path/std_dict.yaml
```

Filter events with unphysical energy fractions:

```bash
h5-prep filter-events /path/in.h5 /path/filtered.h5 --discard-unset
```

## Local SALT scripts

### 1) [Optional] Check and activate conda environment

Use [scripts/utils/check_env.sh](scripts/utils/check_env.sh):

```bash
source scripts/utils/check_env.sh
```

Or with explicit environment name:

```bash
source scripts/utils/check_env.sh my_salt_env
```

If an envoronment is already set up, this step can be skipped.

### 2) Local training

Use [scripts/run_training.sh](scripts/run_training.sh):

```bash
scripts/run_training.sh -c /path/to/config.yaml
```

Multiple config files are supported after a single `-c`:

```bash
scripts/run_training.sh -c /path/to/config1.yaml /path/to/config2.yaml
```

This script runs:

```bash
salt fit -c <config...> --force
```

### 3) Local testing

Use [scripts/run_testing.sh](scripts/run_testing.sh):

```bash
scripts/run_testing.sh -c /path/to/config.yaml -t /path/to/test_file.h5 -d 1
```

Arguments:

- `-c, --config` (required): model config path
- `-t, --test-file` (optional): test HDF5 path; if omitted, inferred from config key `data.test_file`
- `-d, --devices` (optional): trainer devices, default `0`

This script runs:

```bash
salt test -c <config> --data.test_file <test_file> --trainer.devices=<devices>
```

### 4) Dataset preparation wrapper

Use [scripts/utils/prepare_datasets.sh](scripts/utils/prepare_datasets.sh):

```bash
scripts/utils/prepare_datasets.sh -c /path/to/config.yaml
```

This forwards to:

```bash
h5-prep -c /path/to/config.yaml
```

## SLURM workflows

Use wrapper scripts in [scripts/slurm](scripts/slurm), which validate inputs,
export required environment variables, and submit `sbatch` jobs.

### Training submission

Wrapper: [scripts/slurm/wrapper_run_training.sh](scripts/slurm/wrapper_run_training.sh)

```bash
scripts/slurm/wrapper_run_training.sh -c /path/to/config.yaml
```

Multiple configs:

```bash
scripts/slurm/wrapper_run_training.sh -c /path/c1.yaml /path/c2.yaml -e salt
```

Defaults and behavior:

- Conda environment defaults to `salt` (`-e, --env` to override)
- Submits [scripts/slurm/run_training.slurm](scripts/slurm/run_training.slurm)
- SLURM job sets `COMET_OFFLINE_DIRECTORY` per job ID and executes `salt fit`

### Testing submission

Wrapper: [scripts/slurm/wrapper_run_testing.sh](scripts/slurm/wrapper_run_testing.sh)

```bash
scripts/slurm/wrapper_run_testing.sh -c /path/to/config.yaml -t /path/to/test.h5 -d 1 -e salt
```

Defaults and behavior:

- `-t, --test-file` optional; inferred from config `data.test_file` if omitted
- Devices default to `1` in wrapper (`-d, --devices`)
- Conda environment defaults to `salt` (`-e, --env`)
- Submits [scripts/slurm/run_testing.slurm](scripts/slurm/run_testing.slurm)

### Useful SLURM commands

Submit a job manually:

```bash
sbatch scripts/slurm/run_training.slurm
sbatch scripts/slurm/run_testing.slurm
```

Show queued/running jobs:

```bash
squeue -u "$USER"
```

Show detailed information for one job:

```bash
scontrol show job <job_id>
```

Follow live logs (paths used by SLURM scripts in this repository):

```bash
tail -f /project_root/logs/salt_<job_id>.out
tail -f /project_root/logs/salt_<job_id>.err
```

Note: This assumes SALT's repository is located at `project_root/salt-mpp-pflow/`

Show completed job accounting summary:

```bash
sacct -j <job_id> --format=JobID,JobName,State,Elapsed,MaxRSS,NodeList
```

Cancel a job:

```bash
scancel <job_id>
```

### Typical workflow

1. Submit a training job with the desired config(s):

```bash
scripts/slurm/wrapper_run_training.sh -c /path/to/config.yaml -e salt
```

2. Track the job and inspect logs:

```bash
squeue -u "$USER"
tail -f /ptmp/viktorg/SALTImplementation/logs/salt_<train_job_id>.err
```

During training, the logs usually include a Comet link. Open that link to monitor
metrics online and confirm the run quality.

3. After training finishes successfully, submit testing/evaluation with the
correct config and test-file arguments:

```bash
scripts/slurm/wrapper_run_testing.sh \
    -c /path/to/config.yaml \
    -t /path/to/test_file.h5 \
    -d 1 \
    -e salt
```

4. Monitor the test job until it succeeds:

```bash
tail -f /ptmp/viktorg/SALTImplementation/logs/salt_<test_job_id>.err
```

5. Retrieve the trained checkpoint artifact from the training run directory,
then use it for downstream analysis.

Typical checkpoint location:

```bash
/ptmp/viktorg/SALTImplementation/comet_logs/salt_job_<train_job_id>/ckpt/
```

The training SLURM script sets COMET_OFFLINE_DIRECTORY automatically per job,
so the run-specific checkpoint directory follows the same pattern under
comet_logs.

## Repository layout

- [tools](tools): Python package source (`pflow_tools`)
- [tools/utils/h5_prep.py](tools/utils/h5_prep.py): HDF5 preparation CLI
- [scripts](scripts): local training/testing shell scripts
- [scripts/utils](scripts/utils): environment checks and dataset prep wrapper
- [scripts/slurm](scripts/slurm): SLURM job scripts and submission wrappers
