# PFlow Tools

Tools for PFlow analysis and visualization.

## Installation

Install the package from the source directory:

```bash
cd /path/to/pflow-tools
pip install -e .
```

This installs pflow-tools in editable mode, allowing you to use it as a package while making development changes.

## Usage

Once installed, you can import and use the tools:

```python
from pflow_tools import ClusterRegressionAnalyzer

analyzer = ClusterRegressionAnalyzer(
    model_predictions_path="/path/to/predictions.h5",
    config_path="/path/to/config.yaml",
    sample_name="Di-jets",
    output_dir="./output",
    # Optional: infer residual particle type when only N-1 are regressed
    residual_particle_type="MUONS"
)

analyzer.run_all()
```

Or use the command-line script:

```bash
python cluster_regression.py \
    --model-predictions /path/to/predictions.h5 \
    --config /path/to/config.yaml \
    --sample-name "Di-jets" \
    --output-dir /path/to/output
```

If only N-1 particle types are regressed, you can pass a residual type or
let the analyzer infer it when exactly one truth particle type is missing
from config targets:

```bash
python cluster_regression.py \
    --model-predictions /path/to/predictions.h5 \
    --config /path/to/config.yaml \
    --sample-name "Di-jets" \
    --output-dir /path/to/output \
    --residual-particle-type MUONS
```

## Scripts

Run these after `source scripts/check_env.sh`.

### Training

```bash
scripts/run_training.sh -c /path/to/config.yaml
```

Optional base config override:

```bash
scripts/run_training.sh -c /path/to/config.yaml -b /path/to/base.yaml
```

### Testing

```bash
scripts/run_testing.sh -c /path/to/config.yaml -t /path/to/test_file.h5 -d 1
```

### Dataset preparation

```bash
scripts/prepare_datasets.sh -c /path/to/config.yaml
```

## Package Contents

- **ClusterRegressionAnalyzer**: Main class for cluster regression analysis
- **ClusterRegressionPlotter**: Visualization utilities for analysis results

## Dependencies

- numpy
- PyYAML
- matplotlib
- hist
- pflow_analysis
