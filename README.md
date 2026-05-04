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
    output_dir="./output"
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

## Package Contents

- **ClusterRegressionAnalyzer**: Main class for cluster regression analysis
- **ClusterRegressionPlotter**: Visualization utilities for analysis results

## Dependencies

- numpy
- PyYAML
- matplotlib
- hist
- pflow_analysis
