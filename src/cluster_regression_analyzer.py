"""
Cluster Regression Analysis Module

Provides the ClusterRegressionAnalyzer class for analysis of cluster regression
SALT model outputs including scatter plots, 2D histograms, and confusion
matrices for different truth energy fraction ranges.
"""

from pathlib import Path
from typing import List, Tuple
import yaml

import numpy as np
import hist

from pflow_analysis import h5 as h5_module
from .cluster_regression_plotter import ClusterRegressionPlotter


class ClusterRegressionAnalyzer:
    """Analyze cluster regression model predictions.

    Provides analysis of cluster regression model outputs including
    scatter plots, 2D histograms, and confusion matrices for different truth
    energy fraction ranges. Uses ClusterRegressionPlotter for visualization.
    """

    def __init__(
        self,
        model_predictions_path: str,
        config_path: str,
        sample_name: str,
        output_dir: str = "./output",
        target_prefix: str = "clusterParticle_EnergyFraction_Full_",
    ):
        """Initialize the analyzer.

        Parameters
        ----------
        model_predictions_path : str
            Path to the HDF5 file with model predictions
        config_path : str
            Path to the YAML config file used for training
        sample_name : str
            Human-readable name for the sample (e.g., "Di-jets", "Drell-Yan")
        output_dir : str, optional
            Output directory for plots, by default "./output"
        target_prefix : str, optional
            Prefix to remove from target names to get particle types,
            by default "clusterParticle_EnergyFraction_Full_"

        Raises
        ------
        FileNotFoundError
            If model prediction file or config file does not exist
        ValueError
            If config structure is not recognized
        """
        self.model_predictions_path = Path(model_predictions_path)
        self.config_path = Path(config_path)
        self.sample_name = sample_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_prefix = target_prefix

        # Load configuration
        self.config = self._load_config()

        # Load dataset
        self.dataset = h5_module.load_hdf(str(self.model_predictions_path))

        # Extract particle types and task name from config
        self.particle_types, self.task_name = self._extract_particle_types()

        # Auto-detect the prediction prefix by looking at available fields
        self.pred_prefix = self._detect_prediction_prefix()

        # Initialize plotter
        self.plotter = ClusterRegressionPlotter(
            str(self.output_dir), self.sample_name, self.particle_types
        )

        print(f"DEBUG: Detected prediction prefix: {self.pred_prefix}")

    def _load_config(self) -> dict:
        """Load YAML configuration file.

        Returns
        -------
        dict
            Parsed YAML configuration dictionary
        """
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def _extract_particle_types(self) -> Tuple[List[str], str]:
        """Extract particle types and task name from config targets.

        Returns
        -------
        tuple
            (list of particle type names, task name)

        Raises
        ------
        ValueError
            If config structure is not recognized
        """
        if "model" not in self.config or "tasks" not in self.config["model"]["model"]["init_args"]:
            raise ValueError("Config structure not recognized")

        # Get targets from the first task (assuming cluster regression)
        modules = self.config["model"]["model"]["init_args"]["tasks"]["init_args"]["modules"]
        task_name = modules[0]["init_args"]["name"]
        targets = modules[0]["init_args"]["targets"]

        # Extract particle type names from target column names by removing prefix
        # Format: clusterParticle_EnergyFraction_Full_<PARTICLE_TYPE>
        particle_types = [target.replace(self.target_prefix, "") for target in targets]

        return particle_types, task_name

    def _detect_prediction_prefix(self) -> str:
        """Auto-detect prediction column prefix by searching available fields.

        Looks for fields that end with the first target particle type and
        extracts the prefix.

        Returns
        -------
        str
            The prediction column prefix (e.g., "cluster_regression_")
        """
        if not self.particle_types:
            return ""

        # First particle type to search for
        first_particle = self.particle_types[0]
        target_suffix = f"clusterParticle_EnergyFraction_Full_{first_particle}"

        # Search through available fields for one matching this suffix
        for field in self.dataset.clusters.fields:
            if field.endswith(target_suffix):
                # Extract prefix by removing the suffix
                prefix = field[:-len(target_suffix)]
                return prefix

        # Fallback if not found
        return ""

    def _get_fracs(
        self, frac_truth: str, down_lim: float, up_lim: float
    ) -> np.ndarray:
        """Get predicted fractions for events within a truth fraction range.

        Parameters
        ----------
        frac_truth : str
            True energy fraction column name
        down_lim : float
            Lower limit for truth fraction selection
        up_lim : float
            Upper limit for truth fraction selection

        Returns
        -------
        np.ndarray
            Array of shape (n_events, n_particle_types) with predicted fractions
        """
        mask = (
            (self.dataset.clusters[frac_truth] >= down_lim)
            & (self.dataset.clusters[frac_truth] < up_lim)
        )

        predictions = []
        for i, particle_type in enumerate(self.particle_types):
            pred_col = f"{self.pred_prefix}clusterParticle_EnergyFraction_Full_{particle_type}"
            predictions.append(self.dataset.clusters[pred_col][mask])

        return np.array(predictions).T

    def plot_scatter(self, particle_type: str) -> None:
        """Create scatter plot of true vs predicted energy fractions.

        Parameters
        ----------
        particle_type : str
            Particle type to plot (e.g., "PHOTONS", "MUON")
        """
        true_col = f"clusterParticle_EnergyFraction_Full_{particle_type}"
        pred_col = f"{self.pred_prefix}clusterParticle_EnergyFraction_Full_{particle_type}"

        x = np.array(self.dataset.clusters[true_col])
        y = np.array(self.dataset.clusters[pred_col])

        self.plotter.plot_scatter(x, y, particle_type)

    def plot_2d_histogram(self, particle_type: str, bins: int = 10) -> None:
        """Create 2D histogram of true vs predicted energy fractions.

        Parameters
        ----------
        particle_type : str
            Particle type to plot (e.g., "PHOTONS", "MUON")
        bins : int, optional
            Number of bins for the histogram, by default 10
        """
        true_col = f"clusterParticle_EnergyFraction_Full_{particle_type}"
        pred_col = f"{self.pred_prefix}clusterParticle_EnergyFraction_Full_{particle_type}"

        x = np.array(self.dataset.clusters[true_col])
        y = np.array(self.dataset.clusters[pred_col])

        self.plotter.plot_2d_histogram(x, y, particle_type, bins)

    def plot_energy_fractions(self) -> None:
        """Create energy fraction plots across all particles for different truth ranges.

        Generates plots showing average predicted fractions as a function of
        truth energy fraction ranges. Creates both bar plots for each particle type
        and confusion matrix visualizations.
        """
        # Define energy fraction ranges
        ranges = [
            (0.0, 0.2, "e0_20", r"$0.0 \leq e^{\mathrm{true}} < 0.2$"),
            (0.2, 0.5, "e20_50", r"$0.2 \leq e^{\mathrm{true}} < 0.5$"),
            (0.5, 0.6, "e50_60", r"$0.5 \leq e^{\mathrm{true}} < 0.6$"),
            (0.6, 0.7, "e60_70", r"$0.6 \leq e^{\mathrm{true}} < 0.7$"),
            (0.7, 0.8, "e70_80", r"$0.7 \leq e^{\mathrm{true}} < 0.8$"),
            (0.8, 0.9, "e80_90", r"$0.8 \leq e^{\mathrm{true}} < 0.9$"),
            (0.9, 1.0, "e90_100", r"$0.9 \leq e^{\mathrm{true}} < 1.0$"),
            (1.0, 2.0, "e100", r"$e^{\mathrm{true}} = 1.0$"),
        ]

        # Create color map for ranges
        colors = [
            "magenta",
            "cyan",
            "black",
            "darkorange",
            "royalblue",
            "crimson",
            "purple",
            "forestgreen",
        ]

        # Collect values for all ranges and particles
        all_values = {tag: [] for _, _, tag, _ in ranges}
        tag_to_label = {tag: label for _, _, tag, label in ranges}

        for particle_type in self.particle_types:
            true_col = f"clusterParticle_EnergyFraction_Full_{particle_type}"

            for (down_lim, up_lim, tag, label), color in zip(ranges, colors):
                fracs = self._get_fracs(true_col, down_lim, up_lim)

                if len(fracs) == 0:
                    print(
                        f"Warning: No events for {particle_type} in range [{down_lim}, {up_lim}]"
                    )
                    continue

                # Create histogram
                h = hist.Hist(
                    hist.axis.StrCategory(self.particle_types),
                    storage=hist.storage.Weight(),
                )

                # Fill histogram
                h.fill(
                    self.particle_types * len(fracs),
                    weight=fracs.flatten() / len(fracs),
                )

                all_values[tag].append(h.values())

            # Plot energy fractions for this particle
            self.plotter.plot_energy_fractions(
                particle_type, ranges, all_values, tag_to_label, colors
            )

        # Create confusion matrices for each range
        for tag, values_list in all_values.items():
            if len(values_list) == 0:
                continue
            title_label = tag_to_label.get(tag, tag)
            self.plotter.plot_confusion_matrix(values_list, tag, title_label)

    def run_all(self) -> None:
        """Run all analysis steps.

        Sequentially generates scatter plots, 2D histograms, and energy fraction
        analysis plots for all particle types.
        """
        print(f"\n{'='*60}")
        print(f"Analyzing: {self.sample_name}")
        print(f"Model prediction file: {self.model_predictions_path}")
        print(f"Config: {self.config_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Particle types: {len(self.particle_types)}")
        print(f"{'='*60}\n")

        # Run scatter plots
        print("Generating scatter plots...")
        for particle_type in self.particle_types:
            self.plot_scatter(particle_type)

        # Run 2D histograms
        print("\nGenerating 2D histograms...")
        for particle_type in self.particle_types:
            self.plot_2d_histogram(particle_type)

        # Run energy fraction analysis
        print("\nGenerating energy fraction plots...")
        self.plot_energy_fractions()

        print(f"\n{'='*60}")
        print(f"Analysis complete! Output saved to: {self.output_dir}")
        print(f"{'='*60}\n")
