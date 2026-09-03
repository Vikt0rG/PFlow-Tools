"""
Cluster Regression Analysis Module

Provides the ClusterRegressionAnalyzer class for analysis of cluster regression
SALT model outputs including scatter plots, 2D histograms, and confusion
matrices for different truth energy fraction ranges.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import warnings
import yaml

import numpy as np
import hist

from .utils import h5_prep as h5_module
from .cluster_regression_plotter import ClusterRegressionPlotter


class ClusterRegressionAnalyzer:
    """Analyze cluster regression model predictions.

    Provides analysis of cluster regression model outputs including
    scatter plots, 2D histograms, and confusion matrices for different truth
    energy fraction ranges. Uses ClusterRegressionPlotter for visualization.

    Parameters
    ----------
    model_predictions_path : str
        Path to the HDF5 file with model predictions
    config_path : str
        Path to the YAML config file used for training
    sample_name : str, optional
        Human-readable name for the sample (e.g., "Di-jets", "Drell-Yan").
    output_dir : str, optional
        Output directory for plots, by default "./output"
    target_prefix : str, optional
        Prefix to remove from target names to get particle types,
        by default "clusterParticle_EnergyFraction_Full_"
    residual_particle_type : str, optional
        Particle type to infer as a residual (1 - sum of other predictions)
        when it is not explicitly regressed; if provided, it will be added
        to particle_types for plotting
    auto_residual : bool, optional
        If True, infer a residual particle type when exactly one truth
        particle type is present in the dataset but not in the config targets

    Raises
    ------
    FileNotFoundError
        If model prediction file or config file does not exist
    ValueError
        If config structure is not recognized
    """

    def __init__(
        self,
        model_predictions_path: str,
        config_path: str,
        sample_name: Optional[str] = None,
        output_dir: str = "./output",
        target_prefix: str = "clusterParticle_EnergyFraction_Full_",
        residual_particle_type: Optional[str] = None,
        auto_residual: bool = True,
    ):
        self.model_predictions_path = Path(model_predictions_path)
        self.config_path = Path(config_path)
        self.sample_name = sample_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_prefix = target_prefix
        self.residual_particle_type = residual_particle_type
        self.auto_residual = auto_residual
        self.residual_truth_particle_type: Optional[str] = None
        self.residual_label: Optional[str] = None

        self.config = self._load_config()
        self.dataset = h5_module.load_hdf(str(self.model_predictions_path))

        # Extract particle types and task name from config
        self.particle_types, self.task_name = self._extract_particle_types()

        # Auto-detect the prediction prefix by looking at available fields
        self.pred_prefix = self._detect_prediction_prefix()

        # Update particle types if a residual should be inferred or forced
        self._apply_residual_particle_type()

        # Build display labels for plots
        self._build_particle_type_labels()

        # Initialize plotter
        self.plotter = ClusterRegressionPlotter(
            str(self.output_dir),
            self.sample_name,
            self.particle_types,
            particle_type_labels=self.particle_type_labels
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
        """Extract to-be-regressed particle types and task name from config targets.

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

        Raises
        ------
        KeyError
            If no particle types are available to search for
            If no matching field is found for the first particle type
        """
        if not self.particle_types:
            raise KeyError("No particle types found to auto-detect prediction prefix")

        # First particle type to search for
        first_particle = self.particle_types[0]
        target_suffix = f"clusterParticle_EnergyFraction_Full_{first_particle}"

        # Search through available fields for one matching this suffix
        for field in self.dataset.clusters.fields:
            if field.endswith(target_suffix) and len(field) > len(target_suffix):
                # Extract prefix by removing the suffix
                prefix = field[:-len(target_suffix)]
                return prefix

        # Raise an error if no matching field is found
        raise KeyError(f"Could not auto-detect prediction prefix for {target_suffix} in dataset fields!")

    def _collect_truth_particle_types(self) -> List[str]:
        """Collect particle types from config or truth columns in the dataset.

        Returns
        -------
        list
            Particle type names found in truth columns
        """
        truth_particle_types = (
            self.config.get("data", {}).get("truth_particle_types") if self.config else None
        )
        if truth_particle_types:
            return list(truth_particle_types)
        else:            
            warnings.warn("Truth particle types not found in config; auto residual inference may not work.")    

    def _apply_residual_particle_type(self) -> None:
        """Apply residual particle type handling to particle_types."""
        # If a residual particle type is explicitly provided, use it and add to particle_types if not present
        if self.residual_particle_type:
            self.residual_truth_particle_type = self.residual_particle_type
            self.residual_label = self.residual_particle_type
            if self.residual_particle_type not in self.particle_types:
                self.particle_types.append(self.residual_particle_type)
            return

        # If auto_residual is disabled or truth particle types are not found, do not attempt to infer a residual particle type
        if not self.auto_residual or not self._collect_truth_particle_types():
            return

        # Auto-detect residual particle type if exactly one truth particle type is missing from config targets
        truth_particle_types = self._collect_truth_particle_types()
        missing = [t for t in truth_particle_types if t not in self.particle_types]
        if len(missing) == 1:
            self.residual_truth_particle_type = missing[0]
            self.residual_label = str(missing[0])
            self.residual_particle_type = missing[0]
            self.particle_types.append(missing[0])
        elif len(missing) > 1:
            print(
                "Warning: Multiple truth particle types are missing from config targets; "
                "residual particle type inference skipped."
            )

    def _build_particle_type_labels(self) -> None:
        """Build display labels aligned with particle_types."""
        self.particle_type_labels = list(self.particle_types)
        self._particle_label_map = dict(zip(self.particle_types, self.particle_type_labels))

        if not self.residual_truth_particle_type or not self.residual_label:
            return

        if self.residual_truth_particle_type in self._particle_label_map:
            self._particle_label_map[self.residual_truth_particle_type] = self.residual_label
            self.particle_type_labels = [
                self._particle_label_map[ptype] for ptype in self.particle_types
            ]

    def _get_particle_label(self, particle_type: str) -> str:
        """Get display label for a particle type."""
        return self._particle_label_map.get(particle_type, particle_type)

    def _get_predicted_fraction(
        self, particle_type: str, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Get predicted fraction for a particle type.

        If the prediction column is missing and the particle type is marked as
        residual, compute it as 1 - sum(other predictions).
        """
        pred_col = f"{self.pred_prefix}{self.target_prefix}{particle_type}"
        if pred_col in self.dataset.clusters.fields:
            data = np.array(self.dataset.clusters[pred_col])
            return data[mask] if mask is not None else data

        if (
            self.residual_truth_particle_type
            and particle_type == self.residual_truth_particle_type
        ):
            other_types = [t for t in self.particle_types if t != particle_type]
            preds = []
            for other_type in other_types:
                other_col = f"{self.pred_prefix}{self.target_prefix}{other_type}"
                if other_col not in self.dataset.clusters.fields:
                    raise ValueError(
                        "Residual prediction requires all other particle type "
                        "predictions to be present."
                    )
                other_data = np.array(self.dataset.clusters[other_col])
                preds.append(other_data[mask] if mask is not None else other_data)
            summed = np.sum(preds, axis=0)
            return 1.0 - summed

        raise KeyError(
            f"Prediction column not found for particle type '{particle_type}'"
        )

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
        for particle_type in self.particle_types:
            predictions.append(self._get_predicted_fraction(particle_type, mask))

        return np.array(predictions).T

    def plot_scatter(self, particle_type: str) -> None:
        """Create scatter plot of true vs predicted energy fractions.

        Parameters
        ----------
        particle_type : str
            Particle type to plot (e.g., "PHOTONS", "MUON")
        """
        true_col = f"clusterParticle_EnergyFraction_Full_{particle_type}"
        x = np.array(self.dataset.clusters[true_col])
        y = self._get_predicted_fraction(particle_type)

        label = self._get_particle_label(particle_type)
        self.plotter.plot_scatter(x, y, label)

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
        x = np.array(self.dataset.clusters[true_col])
        y = self._get_predicted_fraction(particle_type)

        label = self._get_particle_label(particle_type)
        self.plotter.plot_2d_histogram(x, y, label, bins)

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
                    all_values[tag].append(np.zeros(len(self.particle_type_labels)))
                    continue

                # Create histogram
                h = hist.Hist(
                    hist.axis.StrCategory(self.particle_type_labels),
                    storage=hist.storage.Weight(),
                )

                # Fill histogram
                h.fill(
                    self.particle_type_labels * len(fracs),
                    weight=fracs.flatten() / len(fracs),
                )

                all_values[tag].append(h.values())

            # Plot energy fractions for this particle
            display_name = self._get_particle_label(particle_type)
            self.plotter.plot_energy_fractions(
                display_name, ranges, all_values, tag_to_label, colors
            )

        # Create energy heatmaps for each range
        for tag, values_list in all_values.items():
            if len(values_list) == 0:
                continue
            title_label = tag_to_label.get(tag, tag)
            self.plotter.plot_energy_heatmap(values_list, tag, title_label)

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
