"""
Cluster Regression Plotter Module

Provides the ClusterRegressionPlotter class for creating visualsizations
of cluster regression model outputs.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import hist


class ClusterRegressionPlotter:
    """Handles plotting for cluster regression analysis.

    Responsible for creating all visualization outputs including scatter plots,
    2D histograms, and confusion matrices.
    """

    def __init__(
        self,
        output_dir: str,
        sample_name: str,
        particle_types: List[str],
        particle_type_labels: Optional[List[str]] = None,
    ):
        """Initialize the plotter.

        Parameters
        ----------
        output_dir : str
            Output directory for saving plots
        sample_name : str
            Human-readable name for the sample
        particle_types : List[str]
            List of particle type names for indexing
        particle_type_labels : List[str], optional
            Display labels for particle types, defaults to particle_types
        """
        self.output_dir = Path(output_dir)
        self.sample_name = sample_name
        self.particle_types = particle_types
        self.particle_type_labels = particle_type_labels or particle_types

    def plot_scatter(
        self,
        x: np.ndarray,
        y: np.ndarray,
        particle_type: str
    ) -> None:
        """Create scatter plot of true vs predicted energy fractions.

        Parameters
        ----------
        x : np.ndarray
            True energy fractions
        y : np.ndarray
            Predicted energy fractions
        particle_type : str
            Particle type to plot (e.g., "PHOTONS", "MUON")
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(x, y, alpha=0.5, s=20)

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title(f"Sample: {self.sample_name} - {particle_type}")
        ax.set_xlabel(f"Energy Fraction {particle_type} (true)")
        ax.set_ylabel(f"Energy Fraction {particle_type} (predicted)")
        ax.grid(True, alpha=0.3)

        output_path = (
            self.output_dir
            / f"scatter_{self.sample_name.replace(' ', '_')}_{particle_type}.png"
        )
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved scatter plot: {output_path}")

    def plot_2d_histogram(
        self, x: np.ndarray, y: np.ndarray, particle_type: str, bins: int = 10
    ) -> None:
        """Create 2D histogram of true vs predicted energy fractions.

        Parameters
        ----------
        x : np.ndarray
            True energy fractions
        y : np.ndarray
            Predicted energy fractions
        particle_type : str
            Particle type to plot (e.g., "PHOTONS", "MUON")
        bins : int, optional
            Number of bins for the histogram, by default 10
        """
        mask = np.isfinite(x) & np.isfinite(y)

        fig, ax = plt.subplots(figsize=(8, 8))

        # Create histogram
        xedges = np.linspace(0, 1, bins + 1)
        yedges = np.linspace(0, 1, bins + 1)
        H, _, _ = np.histogram2d(x[mask], y[mask], bins=[xedges, yedges])

        # Mask zero values and plot
        H_masked = np.ma.masked_where(H == 0, H)
        cmap = cm.viridis.copy()
        cmap.set_bad(color="white")

        mesh = ax.pcolormesh(xedges, yedges, H_masked.T, cmap=cmap)

        # Add bin counts as text
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        for i, xc in enumerate(xcenters):
            for j, yc in enumerate(ycenters):
                count = H_masked[i, j]
                if count > 0:
                    ax.text(
                        xc,
                        yc,
                        int(count),
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=6,
                        rotation=45,
                    )

        plt.colorbar(mesh, ax=ax, label="Counts")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title(f"Sample: {self.sample_name} - {particle_type}")
        ax.set_xlabel(f"Energy Fraction {particle_type} (true)")
        ax.set_ylabel(f"Energy Fraction {particle_type} (predicted)")

        output_path = (
            self.output_dir
            / f"h2d_{self.sample_name.replace(' ', '_')}_{particle_type}.png"
        )
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved 2D histogram: {output_path}")

    def plot_energy_fractions(
        self,
        particle_type: str,
        ranges: List[Tuple],
        all_values: Dict[str, List],
        tag_to_label: Dict[str, str],
        colors: List[str],
    ) -> None:
        """Create energy fraction plots for a particle type.

        Parameters
        ----------
        particle_type : str
            Particle type to plot
        ranges : List[Tuple]
            List of (down_lim, up_lim, tag, label) tuples
        all_values : Dict[str, List]
            Dictionary mapping tags to lists of histogram values
        tag_to_label : Dict[str, str]
            Dictionary mapping tags to labels
        colors : List[str]
            List of colors for each range
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        for (down_lim, up_lim, tag, label), color in zip(ranges, colors):
            if tag not in all_values or len(all_values[tag]) == 0:
                continue

            # Get current particle type's values
            particle_idx = self.particle_type_labels.index(particle_type)
            if particle_idx < len(all_values[tag]):
                h_values = all_values[tag][particle_idx]

                h = hist.Hist(
                    hist.axis.StrCategory(self.particle_type_labels),
                    storage=hist.storage.Weight(),
                )
                view = h.view()
                view.value = h_values
                view.variance = np.zeros_like(h_values)

                h.plot(ax=ax, color=color, label=label)

        ax.set_title(f"Sample: {self.sample_name}")
        ax.set_xlabel("Particle type")
        ax.set_ylabel("Average predicted cluster energy fraction")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.legend()

        plt.subplots_adjust(bottom=0.35)

        output_path = (
            self.output_dir
            / f"en_fr_{self.sample_name.replace(' ', '_')}_{particle_type}.png"
        )
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved energy fraction plot: {output_path}")

    def plot_energy_heatmap(
        self, values: List[np.ndarray], tag: str, title_label: str
    ) -> None:
        """Create energy heatmap visualization.

        Parameters
        ----------
        values : list of np.ndarray
            List of arrays, one per particle type, containing average
            predicted energy fractions for each particle category
        tag : str
            Tag for the output filename
        title_label : str
            Label for the plot title describing the energy fraction range
        """
        _, ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.35, left=0.35)

        ax.set_title(f"Sample: {self.sample_name} - {title_label}")
        ax.set_xlabel("true particle type")
        ax.set_ylabel("predicted particle type")

        from pprint import pprint

        pprint("Values for heatmap:")
        pprint(values)

        data = np.array(values).T
        size = len(self.particle_type_labels)
        img = ax.imshow(
            data,
            origin="lower",
            extent=[0, size, 0, size],
            aspect="auto",
            cmap="viridis",
        )

        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label("Average predicted energy fraction")

        threshold = (data.max() + data.min()) / 2.0 if data.size > 0 else 0.5
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                val = data[row, col]
                text_color = "white" if val < threshold else "black"

                ax.text(
                    col + 0.5,
                    row + 0.5,
                    f"{val:.2f}" if val >= 0.01 else ("0" if val == 0 else "<.01"),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=14,
                )

        ax.set_xticks(np.arange(len(self.particle_type_labels)) + 0.5)
        ax.set_yticks(np.arange(len(self.particle_type_labels)) + 0.5)
        ax.set_xticklabels(self.particle_type_labels, rotation=90)
        ax.set_yticklabels(self.particle_type_labels)

        output_path = (
            self.output_dir
            / f"energy_heatmap_{self.sample_name.replace(' ', '_')}_{tag}.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved energy heatmap: {output_path}")
