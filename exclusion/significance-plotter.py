#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "significance-summary.txt"
DEFAULT_OUTPUT = BASE_DIR / "significance-map-comparison.png"
DEFAULT_EXCLUSION_OUTPUT = BASE_DIR / "exclusion-region.png"
DEFAULT_LUMINOSITIES_FB = (450.0, 3000.0)
DEFAULT_EXCLUSION_THRESHOLD = 1.64
SCATTER_CMAP = LinearSegmentedColormap.from_list(
    "significance_threshold_map",
    [
        (0.0, "#8fa6b3"),
        (0.42, "#d7e1e8"),
        (0.5, "#f6f1df"),
        (0.58, "#f3c98b"),
        (1.0, "#b33a2f"),
    ],
)


def select_axis_ticks(values: np.ndarray, max_ticks: int = 5) -> list[float]:
    unique_values = np.unique(values.astype(float))
    if unique_values.size <= max_ticks:
        return unique_values.tolist()

    tick_indices = np.linspace(0, unique_values.size - 1, num=max_ticks, dtype=int)
    return unique_values[np.unique(tick_indices)].tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot mass versus tanphi using the final Asimov significance as a color map."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input summary text file. Default: exclusion/significance-summary.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output image path for the combined comparison plot. "
            "Default: exclusion/significance-map-comparison.png"
        ),
    )
    parser.add_argument(
        "--luminosities",
        type=float,
        nargs="+",
        default=list(DEFAULT_LUMINOSITIES_FB),
        help=(
            "Luminosity columns to plot from significance-summary.txt. "
            "Default: 450 3000"
        ),
    )
    parser.add_argument(
        "--exclusion-output",
        type=Path,
        default=DEFAULT_EXCLUSION_OUTPUT,
        help=(
            "Output image path for the exclusion-region plot. "
            "Default: exclusion/exclusion-region.png"
        ),
    )
    parser.add_argument(
        "--exclusion-threshold",
        type=float,
        default=DEFAULT_EXCLUSION_THRESHOLD,
        help=(
            "Significance threshold used to define the excluded region. "
            "Default: 1.64"
        ),
    )
    return parser.parse_args()


def load_summary(
    summary_path: Path,
    luminosity_fb: float,
) -> list[tuple[float, float, float]]:
    if not summary_path.is_file():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    rows: list[tuple[float, float, float]] = []
    target_column = f"Z_A_{luminosity_fb:g}fb"
    column_index: int | None = None
    with summary_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            parts = [part.strip() for part in stripped.split("|")]
            if line_number == 1:
                if len(parts) < 3:
                    raise ValueError(
                        f"Expected at least 3 columns in the header of {summary_path}."
                    )
                try:
                    column_index = parts.index(target_column)
                except ValueError as exc:
                    raise ValueError(
                        f"Column {target_column} not found in {summary_path}. "
                        f"Available columns: {parts}"
                    ) from exc
                continue

            parts = [part.strip() for part in stripped.split("|")]
            if column_index is None:
                raise ValueError(f"Could not parse the header of {summary_path}")
            if len(parts) <= column_index:
                raise ValueError(
                    f"Expected column {target_column} in {summary_path} at line "
                    f"{line_number}, got: {stripped}"
                )

            mass = float(parts[0])
            tanphi = float(parts[1])
            significance = float(parts[column_index])
            rows.append((mass, tanphi, significance))

    if not rows:
        raise ValueError(f"No data rows found in {summary_path}")

    return rows


def make_plot(
    rows_by_luminosity: list[tuple[float, list[tuple[float, float, float]]]],
    output_path: Path,
    threshold: float,
) -> Path:
    all_significances = np.concatenate(
        [
            np.array([row[2] for row in rows], dtype=float)
            for _, rows in rows_by_luminosity
        ]
    )
    positive_significances = all_significances[all_significances > 0.0]
    if positive_significances.size == 0:
        raise ValueError("All significances are zero; logarithmic color scale is undefined.")

    shared_vmin = float(np.min(positive_significances))
    shared_vmax = float(np.max(all_significances))
    if not (shared_vmin < threshold < shared_vmax):
        raise ValueError(
            "The exclusion threshold must lie inside the positive significance range "
            "to center the scatter color scale."
        )

    log_significances = np.log10(positive_significances)
    norm = TwoSlopeNorm(
        vmin=float(np.min(log_significances)),
        vcenter=math.log10(threshold),
        vmax=math.log10(shared_vmax),
    )

    fig, axes = plt.subplots(
        1,
        len(rows_by_luminosity),
        figsize=(8.4 * len(rows_by_luminosity), 6.2),
        constrained_layout=True,
        squeeze=False,
    )
    axes_row = axes[0]
    scatter = None

    for index, (luminosity_fb, rows) in enumerate(rows_by_luminosity):
        ax = axes_row[index]
        masses = np.array([row[0] for row in rows], dtype=float)
        tanphis = np.array([row[1] for row in rows], dtype=float)
        significances = np.array([row[2] for row in rows], dtype=float)

        color_values = significances.copy()
        color_values[color_values <= 0.0] = shared_vmin
        color_values = np.log10(color_values)

        scatter = ax.scatter(
            masses,
            tanphis,
            c=color_values,
            cmap=SCATTER_CMAP,
            norm=norm,
            s=140,
            edgecolors="black",
            linewidths=0.4,
        )
        ax.set_xlabel("Mass")
        if index == 0:
            ax.set_ylabel("tanphi")
        ax.set_yscale("log")
        x_ticks = select_axis_ticks(masses)
        y_ticks = select_axis_ticks(tanphis)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{value:g}" for value in y_ticks])
        ax.set_title(rf"$Z_A$ at {luminosity_fb:g} fb$^{{-1}}$")
        ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.5)

    if scatter is None:
        raise ValueError("No luminosity panels were created.")

    colorbar = fig.colorbar(scatter, ax=list(axes_row), pad=0.035, fraction=0.06)
    colorbar.set_label(r"$Z_A$ after cuts")
    colorbar.ax.tick_params(labelsize=9, pad=4)
    colorbar.ax.yaxis.labelpad = 18

    tick_candidates = np.array(
        [shared_vmin, 0.5, 1.0, threshold, 3.0, 10.0, 100.0, shared_vmax],
        dtype=float,
    )
    tick_values = sorted(
        {
            value
            for value in tick_candidates
            if shared_vmin <= value <= shared_vmax
        }
    )
    colorbar.set_ticks(np.log10(tick_values))
    colorbar.set_ticklabels(
        [
            f"{value:.2f}" if value < 10.0 else f"{value:.0f}"
            for value in tick_values
        ]
    )

    threshold_position = math.log10(threshold)
    colorbar.ax.axhline(threshold_position, color="black", linewidth=1.4, linestyle="--")
    colorbar.ax.text(
        0.5,
        threshold_position,
        "95% CL",
        transform=colorbar.ax.get_yaxis_transform(),
        va="center",
        ha="center",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.9,
        },
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path.resolve()


def rows_to_grid(
    rows: list[tuple[float, float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masses = sorted({row[0] for row in rows})
    tanphis = sorted({row[1] for row in rows})
    mass_index = {mass: index for index, mass in enumerate(masses)}
    tanphi_index = {tanphi: index for index, tanphi in enumerate(tanphis)}
    grid = np.full((len(tanphis), len(masses)), np.nan, dtype=float)

    for mass, tanphi, significance in rows:
        grid[tanphi_index[tanphi], mass_index[mass]] = significance

    if np.isnan(grid).any():
        raise ValueError("Input rows do not form a complete mass-tanphi grid.")

    return np.array(masses, dtype=float), np.array(tanphis, dtype=float), grid


def make_exclusion_plot(
    rows_by_luminosity: list[tuple[float, list[tuple[float, float, float]]]],
    output_path: Path,
    threshold: float,
) -> Path:
    fig, axes = plt.subplots(
        1,
        len(rows_by_luminosity),
        figsize=(7.5 * len(rows_by_luminosity), 5.5),
        constrained_layout=True,
        squeeze=False,
    )
    axes_row = axes[0]

    legend_handles = [
        Patch(facecolor="#cfe8cf", edgecolor="none", alpha=0.9, label="Allowed"),
        Patch(facecolor="#f3b3b3", edgecolor="none", alpha=0.9, label="Excluded"),
        Line2D([0], [0], color="#8b0000", linewidth=2.0, label=rf"$Z_A = {threshold:g}$"),
    ]

    for index, (luminosity_fb, rows) in enumerate(rows_by_luminosity):
        ax = axes_row[index]
        masses, tanphis, significance_grid = rows_to_grid(rows)
        log_tanphis = np.log10(tanphis)
        mass_grid, log_tanphi_grid = np.meshgrid(masses, log_tanphis)
        significance_min = float(np.min(significance_grid))
        significance_max = float(np.max(significance_grid))
        lower_bound = min(significance_min, threshold) - 1e-9
        upper_bound = max(significance_max, threshold) + 1e-9

        ax.contourf(
            mass_grid,
            log_tanphi_grid,
            significance_grid,
            levels=[lower_bound, threshold, upper_bound],
            colors=["#cfe8cf", "#f3b3b3"],
            alpha=0.9,
        )
        contour = ax.contour(
            mass_grid,
            log_tanphi_grid,
            significance_grid,
            levels=[threshold],
            colors=["#8b0000"],
            linewidths=2.0,
        )

        ax.scatter(
            mass_grid.ravel(),
            log_tanphi_grid.ravel(),
            s=28,
            c="black",
            alpha=0.65,
        )
        ax.set_xlabel("Mass")
        if index == 0:
            ax.set_ylabel("tanphi")
        x_ticks = select_axis_ticks(masses)
        y_ticks = select_axis_ticks(tanphis)
        ax.set_xticks(x_ticks)
        ax.set_yticks(np.log10(y_ticks))
        ax.set_yticklabels([f"{value:g}" for value in y_ticks])
        ax.set_title(
            rf"Exclusion at {luminosity_fb:g} fb$^{{-1}}$ ($Z_A = {threshold:g}$)"
        )
        ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.5)

    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper center",
        ncol=3,
        frameon=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path.resolve()


def main() -> int:
    args = parse_args()
    rows_by_luminosity = [
        (luminosity_fb, load_summary(args.input.resolve(), luminosity_fb))
        for luminosity_fb in args.luminosities
    ]
    output_path = make_plot(
        rows_by_luminosity,
        args.output.resolve(),
        args.exclusion_threshold,
    )
    exclusion_output_path = make_exclusion_plot(
        rows_by_luminosity,
        args.exclusion_output.resolve(),
        args.exclusion_threshold,
    )
    print(f"Wrote {output_path}")
    print(f"Wrote {exclusion_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
