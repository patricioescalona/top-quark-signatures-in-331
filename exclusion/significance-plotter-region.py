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
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "significance-summary.txt"
DEFAULT_OUTPUT = BASE_DIR / "exclusion-region.png"
DEFAULT_LUMINOSITIES_FB = (450.0, 3000.0)
DEFAULT_EXCLUSION_THRESHOLD = 1.64
DEFAULT_CURVE_BAND_FACTOR = 3.0
DEFAULT_MAX_TRIANGLE_SPAN = 8.0
DEFAULT_INTERPOLATION_POINTS = 320
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
        description="Plot the exclusion region in the mass-tanphi plane."
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
            "Output image path for the exclusion-region plot. "
            "Default: exclusion/exclusion-region.png"
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
        "--exclusion-threshold",
        type=float,
        default=DEFAULT_EXCLUSION_THRESHOLD,
        help=(
            "Significance threshold used to define the excluded region. "
            "Default: 1.64"
        ),
    )
    parser.add_argument(
        "--curve-band-factor",
        type=float,
        default=DEFAULT_CURVE_BAND_FACTOR,
        help=(
            "Use only points with significance between threshold/factor and "
            "threshold*factor when building the 95%% CL curve. Default: 3.0"
        ),
    )
    parser.add_argument(
        "--max-triangle-span",
        type=float,
        default=DEFAULT_MAX_TRIANGLE_SPAN,
        help=(
            "Mask triangulation cells whose longest edge is too large after "
            "normalizing by the typical mass and log10(tanphi) spacing. "
            "Default: 8.0"
        ),
    )
    parser.add_argument(
        "--interpolation-points",
        type=int,
        default=DEFAULT_INTERPOLATION_POINTS,
        help=(
            "Number of points per axis in the interpolated grid used to draw the "
            "95%% CL contour. Default: 320"
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


def rows_to_arrays(
    rows: list[tuple[float, float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masses = np.array([row[0] for row in rows], dtype=float)
    tanphis = np.array([row[1] for row in rows], dtype=float)
    significances = np.array([row[2] for row in rows], dtype=float)
    return masses, tanphis, significances


def build_filtered_triangulation(
    masses: np.ndarray,
    log_tanphis: np.ndarray,
    max_triangle_span: float,
) -> mtri.Triangulation:
    triangulation = mtri.Triangulation(masses, log_tanphis)
    if triangulation.triangles.size == 0:
        return triangulation

    unique_masses = np.unique(masses)
    unique_log_tanphis = np.unique(log_tanphis)
    mass_scale = float(np.median(np.diff(unique_masses))) if unique_masses.size > 1 else 1.0
    tanphi_scale = (
        float(np.median(np.diff(unique_log_tanphis)))
        if unique_log_tanphis.size > 1
        else 1.0
    )
    mass_scale = mass_scale if mass_scale > 0.0 else 1.0
    tanphi_scale = tanphi_scale if tanphi_scale > 0.0 else 1.0

    normalized_points = np.column_stack(
        [masses / mass_scale, log_tanphis / tanphi_scale]
    )
    triangle_points = normalized_points[triangulation.triangles]
    edge_lengths = np.stack(
        [
            np.linalg.norm(triangle_points[:, 1] - triangle_points[:, 0], axis=1),
            np.linalg.norm(triangle_points[:, 2] - triangle_points[:, 1], axis=1),
            np.linalg.norm(triangle_points[:, 0] - triangle_points[:, 2], axis=1),
        ],
        axis=1,
    )
    triangle_mask = np.max(edge_lengths, axis=1) > max_triangle_span
    triangulation.set_mask(triangle_mask)
    return triangulation


def build_scatter_norm(
    rows_by_luminosity: list[tuple[float, list[tuple[float, float, float]]]],
    threshold: float,
) -> tuple[float, float, TwoSlopeNorm]:
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
    return shared_vmin, shared_vmax, norm


def select_curve_band_points(
    masses: np.ndarray,
    tanphis: np.ndarray,
    significances: np.ndarray,
    threshold: float,
    curve_band_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if curve_band_factor <= 1.0:
        raise ValueError("--curve-band-factor must be greater than 1.")

    lower_bound = threshold / curve_band_factor
    upper_bound = threshold * curve_band_factor
    band_mask = (significances >= lower_bound) & (significances <= upper_bound)
    if np.count_nonzero(band_mask) < 4:
        raise ValueError(
            "Not enough points remain in the curve band. "
            "Increase --curve-band-factor."
        )

    return masses[band_mask], tanphis[band_mask], significances[band_mask]


def interpolate_log_significance(
    masses: np.ndarray,
    tanphis: np.ndarray,
    log_significances: np.ndarray,
    max_triangle_span: float,
    interpolation_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    log_tanphis = np.log10(tanphis)
    triangulation = build_filtered_triangulation(
        masses,
        log_tanphis,
        max_triangle_span,
    )

    interpolator = mtri.CubicTriInterpolator(triangulation, log_significances)
    mass_axis = np.linspace(float(np.min(masses)), float(np.max(masses)), interpolation_points)
    log_tanphi_axis = np.linspace(
        float(np.min(log_tanphis)),
        float(np.max(log_tanphis)),
        interpolation_points,
    )
    mass_grid, log_tanphi_grid = np.meshgrid(mass_axis, log_tanphi_axis)
    interpolated_log_significance = interpolator(mass_grid, log_tanphi_grid)
    tanphi_grid = np.power(10.0, log_tanphi_grid)
    return mass_grid, tanphi_grid, interpolated_log_significance


def make_exclusion_plot(
    rows_by_luminosity: list[tuple[float, list[tuple[float, float, float]]]],
    output_path: Path,
    threshold: float,
    curve_band_factor: float,
    max_triangle_span: float,
    interpolation_points: int,
) -> Path:
    shared_vmin, shared_vmax, scatter_norm = build_scatter_norm(
        rows_by_luminosity,
        threshold,
    )
    fig, axes = plt.subplots(
        1,
        len(rows_by_luminosity),
        figsize=(8.2 * len(rows_by_luminosity), 6.0),
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
        masses, tanphis, significances = rows_to_arrays(rows)
        clipped_significances = significances.copy()
        clipped_significances[clipped_significances <= 0.0] = shared_vmin
        log_significances = np.log10(clipped_significances)
        curve_masses, curve_tanphis, curve_significances = select_curve_band_points(
            masses,
            tanphis,
            clipped_significances,
            threshold,
            curve_band_factor,
        )
        mass_grid, tanphi_grid, interpolated_log_significance = interpolate_log_significance(
            curve_masses,
            curve_tanphis,
            np.log10(curve_significances),
            max_triangle_span,
            interpolation_points,
        )
        threshold_level = math.log10(threshold)
        valid_log_significance = interpolated_log_significance.compressed()
        if valid_log_significance.size == 0:
            raise ValueError("Interpolation failed; no valid region remained after masking.")
        lower_bound = float(np.min(valid_log_significance)) - 1e-9
        upper_bound = float(np.max(valid_log_significance)) + 1e-9

        ax.contourf(
            mass_grid,
            tanphi_grid,
            interpolated_log_significance,
            levels=[lower_bound, threshold_level, upper_bound],
            colors=["#cfe8cf", "#f3b3b3"],
            alpha=0.28,
        )
        ax.contour(
            mass_grid,
            tanphi_grid,
            interpolated_log_significance,
            levels=[threshold_level],
            colors=["#8b0000"],
            linewidths=2.4,
        )

        scatter = ax.scatter(
            masses,
            tanphis,
            c=log_significances,
            cmap=SCATTER_CMAP,
            norm=scatter_norm,
            s=115,
            edgecolors="black",
            linewidths=0.45,
            zorder=3,
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
        ax.set_title(
            rf"95% CL curve at {luminosity_fb:g} fb$^{{-1}}$"
        )
        ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.5)

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
    output_path = make_exclusion_plot(
        rows_by_luminosity,
        args.output.resolve(),
        args.exclusion_threshold,
        args.curve_band_factor,
        args.max_triangle_span,
        args.interpolation_points,
    )
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
