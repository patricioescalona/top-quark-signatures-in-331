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
import matplotlib.patheffects as pe
import matplotlib.tri as mtri
from matplotlib.lines import Line2D


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "significance-summary.txt"
DEFAULT_OUTPUT = BASE_DIR / "exclusion-region.png"
DEFAULT_LUMINOSITIES_FB = (450.0, 3000.0)
DEFAULT_EXCLUSION_THRESHOLD = 1.64
DEFAULT_CURVE_BAND_FACTOR = 2.0
DEFAULT_MAX_TRIANGLE_SPAN = 20.0
DEFAULT_INTERPOLATION_POINTS = 1000


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
            "threshold*factor when building the 95%% CL curve. Default: 2.0"
        ),
    )
    parser.add_argument(
        "--max-triangle-span",
        type=float,
        default=DEFAULT_MAX_TRIANGLE_SPAN,
        help=(
            "Mask triangulation cells whose longest edge is too large after "
            "normalizing by the typical mass and log10(tanphi) spacing. "
            "Default: 20.0"
        ),
    )
    parser.add_argument(
        "--interpolation-points",
        type=int,
        default=DEFAULT_INTERPOLATION_POINTS,
        help=(
            "Number of points per axis in the interpolated grid used to draw the "
            "95%% CL contour. Default: 1000"
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


def select_curve_band_points(
    masses: np.ndarray,
    tanphis: np.ndarray,
    significances: np.ndarray,
    threshold: float,
    curve_band_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    band_mask = build_curve_band_mask(significances, threshold, curve_band_factor)
    if np.count_nonzero(band_mask) < 4:
        raise ValueError(
            "Not enough points remain in the curve band. "
            "Increase --curve-band-factor."
        )

    return masses[band_mask], tanphis[band_mask], significances[band_mask]


def build_curve_band_mask(
    significances: np.ndarray,
    threshold: float,
    curve_band_factor: float,
) -> np.ndarray:
    if curve_band_factor <= 1.0:
        raise ValueError("--curve-band-factor must be greater than 1.")

    lower_bound = threshold / curve_band_factor
    upper_bound = threshold * curve_band_factor
    return (significances >= lower_bound) & (significances <= upper_bound)


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
    fig, axes = plt.subplots(
        1,
        len(rows_by_luminosity),
        figsize=(8.2 * len(rows_by_luminosity), 6.0),
        constrained_layout=True,
        squeeze=False,
    )
    axes_row = axes[0]

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#b10000",
            linewidth=3.2,
            label=rf"95% CL exclusion ($Z_A = {threshold:g}$)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#f4f1e6",
            markeredgecolor="#4a4a4a",
            markersize=7,
            label="Points used in contour",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#d9e2e8",
            markeredgecolor="#8a8a8a",
            alpha=0.35,
            markersize=7,
            label="Points outside band",
        ),
    ]

    for index, (luminosity_fb, rows) in enumerate(rows_by_luminosity):
        ax = axes_row[index]
        masses, tanphis, significances = rows_to_arrays(rows)
        band_mask = build_curve_band_mask(significances, threshold, curve_band_factor)
        curve_masses, curve_tanphis, curve_significances = select_curve_band_points(
            masses,
            tanphis,
            significances,
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
        contour = ax.contour(
            mass_grid,
            tanphi_grid,
            interpolated_log_significance,
            levels=[threshold_level],
            colors=["#b10000"],
            linewidths=3.2,
            zorder=5,
        )
        contour.set_path_effects(
            [
                pe.Stroke(linewidth=5.8, foreground="white", alpha=0.96),
                pe.Normal(),
            ]
        )

        ax.scatter(
            masses[~band_mask],
            tanphis[~band_mask],
            c="#d9e2e8",
            s=52,
            edgecolors="#8a8a8a",
            linewidths=0.45,
            alpha=0.35,
            zorder=2,
        )
        ax.scatter(
            masses[band_mask],
            tanphis[band_mask],
            c="#f4f1e6",
            s=78,
            edgecolors="#4a4a4a",
            linewidths=0.65,
            alpha=0.95,
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
            rf"95% CL exclusion at {luminosity_fb:g} fb$^{{-1}}$"
        )
        ax.grid(True, which="both", linestyle=":", linewidth=0.55, alpha=0.3)

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
