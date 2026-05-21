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
from matplotlib.colors import LogNorm


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "significance-summary.txt"
DEFAULT_OUTPUT = BASE_DIR / "significance-map.png"


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
        help="Output image path. Default: exclusion/significance-map.png",
    )
    return parser.parse_args()


def load_summary(summary_path: Path) -> list[tuple[float, float, float]]:
    if not summary_path.is_file():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    rows: list[tuple[float, float, float]] = []
    with summary_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("m |"):
                continue

            parts = [part.strip() for part in stripped.split("|")]
            if len(parts) != 3:
                raise ValueError(
                    f"Expected 3 columns in {summary_path} at line {line_number}, got: {stripped}"
                )

            mass, tanphi, significance = map(float, parts)
            rows.append((mass, tanphi, significance))

    if not rows:
        raise ValueError(f"No data rows found in {summary_path}")

    return rows


def make_plot(
    rows: list[tuple[float, float, float]],
    output_path: Path,
) -> Path:
    masses = np.array([row[0] for row in rows], dtype=float)
    tanphis = np.array([row[1] for row in rows], dtype=float)
    significances = np.array([row[2] for row in rows], dtype=float)

    positive_significances = significances[significances > 0.0]
    if positive_significances.size == 0:
        raise ValueError("All significances are zero; logarithmic color scale is undefined.")

    color_values = significances.copy()
    min_positive = float(np.min(positive_significances))
    color_values[color_values <= 0.0] = min_positive

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    scatter = ax.scatter(
        masses,
        tanphis,
        c=color_values,
        cmap="viridis",
        norm=LogNorm(vmin=min_positive, vmax=float(np.max(color_values))),
        s=140,
        edgecolors="black",
        linewidths=0.4,
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label(r"$Z_A$ after cuts")

    ax.set_xlabel("Mass")
    ax.set_ylabel("tanphi")
    ax.set_yscale("log")
    ax.set_xticks(sorted(set(masses)))
    ax.set_yticks(sorted(set(tanphis)))
    ax.set_yticklabels([f"{value:g}" for value in sorted(set(tanphis))])
    ax.set_title(r"Final Asimov Significance $Z_A$")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path.resolve()


def main() -> int:
    args = parse_args()
    rows = load_summary(args.input.resolve())
    output_path = make_plot(rows, args.output.resolve())
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
