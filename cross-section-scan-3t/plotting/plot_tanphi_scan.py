from __future__ import annotations

import argparse
import csv
import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCAN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = SCAN_ROOT / "results"
DEFAULT_FIGURES = SCAN_ROOT / "figures"
DEFAULT_INPUTS = [
    DEFAULT_RESULTS / "bsm-tttbar-tanphi-001.csv",
    DEFAULT_RESULTS / "bsm-tttbar-tanphi-01.csv",
    DEFAULT_RESULTS / "bsm-tttbar-tanphi-1.csv",
    DEFAULT_RESULTS / "bsm-tttbar-tanphi-10.csv",
    DEFAULT_RESULTS / "bsm-tttbar-tanphi-100.csv",
]
TANPHI_PATTERN = re.compile(r"tanphi-(.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot pp -> tt t~ cross sections from one or more scan CSV files."
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="*",
        default=DEFAULT_INPUTS,
        help="Input CSV files to plot together.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_FIGURES / "bsm-tttbar-tanphi-comparison.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--process-math",
        default=r"tt\bar{t}",
        help=(
            "Final-state expression used in the y-axis label. "
            r"For example: 'tt\bar{t}' or 't\bar{t}\bar{t}'."
        ),
    )
    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="Use a linear scale on the y axis instead of the default logarithmic one.",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=None,
        help="Lower limit for the y axis.",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="Upper limit for the y axis.",
    )
    return parser.parse_args()


def load_scan(path: Path) -> tuple[list[float], list[float]]:
    masses: list[float] = []
    cross_sections: list[float] = []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") != "ok":
                continue
            masses.append(float(row["mass"]))
            cross_sections.append(float(row["cross_section_pb"]))

    if not masses:
        raise ValueError(f"No valid 'ok' rows found in {path}")

    return masses, cross_sections


def infer_tanphi_token(path: Path) -> str:
    match = TANPHI_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(f"Could not infer tanphi from file name: {path.name}")

    token = match.group(1)
    if re.fullmatch(r"0\d+", token):
        return f"0.{token[1:]}"
    return token


def format_tanphi_value(token: str) -> str:
    try:
        value = float(token)
    except ValueError:
        return token

    if value <= 0:
        return token

    if math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return "1"

    exponent = round(math.log10(value))
    if math.isclose(value, 10**exponent, rel_tol=1e-12, abs_tol=0.0):
        return rf"10^{{{exponent}}}"

    return f"{value:g}"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 18,
            "axes.labelsize": 28,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 17,
            "axes.linewidth": 0.9,
        }
    )


def main() -> int:
    args = parse_args()
    input_paths = [path.expanduser().resolve() for path in args.inputs]

    scans = []
    all_xs: list[float] = []
    all_masses: list[float] = []

    for path in input_paths:
        masses, cross_sections = load_scan(path)
        scans.append((path, masses, cross_sections))
        all_masses.extend(masses)
        all_xs.extend(cross_sections)

    configure_style()

    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    colors = ["#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"]

    for color, (path, masses, cross_sections) in zip(colors, scans, strict=False):
        tanphi_token = infer_tanphi_token(path)
        label = rf"$\tan\phi = {format_tanphi_value(tanphi_token)}$"
        ax.plot(masses, cross_sections, color=color, linewidth=1.6, alpha=0.95, label=label)

    x_min = min(all_masses)
    x_max = max(all_masses)
    y_min = min(all_xs)
    y_max = max(all_xs)
    padding = 0.1 * (y_max - y_min) if y_max > y_min else 1.0

    ax.set_xlim(x_min, x_max)
    if not args.linear_y:
        ax.set_yscale("log")
        lower = args.ymin if args.ymin is not None else y_min
        upper = args.ymax if args.ymax is not None else y_max
        ax.set_ylim(lower, upper)
    else:
        lower = args.ymin if args.ymin is not None else y_min - padding
        upper = args.ymax if args.ymax is not None else y_max + padding
        ax.set_ylim(lower, upper)

    ax.set_xlabel(r"$m_a\ [\mathrm{GeV}]$")
    ax.set_ylabel(rf"$\sigma(pp \to {args.process_math})\ [\mathrm{{pb}}]$")
    ax.legend(loc="best", frameon=True)
    ax.grid(False)

    fig.tight_layout()

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
