#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

MPLCONFIGDIR = Path(__file__).resolve().parent / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt


DEFAULT_INPUT = Path(__file__).resolve().with_name("decays.csv")
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("fixed-mass-branchings.png")
DEFAULT_MASSES = [200.0, 314.2857, 330.6122, 350.0, 700.0, 1000.0]

CHANNEL_STYLES = {
    r"$A \to t\bar{t}$": {"color": "#2ca02c", "linestyle": "-", "linewidth": 2.4},
    r"$A \to c\bar{t} + t\bar{c}$": {"color": "#1f77b4", "linestyle": "-", "linewidth": 2.4},
    r"$A \to W^- \bar{b}\, t + \mathrm{c.c.}$": {"color": "#d62728", "linestyle": "--", "linewidth": 2.2},
    r"$A \to c\bar{c}$": {"color": "#9467bd", "linestyle": ":", "linewidth": 2.2},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot fixed-mass branching ratios from decays.csv."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input CSV file. Default: decays/decays.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PNG file. Default: decays/fixed-mass-branchings.png",
    )
    parser.add_argument(
        "--masses",
        type=float,
        nargs="+",
        default=DEFAULT_MASSES,
        help="Target masses to plot. Nearest masses in the scan are used.",
    )
    return parser.parse_args()


def load_rows(input_path: Path) -> list[dict[str, float | str]]:
    with input_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "mass",
            "tanphi",
            "primary decay channel",
            "primary decay branching",
            "second decay channel",
            "second decay branching",
            "third decay channel",
            "third decay branching",
            "fourth decay channel",
            "fourth decay branching",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {input_path}: {', '.join(sorted(missing))}")

        rows: list[dict[str, float | str]] = []
        for row in reader:
            parsed: dict[str, float | str] = {}
            for key, value in row.items():
                value = value.strip()
                if key in {"mass", "tanphi"}:
                    parsed[key] = float(value)
                elif key.endswith("branching") and value:
                    parsed[key] = float(value)
                else:
                    parsed[key] = value
            rows.append(parsed)
    if not rows:
        raise ValueError(f"No rows found in {input_path}")
    return rows


def branching_map(row: dict[str, float | str]) -> dict[str, float]:
    mapping: dict[str, float] = {}
    for ordinal in ["primary", "second", "third", "fourth"]:
        channel_key = f"{ordinal} decay channel"
        branching_key = f"{ordinal} decay branching"
        channel = str(row.get(channel_key, "")).strip()
        branching = row.get(branching_key, "")
        if channel and branching != "":
            mapping[channel] = float(branching)
    return mapping


def collect_available_masses(rows: list[dict[str, float | str]]) -> list[float]:
    return sorted({float(row["mass"]) for row in rows})


def nearest_mass(target: float, available: list[float]) -> float:
    return min(available, key=lambda value: abs(value - target))


def build_series(rows: list[dict[str, float | str]], mass_value: float) -> tuple[list[float], dict[str, list[float]]]:
    selected = sorted(
        (row for row in rows if math.isclose(float(row["mass"]), mass_value, rel_tol=0.0, abs_tol=1e-9)),
        key=lambda row: float(row["tanphi"]),
    )
    tanphis = [float(row["tanphi"]) for row in selected]
    series = {
        r"$A \to t\bar{t}$": [],
        r"$A \to c\bar{t} + t\bar{c}$": [],
        r"$A \to W^- \bar{b}\, t + \mathrm{c.c.}$": [],
        r"$A \to c\bar{c}$": [],
    }

    for row in selected:
        channels = branching_map(row)
        series[r"$A \to t\bar{t}$"].append(channels.get("t t~", 0.0))
        series[r"$A \to c\bar{t} + t\bar{c}$"].append(
            channels.get("c t~", 0.0) + channels.get("t c~", 0.0)
        )
        series[r"$A \to W^- \bar{b}\, t + \mathrm{c.c.}$"].append(
            channels.get("W- b~ t", 0.0) + channels.get("t~ b W+", 0.0)
        )
        series[r"$A \to c\bar{c}$"].append(channels.get("c c~", 0.0))

    return tanphis, series


def main() -> int:
    args = parse_args()
    rows = load_rows(args.input)
    available_masses = collect_available_masses(rows)
    selected_masses: list[float] = []
    for target in args.masses:
        mass = nearest_mass(target, available_masses)
        if mass not in selected_masses:
            selected_masses.append(mass)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "semibold",
            "axes.labelsize": 12,
            "axes.facecolor": "#fbfbf8",
            "figure.facecolor": "white",
            "axes.edgecolor": "#555555",
            "axes.linewidth": 0.9,
            "xtick.color": "#2f2f2f",
            "ytick.color": "#2f2f2f",
        }
    )

    nplots = len(selected_masses)
    ncols = 2
    nrows = math.ceil(nplots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 4.4 * nrows), constrained_layout=True)
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    legend_handles = []
    legend_labels = []

    for axis, mass_value in zip(axes_list, selected_masses):
        tanphis, series = build_series(rows, mass_value)
        for label, values in series.items():
            style = CHANNEL_STYLES[label]
            line, = axis.plot(tanphis, values, label=label, **style)
            if label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(label)

        axis.set_xscale("log")
        axis.set_ylim(0.0, 1.05)
        axis.set_xlabel(r"$\tan\phi$")
        axis.set_ylabel("Branching ratio")
        axis.set_title(rf"$m_A = {mass_value:.4g}\,\mathrm{{GeV}}$")
        axis.grid(True, which="major", color="#d9d9d9", linestyle=":", linewidth=0.8, alpha=0.75)
        axis.grid(True, which="minor", color="#ebebeb", linestyle=":", linewidth=0.5, alpha=0.55)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    for axis in axes_list[nplots:]:
        axis.set_visible(False)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#d0d0d0",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot to {args.output.resolve()}")
    print("Used masses:", ", ".join(f"{mass:.6g}" for mass in selected_masses))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
