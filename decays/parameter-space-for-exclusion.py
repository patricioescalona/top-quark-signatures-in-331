#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

MPLCONFIGDIR = Path(__file__).resolve().parent / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D


DEFAULT_INPUT = Path(__file__).resolve().with_name("decays.csv")
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("parameter-space-for-exclusion.txt")
DEFAULT_NEVENTS = 10000
TOP_PAIR_CHANNEL = "t t~"
CHARGE_CONJUGATE_LABELS = {
    "-24": "24",
    "24": "-24",
    "d": "d~",
    "d~": "d",
    "u": "u~",
    "u~": "u",
    "s": "s~",
    "s~": "s",
    "c": "c~",
    "c~": "c",
    "b": "b~",
    "b~": "b",
    "t": "t~",
    "t~": "t",
    "e-": "e+",
    "e+": "e-",
    "mu-": "mu+",
    "mu+": "mu-",
    "tau-": "tau+",
    "tau+": "tau-",
    "ve": "ve~",
    "ve~": "ve",
    "vm": "vm~",
    "vm~": "vm",
    "vt": "vt~",
    "vt~": "vt",
    "W+": "W-",
    "W-": "W+",
    "Z": "Z",
    "H": "H",
    "AP": "AP",
    "gamma": "gamma",
    "g": "g",
}


@dataclass(frozen=True)
class Point:
    mass_label: str
    tanphi_label: str
    mass_value: float
    tanphi_value: float


@dataclass(frozen=True)
class ScanRow:
    mass_value: float
    tanphi_value: float
    width_ratio: float
    primary_channel: str
    selected: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a list of generator commands for exclusion points that pass "
            "the narrow-width approximation and whose primary decay channel is not t t~."
        )
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
        help="Output text file. Default: decays/parameter-space-for-exclusion.txt",
    )
    parser.add_argument(
        "--nevents",
        type=int,
        default=DEFAULT_NEVENTS,
        help="Number of events to pass to generator-and-compressor.py. Default: 10000",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help=(
            "Output PNG file for the parameter-space plot. "
            "Default: same path as --output, but with .png extension"
        ),
    )
    args = parser.parse_args()

    if args.nevents <= 0:
        parser.error("--nevents must be a positive integer.")
    if args.input.suffix.lower() != ".csv":
        parser.error("--input must point to a .csv file.")
    if args.output.suffix.lower() != ".txt":
        parser.error("--output must point to a .txt file.")
    if args.plot_output is None:
        args.plot_output = args.output.with_suffix(".png")
    if args.plot_output.suffix.lower() != ".png":
        parser.error("--plot-output must point to a .png file.")

    return args


def normalize_label(raw_value: str) -> str:
    return f"{float(raw_value):.12g}"


def load_points(rows: list[dict[str, str]]) -> list[Point]:
    points: list[Point] = []
    for row in rows:
        if not passes_filters(row):
            continue
        points.append(
            Point(
                mass_label=normalize_label(row["mass"]),
                tanphi_label=normalize_label(row["tanphi"]),
                mass_value=float(row["mass"]),
                tanphi_value=float(row["tanphi"]),
            )
        )

    points.sort(key=lambda point: (point.mass_value, point.tanphi_value))
    return points


def load_rows(input_path: Path) -> list[dict[str, str]]:
    with input_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required_fields = {
            "mass",
            "tanphi",
            "total width/mass",
            "primary decay channel",
        }
        missing_fields = required_fields.difference(reader.fieldnames or [])
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Missing required CSV columns: {missing}")
        return list(reader)


def passes_filters(row: dict[str, str]) -> bool:
    width_ratio = float(row["total width/mass"])
    if width_ratio > 0.1:
        return False

    primary_channel = row["primary decay channel"].strip()
    if primary_channel == TOP_PAIR_CHANNEL:
        return False

    return True


def group_points_by_mass(points: list[Point]) -> dict[str, list[Point]]:
    grouped: dict[str, list[Point]] = {}
    for point in points:
        grouped.setdefault(point.mass_label, []).append(point)
    return grouped


def split_log_uniform_runs(points: list[Point]) -> list[list[Point]]:
    if len(points) < 2:
        return [points]

    runs: list[list[Point]] = []
    current_run = [points[0]]
    current_step: float | None = None

    for previous, current in zip(points, points[1:]):
        log_gap = math.log10(current.tanphi_value) - math.log10(previous.tanphi_value)
        if current_step is None:
            current_run.append(current)
            current_step = log_gap
            continue

        if math.isclose(log_gap, current_step, rel_tol=1e-5, abs_tol=1e-5):
            current_run.append(current)
            continue

        runs.append(current_run)
        current_run = [current]
        current_step = None

    runs.append(current_run)
    return runs


def build_command(nevents: int, mass_label: str, points: list[Point]) -> str:
    prefix = (
        f"python3 exclusion/generator-and-compressor.py {nevents} "
        f"--mass {mass_label}"
    )
    if len(points) == 1:
        return f"{prefix} --tanphi {points[0].tanphi_label}"

    return (
        f"{prefix} --tanphi-range {points[0].tanphi_label} "
        f"{points[-1].tanphi_label} --tanphi-points {len(points)}"
    )


def build_output_lines(points: list[Point], nevents: int) -> list[str]:
    grouped = group_points_by_mass(points)
    lines: list[str] = []

    for mass_label in sorted(grouped, key=float):
        runs = split_log_uniform_runs(grouped[mass_label])
        for run in runs:
            if not run:
                continue
            lines.append(build_command(nevents, mass_label, run))

    return lines


def write_output(output_path: Path, lines: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    if content:
        content += "\n"
    output_path.write_text(content)


def build_scan_rows(rows: list[dict[str, str]]) -> list[ScanRow]:
    scan_rows = [
        ScanRow(
            mass_value=float(row["mass"]),
            tanphi_value=float(row["tanphi"]),
            width_ratio=float(row["total width/mass"]),
            primary_channel=row["primary decay channel"].strip(),
            selected=passes_filters(row),
        )
        for row in rows
    ]
    scan_rows.sort(key=lambda row: (row.mass_value, row.tanphi_value))
    return scan_rows


def charge_conjugate_channel(channel: str) -> str:
    labels = channel.split()
    conjugated = [CHARGE_CONJUGATE_LABELS.get(label, label) for label in labels]
    return " ".join(reversed(conjugated))


def group_channel_label(channel: str) -> str:
    conjugate = charge_conjugate_channel(channel)
    first = min(channel, conjugate)
    second = max(channel, conjugate)
    if first == second:
        return first
    return f"{first} / {second}"


def build_channel_codes(scan_rows: list[ScanRow]) -> tuple[list[int], dict[str, int]]:
    unique_channels = sorted(
        {group_channel_label(row.primary_channel) for row in scan_rows if row.primary_channel}
    )
    mapping = {channel: index for index, channel in enumerate(unique_channels)}
    codes = [mapping[group_channel_label(row.primary_channel)] for row in scan_rows]
    return codes, mapping


def build_categorical_cmap(size: int) -> ListedColormap:
    base = [
        (0.89, 0.10, 0.11),
        (0.12, 0.47, 0.71),
        (0.20, 0.63, 0.17),
        (1.00, 0.50, 0.00),
        (0.60, 0.31, 0.64),
        (0.65, 0.34, 0.16),
        (0.97, 0.51, 0.75),
        (0.50, 0.50, 0.50),
    ]
    if size <= len(base):
        return ListedColormap(base[:size])

    colors: list[tuple[float, float, float]] = []
    while len(colors) < size:
        colors.extend(base)
    return ListedColormap(colors[:size])


def plot_parameter_space(scan_rows: list[ScanRow], output_path: Path) -> None:
    if not scan_rows:
        raise ValueError("No rows available to plot.")

    selected_rows = [row for row in scan_rows if row.selected]
    if not selected_rows:
        raise ValueError("No selected rows available to plot.")

    masses = [row.mass_value for row in selected_rows]
    tanphis = [row.tanphi_value for row in selected_rows]
    channel_codes, channel_mapping = build_channel_codes(selected_rows)
    cmap_channels = build_categorical_cmap(max(len(channel_mapping), 1))

    fig, ax = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)
    ax.scatter(
        masses,
        tanphis,
        c=channel_codes,
        cmap=cmap_channels,
        s=80,
        edgecolors="black",
        linewidths=0.3,
    )
    ax.set_title("Primary Decay Channel")
    ax.set_xlabel("mass")
    ax.set_ylabel("tanphi")
    ax.set_yscale("log")

    legend_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=channel,
            markerfacecolor=cmap_channels(index),
            markeredgecolor="black",
            markersize=8,
        )
        for channel, index in channel_mapping.items()
    ]
    ax.legend(
        handles=legend_items,
        title="Channel",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    points = load_points(rows)
    scan_rows = build_scan_rows(rows)
    lines = build_output_lines(points, args.nevents)
    write_output(args.output, lines)
    plot_parameter_space(scan_rows, args.plot_output)
    print(f"Wrote {len(lines)} command(s) to {args.output}")
    print(f"Wrote plot to {args.plot_output}")


if __name__ == "__main__":
    main()
