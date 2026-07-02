#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

MPLCONFIGDIR = Path(__file__).resolve().parent / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext
from matplotlib.lines import Line2D


DEFAULT_INPUT = Path(__file__).resolve().with_name("decays.csv")
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("decays.png")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot decay scan results from decays.csv."
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
        help="Output PNG file. Default: decays/decays.png",
    )
    return parser.parse_args()


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


def build_channel_codes(channels: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    grouped_channels = channels.map(group_channel_label)
    unique_channels = sorted(grouped_channels.dropna().unique())
    mapping = {channel: index for index, channel in enumerate(unique_channels)}
    return grouped_channels.map(mapping), mapping


def build_categorical_cmap(size: int) -> ListedColormap:
    base = [
        (0.89, 0.10, 0.11),  # red
        (0.12, 0.47, 0.71),  # blue
        (0.20, 0.63, 0.17),  # green
        (1.00, 0.50, 0.00),  # orange
        (0.60, 0.31, 0.64),  # purple
        (0.65, 0.34, 0.16),  # brown
        (0.97, 0.51, 0.75),  # pink
        (0.50, 0.50, 0.50),  # gray
    ]
    if size <= len(base):
        return ListedColormap(base[:size])

    colors: list[tuple[float, float, float]] = []
    while len(colors) < size:
        colors.extend(base)
    return ListedColormap(colors[:size])


def main() -> int:
    args = parse_args()
    data = pd.read_csv(args.input)
    if data.empty:
        raise ValueError(f"No rows found in {args.input}")

    required_columns = [
        "mass",
        "tanphi",
        "total width/mass",
        "primary decay channel",
    ]
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns in {args.input}: {', '.join(missing)}")

    data = data.copy()
    data["mass"] = pd.to_numeric(data["mass"])
    data["tanphi"] = pd.to_numeric(data["tanphi"])
    data["total width/mass"] = pd.to_numeric(data["total width/mass"])
    channel_codes, channel_mapping = build_channel_codes(data["primary decay channel"])
    cmap_channels = build_categorical_cmap(max(len(channel_mapping), 1))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    scatter_channels = axes[0].scatter(
        data["mass"],
        data["tanphi"],
        c=channel_codes,
        cmap=cmap_channels,
        s=80,
        edgecolors="black",
        linewidths=0.3,
    )
    axes[0].set_title("Main Decay Channel")
    axes[0].set_xlabel("mass")
    axes[0].set_ylabel("tanphi")
    axes[0].set_yscale("log")

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
    if legend_items:
        axes[0].legend(
            handles=legend_items,
            title="Channel",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )

    positive_width_ratio = data["total width/mass"][data["total width/mass"] > 0]
    if positive_width_ratio.empty:
        raise ValueError("All width/mass values are non-positive; cannot use log color scale.")
    tick_min_exp = math.floor(math.log10(positive_width_ratio.min()))
    tick_max_exp = math.ceil(math.log10(positive_width_ratio.max()))
    colorbar_ticks = [10**exponent for exponent in range(tick_min_exp, tick_max_exp + 1)]
    width_ratio_cmap = "cividis"
    width_ratio_norm = LogNorm(
        vmin=10**tick_min_exp,
        vmax=10**tick_max_exp,
    )
    low_width_mask = data["total width/mass"] < 0.1
    high_width_mask = ~low_width_mask

    axes[1].scatter(
        data.loc[high_width_mask, "mass"],
        data.loc[high_width_mask, "tanphi"],
        c=data.loc[high_width_mask, "total width/mass"],
        cmap=width_ratio_cmap,
        norm=width_ratio_norm,
        s=80,
        marker="o",
        edgecolors="black",
        linewidths=0.3,
    )
    axes[1].scatter(
        data.loc[low_width_mask, "mass"],
        data.loc[low_width_mask, "tanphi"],
        c=data.loc[low_width_mask, "total width/mass"],
        cmap=width_ratio_cmap,
        norm=width_ratio_norm,
        s=120,
        marker="*",
        edgecolors="black",
        linewidths=0.3,
    )
    axes[1].set_title("Width / Mass")
    axes[1].set_xlabel("mass")
    axes[1].set_ylabel("tanphi")
    axes[1].set_yscale("log")
    colorbar = fig.colorbar(
        cm.ScalarMappable(norm=width_ratio_norm, cmap=width_ratio_cmap),
        ax=axes[1],
        label="width / mass",
        ticks=colorbar_ticks,
    )
    colorbar.formatter = LogFormatterMathtext(base=10.0)
    colorbar.update_ticks()

    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"Wrote plot to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
