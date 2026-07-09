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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D


DEFAULT_INPUT = Path(__file__).resolve().with_name("decays.csv")
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("decays.png")
DEFAULT_NARROW_WIDTH_INPUT = Path(__file__).resolve().with_name("narrow-width.csv")
NWA_THRESHOLD = 0.1

WIDTH_RATIO_CMAP = LinearSegmentedColormap.from_list(
    "width_ratio_threshold_map",
    [
        (0.0, "#1d4f8c"),
        (0.42, "#d6e3f3"),
        (0.5, "#f7f4ea"),
        (0.58, "#efc98e"),
        (1.0, "#e0b11f"),
    ],
)

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
    parser.add_argument(
        "--narrow-width-input",
        type=Path,
        default=DEFAULT_NARROW_WIDTH_INPUT,
        help="Input CSV file with narrow-width boundary curves. Default: decays/narrow-width.csv",
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
        (0.82, 0.23, 0.20),  # muted red
        (0.18, 0.47, 0.70),  # steel blue
        (0.20, 0.60, 0.32),  # soft green
        (0.90, 0.58, 0.16),  # warm orange
        (0.51, 0.36, 0.70),  # violet
        (0.55, 0.39, 0.26),  # brown
        (0.87, 0.49, 0.67),  # pink
        (0.45, 0.48, 0.52),  # slate gray
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
    narrow_width_data = pd.read_csv(args.narrow_width_input)

    required_columns = [
        "mass",
        "tanphi",
        "total width/mass",
        "primary decay channel",
    ]
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns in {args.input}: {', '.join(missing)}")
    narrow_width_required_columns = ["mass", "tanphi_low", "tanphi_high"]
    narrow_width_missing = [
        column for column in narrow_width_required_columns if column not in narrow_width_data.columns
    ]
    if narrow_width_missing:
        raise ValueError(
            "Missing required columns in "
            f"{args.narrow_width_input}: {', '.join(narrow_width_missing)}"
        )

    data = data.copy()
    data["mass"] = pd.to_numeric(data["mass"])
    data["tanphi"] = pd.to_numeric(data["tanphi"])
    data["total width/mass"] = pd.to_numeric(data["total width/mass"])
    narrow_width_data = narrow_width_data.copy()
    narrow_width_data["mass"] = pd.to_numeric(narrow_width_data["mass"])
    narrow_width_data["tanphi_low"] = pd.to_numeric(
        narrow_width_data["tanphi_low"], errors="coerce"
    )
    narrow_width_data["tanphi_high"] = pd.to_numeric(
        narrow_width_data["tanphi_high"], errors="coerce"
    )
    narrow_width_low = narrow_width_data.dropna(subset=["tanphi_low"]).sort_values("mass")
    narrow_width_high = narrow_width_data.dropna(subset=["tanphi_high"]).sort_values("mass")
    channel_codes, channel_mapping = build_channel_codes(data["primary decay channel"])
    cmap_channels = build_categorical_cmap(max(len(channel_mapping), 1))

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 16,
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

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.6), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, hspace=0.02, wspace=0.04)

    axes[0].scatter(
        data["mass"],
        data["tanphi"],
        c=channel_codes,
        cmap=cmap_channels,
        s=92,
        edgecolors="#242424",
        linewidths=0.45,
        alpha=0.96,
    )
    axes[0].set_title("Main Decay Channel", pad=10)
    axes[0].set_xlabel("Mass")
    axes[0].set_ylabel(r"$\tan\phi$")
    axes[0].set_yscale("log")

    legend_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=channel,
            markerfacecolor=cmap_channels(index),
            markeredgecolor="#242424",
            markersize=10,
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
            frameon=True,
            fancybox=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor="#d0d0d0",
        )
    if not narrow_width_low.empty:
        axes[0].plot(
            narrow_width_low["mass"],
            narrow_width_low["tanphi_low"],
            color="#111111",
            linewidth=2.0,
            linestyle="--",
            alpha=0.95,
            zorder=4,
        )
    if not narrow_width_high.empty:
        axes[0].plot(
            narrow_width_high["mass"],
            narrow_width_high["tanphi_high"],
            color="#111111",
            linewidth=2.0,
            linestyle="-.",
            alpha=0.95,
            zorder=4,
        )

    positive_width_ratio = data["total width/mass"][data["total width/mass"] > 0]
    if positive_width_ratio.empty:
        raise ValueError("All width/mass values are non-positive; cannot use log color scale.")
    width_ratio_min = float(positive_width_ratio.min())
    width_ratio_max = float(positive_width_ratio.max())
    if not (width_ratio_min < NWA_THRESHOLD < width_ratio_max):
        raise ValueError("The NWA threshold must lie inside the positive width/mass range.")

    log_width_ratio = data["total width/mass"].clip(lower=width_ratio_min).map(math.log10)
    width_ratio_norm = TwoSlopeNorm(
        vmin=math.log10(width_ratio_min),
        vcenter=math.log10(NWA_THRESHOLD),
        vmax=math.log10(width_ratio_max),
    )
    scatter_width_ratio = axes[1].scatter(
        data["mass"],
        data["tanphi"],
        c=log_width_ratio,
        cmap=WIDTH_RATIO_CMAP,
        norm=width_ratio_norm,
        s=92,
        marker="o",
        edgecolors="#242424",
        linewidths=0.45,
        alpha=0.96,
    )
    axes[1].set_title("Width / Mass", pad=10)
    axes[1].set_xlabel("Mass")
    axes[1].set_ylabel(r"$\tan\phi$")
    axes[1].set_yscale("log")
    if not narrow_width_low.empty:
        axes[1].plot(
            narrow_width_low["mass"],
            narrow_width_low["tanphi_low"],
            color="#111111",
            linewidth=2.0,
            linestyle="--",
            alpha=0.95,
            zorder=4,
        )
    if not narrow_width_high.empty:
        axes[1].plot(
            narrow_width_high["mass"],
            narrow_width_high["tanphi_high"],
            color="#111111",
            linewidth=2.0,
            linestyle="-.",
            alpha=0.95,
            zorder=4,
        )

    for ax in axes:
        ax.grid(True, which="major", color="#d9d9d9", linestyle=":", linewidth=0.8, alpha=0.75)
        ax.grid(True, which="minor", color="#ebebeb", linestyle=":", linewidth=0.5, alpha=0.55)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    colorbar = fig.colorbar(
        scatter_width_ratio,
        ax=axes[1],
        label="width / mass",
        pad=0.03,
        fraction=0.06,
    )
    colorbar_ticks = [
        value
        for value in [width_ratio_min, 1e-4, 1e-3, 1e-2, NWA_THRESHOLD, 1.0, 10.0, 100.0, width_ratio_max]
        if width_ratio_min <= value <= width_ratio_max
    ]
    colorbar.set_ticks([math.log10(value) for value in colorbar_ticks])
    colorbar.set_ticklabels([f"{value:g}" for value in colorbar_ticks])
    threshold_position = math.log10(NWA_THRESHOLD)
    colorbar.ax.axhline(threshold_position, color="black", linewidth=1.4, linestyle="--")
    colorbar.ax.text(
        0.5,
        threshold_position,
        "NWA = 0.1",
        transform=colorbar.ax.get_yaxis_transform(),
        va="center",
        ha="center",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#cfcfcf",
            "alpha": 0.96,
        },
    )

    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"Wrote plot to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
