from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCAN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = SCAN_ROOT / "results"
DEFAULT_FIGURES = SCAN_ROOT / "figures"
DEFAULT_BSM_FILES = [
    DEFAULT_RESULTS / "bsm-ttbar-tanphi-001.csv",
    DEFAULT_RESULTS / "bsm-ttbar-tanphi-01.csv",
    DEFAULT_RESULTS / "bsm-ttbar-tanphi-1.csv",
    DEFAULT_RESULTS / "bsm-ttbar-tanphi-10.csv",
    DEFAULT_RESULTS / "bsm-ttbar-tanphi-100.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot SM and BSM ttbar cross sections on the same figure."
    )
    parser.add_argument(
        "--sm",
        type=Path,
        default=DEFAULT_RESULTS / "sm-ttbar.csv",
        help="CSV file with the SM ttbar cross section.",
    )
    parser.add_argument(
        "--bsm",
        type=Path,
        nargs="*",
        default=DEFAULT_BSM_FILES,
        help="CSV files with the BSM ttbar cross sections.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_FIGURES / "sm-bsm-bg-comparison.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use a logarithmic scale on the y axis.",
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


def tanphi_label_from_path(path: Path) -> str:
    token = path.stem.rsplit("tanphi-", 1)[-1]
    if token == "001":
        value = "0.01"
    elif token == "01":
        value = "0.1"
    else:
        value = token
    return rf"$\tan\phi = {value}$"


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

    sm_path = args.sm.expanduser().resolve()
    bsm_paths = [path.expanduser().resolve() for path in args.bsm]

    sm_mass, sm_xs = load_scan(sm_path)
    bsm_scans = [(path, *load_scan(path)) for path in bsm_paths]

    configure_style()

    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    bsm_colors = ["#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"]
    bsm_handles = []
    bsm_labels = []
    all_xs = list(sm_xs)

    for color, (path, masses, cross_sections) in zip(bsm_colors, bsm_scans):
        all_xs.extend(cross_sections)
        handle = ax.plot(
            masses,
            cross_sections,
            color=color,
            linewidth=1.6,
            alpha=0.95,
            zorder=2,
        )[0]
        bsm_handles.append(handle)
        bsm_labels.append(tanphi_label_from_path(path))

    sm_handle = ax.plot(
        sm_mass,
        sm_xs,
        color="black",
        linewidth=2.8,
        label="SM",
        zorder=3,
    )[0]

    x_min = min(sm_mass)
    x_max = max(sm_mass)
    y_min = min(all_xs)
    y_max = max(all_xs)
    padding = 0.1 * (y_max - y_min) if y_max > y_min else 1.0

    ax.set_xlim(x_min, x_max)
    if args.log_y:
        ax.set_yscale("log")
        lower = args.ymin if args.ymin is not None else y_min
        upper = args.ymax if args.ymax is not None else y_max
        ax.set_ylim(lower, upper)
    else:
        lower = args.ymin if args.ymin is not None else y_min - padding
        upper = args.ymax if args.ymax is not None else y_max + padding
        ax.set_ylim(lower, upper)
    ax.set_xticks([200, 500, 800, 1100, 1400])
    ax.set_xlabel(r"$m_a\ [\mathrm{GeV}]$")
    ax.set_ylabel(r"$\sigma(pp \to t\bar{t})\ [\mathrm{pb}]$")
    ax.legend(
        [sm_handle, *bsm_handles],
        ["SM", *bsm_labels],
        loc="best",
        frameon=True,
        title="BSM",
        title_fontsize=17,
    )
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
