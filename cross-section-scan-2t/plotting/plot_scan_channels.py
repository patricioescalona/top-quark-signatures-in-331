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
TANPHI_PATTERN = re.compile(r"tanphi-(.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot pp -> aa, pp -> at, and pp -> tt cross sections from scan CSV files."
    )
    parser.add_argument(
        "--aa",
        type=Path,
        default=DEFAULT_RESULTS / "varI-BM3-aa-200-1400.csv",
        help="CSV file for the aa channel.",
    )
    parser.add_argument(
        "--at",
        type=Path,
        default=DEFAULT_RESULTS / "varI-BM3-at-200-1400.csv",
        help="CSV file for the at channel.",
    )
    parser.add_argument(
        "--tt",
        type=Path,
        default=DEFAULT_RESULTS / "varI-BM3-tt-200-1400.csv",
        help="CSV file for the tt channel.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_FIGURES / "varI-BM3-channels.pdf",
        help="Output figure path. By default this goes to cross-section-scan-2t/figures/.",
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


def infer_tanphi_token(path: Path) -> str | None:
    match = TANPHI_PATTERN.search(path.stem)
    if match is None:
        return None

    token = match.group(1)
    if re.fullmatch(r"0\d+", token):
        token = f"0.{token[1:]}"

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


def infer_tanphi_label(paths: list[Path]) -> str | None:
    values = {
        format_tanphi_value(token)
        for path in paths
        if (token := infer_tanphi_token(path)) is not None
    }

    if not values:
        return None

    if len(values) != 1:
        joined_values = ", ".join(sorted(values))
        raise ValueError(
            "Could not infer a unique tanphi value from the input CSV names: "
            f"{joined_values}"
        )

    return rf"$\tan\phi = {values.pop()}$"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 20,
            "axes.labelsize": 32,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 22,
            "axes.linewidth": 0.9,
        }
    )


def main() -> int:
    args = parse_args()

    aa_path = args.aa.expanduser().resolve()
    at_path = args.at.expanduser().resolve()
    tt_path = args.tt.expanduser().resolve()

    aa_mass, aa_xs = load_scan(aa_path)
    at_mass, at_xs = load_scan(at_path)
    tt_mass, tt_xs = load_scan(tt_path)
    tanphi_label = infer_tanphi_label([aa_path, at_path, tt_path])

    configure_style()

    fig, ax = plt.subplots(figsize=(7.2, 6.0))

    ax.plot(aa_mass, aa_xs, color="#2a6fcb", linewidth=1.5, label=r"$pp \to aa$")
    ax.plot(
        at_mass,
        at_xs,
        color="#ff7f0e",
        linewidth=1.5,
        linestyle="--",
        label=r"$pp \to at$",
    )
    ax.plot(
        tt_mass,
        tt_xs,
        color="#00aa22",
        linewidth=1.5,
        linestyle="-.",
        label=r"$pp \to tt$",
    )

    ax.set_yscale("log")
    ax.set_xlim(200, 1400)
    ax.set_ylim(1e-10, 1e2)
    ax.set_xticks([200, 500, 800, 1100, 1400])
    ax.set_xlabel(r"$m_a\ [\mathrm{GeV}]$")
    ax.set_ylabel(r"$\sigma\ [\mathrm{pb}]$")
    ax.legend(loc="best", frameon=True, title=tanphi_label)
    ax.grid(False)

    fig.tight_layout()

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
