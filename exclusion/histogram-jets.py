#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import awkward as ak
import numpy as np
import vector

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_DIR = SCRIPT_DIR / "generated-signal"
DEFAULT_BACKGROUND_DIR = SCRIPT_DIR / "generated-background"
DEFAULT_BACKGROUND_XSEC_CSV = DEFAULT_BACKGROUND_DIR / "xsec-background.csv"
DEFAULT_LUMINOSITIES_FB = (450.0, 3000.0)
GENERATED_DIR_PATTERN = re.compile(r"^generated-m-.+-tanphi-.+$")
PROCESS_FILE_PATTERN = "proc-*.parquet"
BACKGROUND_PROCESS_FILE_PATTERN = "proc-*-background.parquet"
PROCESS_COLORS = [
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#e15759",
    "#b07aa1",
    "#76b7b2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build weighted jet histograms for every generated signal point."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Directory containing generated-m-*-tanphi-* folders.",
    )
    parser.add_argument(
        "--luminosities",
        type=float,
        nargs="+",
        default=list(DEFAULT_LUMINOSITIES_FB),
        help="Integrated luminosities in fb^-1 used to build the output plots.",
    )
    parser.add_argument(
        "--background-dir",
        type=Path,
        default=DEFAULT_BACKGROUND_DIR,
        help="Directory containing the generated background parquets.",
    )
    parser.add_argument(
        "--background-xsec-csv",
        type=Path,
        default=DEFAULT_BACKGROUND_XSEC_CSV,
        help="CSV file containing the background cross sections and generated events.",
    )
    parser.add_argument(
        "--yscale",
        choices=("linear", "log"),
        default="linear",
        help="Y-axis scale for the histograms.",
    )
    parser.add_argument(
        "--background",
        choices=("yes", "no"),
        default="no",
        help="Overlay the total background as a dashed outline.",
    )
    return parser.parse_args()


def discover_generated_dirs(base_dir: Path) -> list[Path]:
    generated_dirs = [
        path.resolve()
        for path in sorted(base_dir.iterdir())
        if path.is_dir() and GENERATED_DIR_PATTERN.fullmatch(path.name)
    ]
    if not generated_dirs:
        raise FileNotFoundError(f"No generated-m-*-tanphi-* directories found in {base_dir}")
    return generated_dirs


def discover_signal_parquets(parquet_dir: Path) -> list[tuple[int, Path]]:
    parquet_entries: list[tuple[int, Path]] = []
    for parquet_path in sorted(parquet_dir.glob(PROCESS_FILE_PATTERN)):
        stem_parts = parquet_path.stem.split("-")
        if len(stem_parts) < 3 or stem_parts[0] != "proc":
            continue
        try:
            process_number = int(stem_parts[1])
        except ValueError:
            continue
        parquet_entries.append((process_number, parquet_path.resolve()))

    if not parquet_entries:
        raise FileNotFoundError(f"No signal parquet files found in {parquet_dir}")

    return parquet_entries


def discover_background_parquets(parquet_dir: Path) -> list[tuple[int, Path]]:
    parquet_entries: list[tuple[int, Path]] = []
    for parquet_path in sorted(parquet_dir.glob(BACKGROUND_PROCESS_FILE_PATTERN)):
        stem_parts = parquet_path.stem.split("-")
        if len(stem_parts) < 3 or stem_parts[0] != "proc":
            continue
        try:
            process_number = int(stem_parts[1])
        except ValueError:
            continue
        parquet_entries.append((process_number, parquet_path.resolve()))

    if not parquet_entries:
        raise FileNotFoundError(f"No background parquet files found in {parquet_dir}")

    return parquet_entries


def find_xsec_csv(generated_dir: Path) -> Path:
    candidates = sorted(generated_dir.glob("xsec-*.csv"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one xsec CSV in {generated_dir}")
    return candidates[0].resolve()


def load_xsec_rows(xsec_csv_path: Path) -> dict[int, dict[str, float | str]]:
    with xsec_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        xsecs_by_process = {
            int(row["process"]): {
                "cross_section_pb": float(row["cross section pb"]),
                "generated_events": float(row["N events (generated)"]),
                "description": row["description"].strip(),
            }
            for row in reader
        }

    if not xsecs_by_process:
        raise ValueError(f"No cross-section rows found in {xsec_csv_path}")

    return xsecs_by_process


def build_lepton_vectors(leptons: ak.Array) -> ak.Array:
    return ak.zip(
        {
            "pt": leptons.PT,
            "eta": leptons.Eta,
            "phi": leptons.Phi,
            "mass": leptons.Mass,
            "charge": leptons.Charge,
        },
        with_name="Momentum4D",
    )


def build_cut_masks(events: ak.Array) -> dict[str, ak.Array]:
    jets = events.Jet
    bjets_passing_cut_i = (jets.BTag > 0) & (jets.PT > 20.0) & (abs(jets.Eta) < 2.5)
    cut_i = ak.any(bjets_passing_cut_i, axis=1)

    cut_ii = cut_i & (events.met > 20.0)

    leptons = build_lepton_vectors(events.Lepton)
    leptons_passing_kinematics = leptons[(leptons.pt > 10.0) & (abs(leptons.eta) < 2.5)]
    exactly_two_selected_leptons = ak.num(leptons_passing_kinematics, axis=1) == 2

    padded_leptons = ak.pad_none(leptons_passing_kinematics, 2, axis=1, clip=True)
    first_lepton = padded_leptons[:, 0]
    second_lepton = padded_leptons[:, 1]
    same_sign_leptons = (
        exactly_two_selected_leptons
        & ak.fill_none(first_lepton.charge * second_lepton.charge > 0, False)
    )
    cut_iii = cut_ii & same_sign_leptons

    dilepton_mass = ak.fill_none((first_lepton + second_lepton).mass, 0.0)
    cut_iv = cut_iii & (dilepton_mass > 100.0)

    return {
        "cut_i": cut_i,
        "cut_ii": cut_ii,
        "cut_iii": cut_iii,
        "cut_iv": cut_iv,
    }


def build_process_label(process_number: int, description: str) -> str:
    return f"proc {process_number}"


def load_weighted_event_counts(
    parquet_entries: list[tuple[int, Path]],
    xsecs_by_process: dict[int, dict[str, float | str]],
    luminosity_fb: float,
) -> dict[str, list[dict[str, object]]]:
    counts_by_variant: dict[str, list[dict[str, object]]] = {
        "inclusive": [],
        "after-cuts": [],
    }

    for process_number, parquet_path in parquet_entries:
        if process_number not in xsecs_by_process:
            raise ValueError(f"Missing cross-section information for process {process_number}")

        xsec_info = xsecs_by_process[process_number]
        generated_events = float(xsec_info["generated_events"])
        if generated_events <= 0:
            raise ValueError(
                f"Generated event count must be positive for process {process_number}"
            )

        events = ak.from_parquet(parquet_path)
        n_jets = ak.to_numpy(events["n_jets"]).astype(int)
        n_bjets = ak.to_numpy(events["n_bjets"]).astype(int)
        final_cut_mask = ak.to_numpy(build_cut_masks(events)["cut_iv"]).astype(bool)
        event_weight = float(xsec_info["cross_section_pb"]) / generated_events * luminosity_fb * 1000.0
        weights = np.full(n_jets.shape, event_weight, dtype=float)

        base_entry = {
            "process_number": process_number,
            "label": build_process_label(process_number, str(xsec_info["description"])),
        }
        counts_by_variant["inclusive"].append(
            {
                **base_entry,
                "jet_counts": n_jets,
                "bjet_counts": n_bjets,
                "weights": weights,
            }
        )
        counts_by_variant["after-cuts"].append(
            {
                **base_entry,
                "jet_counts": n_jets[final_cut_mask],
                "bjet_counts": n_bjets[final_cut_mask],
                "weights": weights[final_cut_mask],
            }
        )

    return counts_by_variant


def load_total_background_counts(
    parquet_entries: list[tuple[int, Path]],
    xsecs_by_process: dict[int, dict[str, float | str]],
    luminosity_fb: float,
) -> dict[str, dict[str, np.ndarray]]:
    jet_counts_by_variant: dict[str, list[np.ndarray]] = {"inclusive": [], "after-cuts": []}
    bjet_counts_by_variant: dict[str, list[np.ndarray]] = {"inclusive": [], "after-cuts": []}
    weights_by_variant: dict[str, list[np.ndarray]] = {"inclusive": [], "after-cuts": []}

    for process_number, parquet_path in parquet_entries:
        if process_number not in xsecs_by_process:
            raise ValueError(f"Missing background cross-section information for process {process_number}")

        xsec_info = xsecs_by_process[process_number]
        generated_events = float(xsec_info["generated_events"])
        if generated_events <= 0:
            raise ValueError(
                f"Generated event count must be positive for background process {process_number}"
            )

        events = ak.from_parquet(parquet_path)
        n_jets = ak.to_numpy(events["n_jets"]).astype(int)
        n_bjets = ak.to_numpy(events["n_bjets"]).astype(int)
        final_cut_mask = ak.to_numpy(build_cut_masks(events)["cut_iv"]).astype(bool)
        event_weight = float(xsec_info["cross_section_pb"]) / generated_events * luminosity_fb * 1000.0
        weights = np.full(n_jets.shape, event_weight, dtype=float)

        jet_counts_by_variant["inclusive"].append(n_jets)
        bjet_counts_by_variant["inclusive"].append(n_bjets)
        weights_by_variant["inclusive"].append(weights)
        jet_counts_by_variant["after-cuts"].append(n_jets[final_cut_mask])
        bjet_counts_by_variant["after-cuts"].append(n_bjets[final_cut_mask])
        weights_by_variant["after-cuts"].append(weights[final_cut_mask])

    background_counts: dict[str, dict[str, np.ndarray]] = {}
    for variant in ("inclusive", "after-cuts"):
        background_counts[variant] = {
            "jet_counts": np.concatenate(jet_counts_by_variant[variant]),
            "bjet_counts": np.concatenate(bjet_counts_by_variant[variant]),
            "weights": np.concatenate(weights_by_variant[variant]),
        }

    return background_counts


def build_integer_bins(max_value: int) -> np.ndarray:
    return np.arange(-0.5, max_value + 1.5, 1.0)


def make_plot(
    process_entries: list[dict[str, object]],
    background_counts: dict[str, np.ndarray] | None,
    luminosity_fb: float,
    output_path: Path,
    sample_label: str,
    point_label: str,
    yscale: str,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    all_jet_counts = [np.asarray(entry["jet_counts"], dtype=int) for entry in process_entries]
    all_bjet_counts = [np.asarray(entry["bjet_counts"], dtype=int) for entry in process_entries]
    background_jet_counts = (
        np.asarray(background_counts["jet_counts"], dtype=int)
        if background_counts is not None
        else np.array([], dtype=int)
    )
    background_bjet_counts = (
        np.asarray(background_counts["bjet_counts"], dtype=int)
        if background_counts is not None
        else np.array([], dtype=int)
    )
    max_jet_count = max(
        [int(np.max(values)) if values.size else 0 for values in all_jet_counts]
        + [int(np.max(background_jet_counts)) if background_jet_counts.size else 0]
    )
    max_bjet_count = max(
        [int(np.max(values)) if values.size else 0 for values in all_bjet_counts]
        + [int(np.max(background_bjet_counts)) if background_bjet_counts.size else 0]
    )
    plot_specs = (
        (
            axes[0],
            "jet_counts",
            background_jet_counts,
            max_jet_count,
            "Jet multiplicity",
            "Number of jets",
        ),
        (
            axes[1],
            "bjet_counts",
            background_bjet_counts,
            max_bjet_count,
            "Positive b-tag multiplicity",
            "Number of b-tagged jets",
        ),
    )

    background_weights = (
        np.asarray(background_counts["weights"], dtype=float)
        if background_counts is not None
        else np.array([], dtype=float)
    )

    for axis, field_name, background_values, max_value, title, xlabel in plot_specs:
        values_by_process = [np.asarray(entry[field_name], dtype=int) for entry in process_entries]
        weights_by_process = [np.asarray(entry["weights"], dtype=float) for entry in process_entries]
        colors = [
            PROCESS_COLORS[index % len(PROCESS_COLORS)]
            for index, _ in enumerate(process_entries)
        ]
        labels = [str(entry["label"]) for entry in process_entries]
        bins = build_integer_bins(max_value)
        axis.hist(
            values_by_process,
            bins=bins,
            weights=weights_by_process,
            stacked=True,
            color=colors,
            label=labels,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.88,
        )
        if background_counts is not None:
            axis.hist(
                background_values,
                bins=bins,
                weights=background_weights,
                histtype="step",
                linestyle="--",
                color="black",
                linewidth=1.8,
                label="background total",
            )
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Expected signal events")
        axis.set_xticks(np.arange(0, max_value + 1, 1))
        axis.set_yscale(yscale)
        axis.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.55)
        axis.legend(fontsize=8)

    total_expected_events = float(
        sum(np.sum(np.asarray(entry["weights"], dtype=float)) for entry in process_entries)
    )
    fig.suptitle(
        rf"Signal {point_label} {sample_label} at {luminosity_fb:g} fb$^{{-1}}$"
        + f"\nTotal expected events: {total_expected_events:.2f}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.resolve()


def build_output_path(histogram_dir: Path, luminosity_fb: float, variant: str) -> Path:
    if variant == "inclusive":
        return histogram_dir / f"jets-histogram-lumi-{luminosity_fb:g}.png"
    return histogram_dir / f"jets-histogram-{variant}-lumi-{luminosity_fb:g}.png"


def process_generated_dir(
    generated_dir: Path,
    luminosities: list[float],
    background_parquet_entries: list[tuple[int, Path]] | None,
    background_xsecs_by_process: dict[int, dict[str, float | str]] | None,
    yscale: str,
) -> list[Path]:
    parquet_entries = discover_signal_parquets(generated_dir / "parquets")
    xsecs_by_process = load_xsec_rows(find_xsec_csv(generated_dir))
    histogram_dir = generated_dir / "histogram"
    output_paths: list[Path] = []

    for luminosity_fb in luminosities:
        background_counts_by_variant = (
            load_total_background_counts(
                background_parquet_entries,
                background_xsecs_by_process,
                luminosity_fb,
            )
            if background_parquet_entries is not None and background_xsecs_by_process is not None
            else None
        )
        counts_by_variant = load_weighted_event_counts(
            parquet_entries,
            xsecs_by_process,
            luminosity_fb,
        )
        for variant, process_entries in counts_by_variant.items():
            output_paths.append(
                make_plot(
                    process_entries,
                    background_counts_by_variant[variant] if background_counts_by_variant else None,
                    luminosity_fb,
                    build_output_path(histogram_dir, luminosity_fb, variant),
                    "after cuts" if variant == "after-cuts" else "inclusive",
                    generated_dir.name.removeprefix("generated-"),
                    yscale,
                )
            )

    return output_paths


def main() -> int:
    args = parse_args()
    vector.register_awkward()
    background_parquet_entries = None
    background_xsecs_by_process = None
    if args.background == "yes":
        background_parquet_entries = discover_background_parquets(args.background_dir / "parquets")
        background_xsecs_by_process = load_xsec_rows(args.background_xsec_csv)

    for generated_dir in discover_generated_dirs(args.base_dir):
        for output_path in process_generated_dir(
            generated_dir,
            args.luminosities,
            background_parquet_entries,
            background_xsecs_by_process,
            args.yscale,
        ):
            print(f"Wrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
