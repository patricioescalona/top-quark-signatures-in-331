#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import awkward as ak
import numpy as np
import vector

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DEFAULT_PARQUET_DIR = BASE_DIR / "parquets"
DEFAULT_XSEC_CSV = BASE_DIR / "xsec-background.csv"
DEFAULT_LUMINOSITIES_FB = (450.0, 3000.0)
PROCESS_FILE_GLOB = "proc-*-background.parquet"
PROCESS_COLORS = {
    1: "#4e79a7",
    2: "#f28e2b",
    3: "#59a14f",
    4: "#e15759",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build weighted jet-multiplicity histograms for the total background."
        )
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=DEFAULT_PARQUET_DIR,
        help="Directory containing proc-*-background.parquet files.",
    )
    parser.add_argument(
        "--xsec-csv",
        type=Path,
        default=DEFAULT_XSEC_CSV,
        help="CSV file containing the process cross sections and generated events.",
    )
    parser.add_argument(
        "--luminosities",
        type=float,
        nargs="+",
        default=list(DEFAULT_LUMINOSITIES_FB),
        help="Integrated luminosities in fb^-1 used to build the output plots.",
    )
    return parser.parse_args()


def discover_background_parquets(parquet_dir: Path) -> list[tuple[int, Path]]:
    parquet_entries: list[tuple[int, Path]] = []
    for parquet_path in sorted(parquet_dir.glob(PROCESS_FILE_GLOB)):
        stem_parts = parquet_path.stem.split("-")
        if len(stem_parts) < 3 or stem_parts[0] != "proc":
            continue

        try:
            process_number = int(stem_parts[1])
        except ValueError:
            continue

        parquet_entries.append((process_number, parquet_path.resolve()))

    if not parquet_entries:
        raise FileNotFoundError(
            f"No background parquet files matching {PROCESS_FILE_GLOB} found in {parquet_dir}"
        )

    return parquet_entries


def load_xsec_rows(xsec_csv_path: Path) -> dict[int, dict[str, float]]:
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


def load_weighted_event_counts(
    parquet_entries: list[tuple[int, Path]],
    xsecs_by_process: dict[int, dict[str, float]],
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
        generated_events = xsec_info["generated_events"]
        if generated_events <= 0:
            raise ValueError(
                f"Generated event count must be positive for process {process_number}"
            )

        events = ak.from_parquet(parquet_path)
        n_jets = ak.to_numpy(events["n_jets"]).astype(int)
        n_bjets = ak.to_numpy(events["n_bjets"]).astype(int)
        final_cut_mask = ak.to_numpy(build_cut_masks(events)["cut_iv"]).astype(bool)
        event_weight = (
            xsec_info["cross_section_pb"] / generated_events * luminosity_fb * 1000.0
        )
        weights = np.full(n_jets.shape, event_weight, dtype=float)

        counts_by_variant["inclusive"].append(
            {
                "process_number": process_number,
                "label": build_process_label(process_number, xsec_info["description"]),
                "jet_counts": n_jets,
                "bjet_counts": n_bjets,
                "weights": weights,
            }
        )
        counts_by_variant["after-cuts"].append(
            {
                "process_number": process_number,
                "label": build_process_label(process_number, xsec_info["description"]),
                "jet_counts": n_jets[final_cut_mask],
                "bjet_counts": n_bjets[final_cut_mask],
                "weights": weights[final_cut_mask],
            }
        )

    return counts_by_variant


def build_process_label(process_number: int, description: str) -> str:
    process_labels = {
        1: "proc 1: ttbar",
        2: "proc 2: tW",
        3: "proc 3: llbb",
        4: "proc 4: jjW±W±",
    }
    return process_labels.get(process_number, f"proc {process_number}: {description}")


def build_integer_bins(values: np.ndarray) -> np.ndarray:
    max_value = int(np.max(values)) if values.size else 0
    return np.arange(-0.5, max_value + 1.5, 1.0)


def make_plot(
    process_entries: list[dict[str, object]],
    luminosity_fb: float,
    output_path: Path,
    sample_label: str,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    all_jet_counts = [entry["jet_counts"] for entry in process_entries]
    all_bjet_counts = [entry["bjet_counts"] for entry in process_entries]
    max_jet_count = max((int(np.max(values)) if len(values) else 0) for values in all_jet_counts)
    max_bjet_count = max((int(np.max(values)) if len(values) else 0) for values in all_bjet_counts)
    plot_specs = (
        (
            axes[0],
            "jet_counts",
            max_jet_count,
            "Jet multiplicity",
            "Number of jets",
        ),
        (
            axes[1],
            "bjet_counts",
            max_bjet_count,
            "Positive b-tag multiplicity",
            "Number of b-tagged jets",
        ),
    )

    for axis, field_name, max_value, title, xlabel in plot_specs:
        bins = build_integer_bins(np.array([max_value], dtype=int))
        values_by_process = [entry[field_name] for entry in process_entries]
        weights_by_process = [entry["weights"] for entry in process_entries]
        colors = [
            PROCESS_COLORS.get(int(entry["process_number"]), "#9c755f")
            for entry in process_entries
        ]
        labels = [str(entry["label"]) for entry in process_entries]
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
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Expected background events")
        axis.set_xticks(np.arange(0, max_value + 1, 1))
        axis.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.55)
        axis.legend(fontsize=8)

    total_expected_events = float(
        sum(np.sum(np.asarray(entry["weights"], dtype=float)) for entry in process_entries)
    )
    fig.suptitle(
        rf"Total background {sample_label} weighted by $\sigma \times \mathcal{{L}}$ at {luminosity_fb:g} fb$^{{-1}}$"
        + f"\nTotal expected events: {total_expected_events:.2f}"
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path.resolve()


def build_output_path(script_dir: Path, luminosity_fb: float, variant: str) -> Path:
    if variant == "inclusive":
        return script_dir / f"jets-histogram-background-lumi-{luminosity_fb:g}.png"
    return script_dir / f"jets-histogram-background-{variant}-lumi-{luminosity_fb:g}.png"


def main() -> int:
    args = parse_args()
    vector.register_awkward()
    parquet_entries = discover_background_parquets(args.parquet_dir)
    xsecs_by_process = load_xsec_rows(args.xsec_csv)

    for luminosity_fb in args.luminosities:
        counts_by_variant = load_weighted_event_counts(
            parquet_entries,
            xsecs_by_process,
            luminosity_fb,
        )
        for variant, process_entries in counts_by_variant.items():
            output_path = make_plot(
                process_entries,
                luminosity_fb,
                build_output_path(SCRIPT_DIR, luminosity_fb, variant),
                "after cuts" if variant == "after-cuts" else "inclusive",
            )
            print(f"Wrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
