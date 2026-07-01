#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import awkward as ak
import vector

from generated_signal_paths import (
    build_generated_dir,
    discover_generated_dirs,
    format_value_for_filename,
)


PROC_PATTERN = re.compile(r"^proc-(\d+)-m-.*-tanphi-.*\.parquet$")
EXPECTED_PROCESSES = {1, 2, 3, 4, 5, 6}
LUMINOSITY_450_FB = 450.0
LUMINOSITY_3000_FB = 3000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read generated parquet files, apply the cutflow, and write one CSV "
            "per generated folder."
        )
    )
    parser.add_argument("--mass", help="Mass label used in the folder name.")
    parser.add_argument("--tanphi", help="Tanphi label used in the folder name.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Base exclusion directory. Default: exclusion/",
    )
    return parser.parse_args()


def build_output_csv_path(generated_dir: Path, mass: str, tanphi: str) -> Path:
    return generated_dir / (
        f"cuts-m-{format_value_for_filename(mass)}"
        f"-tanphi-{format_value_for_filename(tanphi)}.csv"
    )


def build_efficiencies_csv_path(generated_dir: Path, mass: str, tanphi: str) -> Path:
    return generated_dir / (
        f"efficiencies-m-{format_value_for_filename(mass)}"
        f"-tanphi-{format_value_for_filename(tanphi)}.csv"
    )


def build_xsec_csv_path(generated_dir: Path, mass: str, tanphi: str) -> Path:
    return generated_dir / (
        f"xsec-m-{format_value_for_filename(mass)}"
        f"-tanphi-{format_value_for_filename(tanphi)}.csv"
    )


def discover_parquets(parquet_dir: Path) -> list[tuple[int, Path]]:
    parquets: list[tuple[int, Path]] = []
    for path in parquet_dir.glob("proc-*.parquet"):
        match = PROC_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        parquets.append((int(match.group(1)), path))

    if not parquets:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

    process_numbers = {process_number for process_number, _ in parquets}
    if process_numbers != EXPECTED_PROCESSES:
        raise ValueError(
            f"Expected parquet files for processes {sorted(EXPECTED_PROCESSES)} in "
            f"{parquet_dir}, but found {sorted(process_numbers)}."
        )

    return sorted(parquets, key=lambda item: item[0])


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


def count_events(mask: ak.Array) -> int:
    return int(ak.sum(mask))


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


def summarize_cutflow(process_number: int, parquet_path: Path) -> dict[str, int]:
    events = ak.from_parquet(parquet_path)
    total_events = len(events)
    cut_masks = build_cut_masks(events)

    return {
        "proc": process_number,
        "total_events": total_events,
        "pass_cut_i_bjet": count_events(cut_masks["cut_i"]),
        "pass_cut_ii_met": count_events(cut_masks["cut_ii"]),
        "pass_cut_iii_same_sign_dilepton": count_events(cut_masks["cut_iii"]),
        "pass_cut_iv_mll": count_events(cut_masks["cut_iv"]),
    }


def build_transposed_cutflow_rows(
    cutflow_rows: list[dict[str, int]],
    process_weights: dict[int, dict[str, float]],
) -> list[list[str | int | float]]:
    rows_by_process = {int(row["proc"]): row for row in cutflow_rows}
    process_numbers = sorted(rows_by_process)

    transposed_rows: list[list[str | int | float]] = [[
        "process",
        *[f"proc_{process_number}" for process_number in process_numbers],
        "total 450/fb",
        "total 3000/fb",
    ]]

    fields = [
        ("total_events", "total_events"),
        ("pass_cut_i_bjet", "pass_cut_i_bjet"),
        ("pass_cut_ii_met", "pass_cut_ii_met"),
        ("pass_cut_iii_same_sign_dilepton", "pass_cut_iii_same_sign_dilepton"),
        ("pass_cut_iv_mll", "pass_cut_iv_mll"),
    ]

    for label, field_name in fields:
        raw_values = [
            rows_by_process[process_number][field_name] for process_number in process_numbers
        ]
        total_450_fb = 0.0
        total_3000_fb = 0.0
        for process_number in process_numbers:
            event_weight = process_weights[process_number]["event_weight_pb"]
            weighted_events_pb = rows_by_process[process_number][field_name] * event_weight
            total_450_fb += weighted_events_pb * LUMINOSITY_450_FB * 1000.0
            total_3000_fb += weighted_events_pb * LUMINOSITY_3000_FB * 1000.0
        transposed_rows.append(
            [label, *raw_values, f"{total_450_fb:.6f}", f"{total_3000_fb:.6f}"]
        )

    percentage_fields = [
        ("pass_cut_i_bjet_frac", "pass_cut_i_bjet"),
        ("pass_cut_ii_met_frac", "pass_cut_ii_met"),
        ("pass_cut_iii_same_sign_dilepton_frac", "pass_cut_iii_same_sign_dilepton"),
        ("pass_cut_iv_mll_frac", "pass_cut_iv_mll"),
    ]

    for label, field_name in percentage_fields:
        percentage_values = []
        total_initial_450_fb = 0.0
        total_initial_3000_fb = 0.0
        total_passing_450_fb = 0.0
        total_passing_3000_fb = 0.0

        for process_number in process_numbers:
            total_events = rows_by_process[process_number]["total_events"]
            passing_events = rows_by_process[process_number][field_name]
            percentage = 0.0 if total_events == 0 else (passing_events / total_events)
            percentage_values.append(f"{percentage:.6f}")

            event_weight = process_weights[process_number]["event_weight_pb"]
            total_initial_450_fb += total_events * event_weight * LUMINOSITY_450_FB * 1000.0
            total_initial_3000_fb += (
                total_events * event_weight * LUMINOSITY_3000_FB * 1000.0
            )
            total_passing_450_fb += (
                passing_events * event_weight * LUMINOSITY_450_FB * 1000.0
            )
            total_passing_3000_fb += (
                passing_events * event_weight * LUMINOSITY_3000_FB * 1000.0
            )

        combined_efficiency_450_fb = (
            0.0 if total_initial_450_fb == 0.0 else total_passing_450_fb / total_initial_450_fb
        )
        combined_efficiency_3000_fb = (
            0.0
            if total_initial_3000_fb == 0.0
            else total_passing_3000_fb / total_initial_3000_fb
        )
        transposed_rows.append(
            [
                label,
                *percentage_values,
                f"{combined_efficiency_450_fb:.6f}",
                f"{combined_efficiency_3000_fb:.6f}",
            ]
        )

    return transposed_rows


def write_cutflow_csv(output_path: Path, rows: list[list[str | int | float]]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def load_process_weights(
    generated_dir: Path, mass: str, tanphi: str
) -> dict[int, dict[str, float]]:
    xsec_csv_path = build_xsec_csv_path(generated_dir, mass, tanphi)
    if not xsec_csv_path.is_file():
        raise FileNotFoundError(f"Cross-section CSV not found: {xsec_csv_path}")

    process_weights: dict[int, dict[str, float]] = {}
    with xsec_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            process_number = int(row["process"])
            cross_section_pb = float(row["cross section pb"])
            generated_events = float(row["N events (generated)"])
            if generated_events <= 0:
                raise ValueError(
                    f"Generated event count must be positive for process "
                    f"{process_number} in {xsec_csv_path}."
                )
            process_weights[process_number] = {
                "cross_section_pb": cross_section_pb,
                "generated_events": generated_events,
                "event_weight_pb": cross_section_pb / generated_events,
            }

    if set(process_weights) != EXPECTED_PROCESSES:
        raise ValueError(
            f"Expected weight entries for processes {sorted(EXPECTED_PROCESSES)} in "
            f"{xsec_csv_path}, but found {sorted(process_weights)}."
        )

    return process_weights


def compute_efficiency_rows(
    parquet_entries: list[tuple[int, Path]],
    process_weights: dict[int, dict[str, float]],
) -> list[dict[str, str]]:
    weighted_total = 0.0
    weighted_cut_i = 0.0
    weighted_cut_ii = 0.0
    weighted_cut_iii = 0.0
    weighted_cut_iv = 0.0

    for process_number, parquet_path in parquet_entries:
        events = ak.from_parquet(parquet_path)
        cut_masks = build_cut_masks(events)
        event_weight = process_weights[process_number]["event_weight_pb"]
        generated_events = process_weights[process_number]["generated_events"]

        weighted_total += generated_events * event_weight
        weighted_cut_i += count_events(cut_masks["cut_i"]) * event_weight
        weighted_cut_ii += count_events(cut_masks["cut_ii"]) * event_weight
        weighted_cut_iii += count_events(cut_masks["cut_iii"]) * event_weight
        weighted_cut_iv += count_events(cut_masks["cut_iv"]) * event_weight

    if weighted_total <= 0.0:
        raise ValueError("The total weighted yield is zero; efficiencies are undefined.")

    def build_efficiency_row(cut_label: str, cross_section_pb: float) -> dict[str, str]:
        return {
            "cut": cut_label,
            "efficiency": f"{cross_section_pb / weighted_total:.6f}",
            "cross_section_pb": f"{cross_section_pb:.6f}",
            "yield_450_fb": f"{cross_section_pb * LUMINOSITY_450_FB * 1000.0:.6f}",
            "yield_3000_fb": f"{cross_section_pb * LUMINOSITY_3000_FB * 1000.0:.6f}",
        }

    return [
        build_efficiency_row("I", weighted_cut_i),
        build_efficiency_row("II", weighted_cut_ii),
        build_efficiency_row("III", weighted_cut_iii),
        build_efficiency_row("IV", weighted_cut_iv),
    ]


def write_efficiencies_csv(output_path: Path, rows: list[dict[str, str]]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cut",
                "efficiency",
                "cross_section_pb",
                "yield_450_fb",
                "yield_3000_fb",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def run_cutflow_for_generated_dir(
    mass: str, tanphi: str, generated_dir: Path
) -> tuple[Path, Path]:
    if not generated_dir.is_dir():
        raise FileNotFoundError(f"Generated directory not found: {generated_dir}")

    parquet_dir = generated_dir / "parquets"
    parquet_entries = discover_parquets(parquet_dir)
    rows = [
        summarize_cutflow(process_number, parquet_path)
        for process_number, parquet_path in parquet_entries
    ]
    process_weights = load_process_weights(generated_dir, mass, tanphi)
    cuts_output_path = build_output_csv_path(generated_dir, mass, tanphi)
    write_cutflow_csv(
        cuts_output_path,
        build_transposed_cutflow_rows(rows, process_weights),
    )

    efficiencies_rows = compute_efficiency_rows(parquet_entries, process_weights)
    efficiencies_output_path = build_efficiencies_csv_path(generated_dir, mass, tanphi)
    write_efficiencies_csv(efficiencies_output_path, efficiencies_rows)

    return cuts_output_path, efficiencies_output_path


def main() -> int:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    vector.register_awkward()

    if (args.mass is None) != (args.tanphi is None):
        raise ValueError("Use both --mass and --tanphi together, or neither of them.")

    if args.mass is not None and args.tanphi is not None:
        cuts_output_path, efficiencies_output_path = run_cutflow_for_generated_dir(
            args.mass,
            args.tanphi,
            build_generated_dir(base_dir, args.mass, args.tanphi),
        )
        print(f"Wrote {cuts_output_path}")
        print(f"Wrote {efficiencies_output_path}")
        return 0

    for mass, tanphi, generated_dir in discover_generated_dirs(base_dir):
        cuts_output_path, efficiencies_output_path = run_cutflow_for_generated_dir(
            mass, tanphi, generated_dir
        )
        print(f"Wrote {cuts_output_path}")
        print(f"Wrote {efficiencies_output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
