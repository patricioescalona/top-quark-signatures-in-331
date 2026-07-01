#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from generated_signal_paths import (
    build_generated_dir,
    discover_generated_dirs,
    format_value_for_filename,
)


LUMINOSITY_450_FB = 450.0
LUMINOSITY_3000_FB = 3000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read signal cuts/xsec CSV files and write one effective cross-section "
            "and yields CSV per generated folder."
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
    args = parser.parse_args()

    if (args.mass is None) != (args.tanphi is None):
        parser.error("Use both --mass and --tanphi together, or neither of them.")

    return args


def build_cuts_csv_path(generated_dir: Path, mass: str, tanphi: str) -> Path:
    return generated_dir / (
        f"cuts-m-{format_value_for_filename(mass)}"
        f"-tanphi-{format_value_for_filename(tanphi)}.csv"
    )


def build_xsec_csv_path(generated_dir: Path, mass: str, tanphi: str) -> Path:
    return generated_dir / (
        f"xsec-m-{format_value_for_filename(mass)}"
        f"-tanphi-{format_value_for_filename(tanphi)}.csv"
    )


def build_output_csv_path(generated_dir: Path, mass: str, tanphi: str) -> Path:
    return generated_dir / (
        f"effective-cross-section-and-yields-m-{format_value_for_filename(mass)}"
        f"-tanphi-{format_value_for_filename(tanphi)}.csv"
    )


def load_cutflow_rows(cuts_csv_path: Path) -> dict[int, dict[str, str]]:
    with cuts_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, skipinitialspace=True)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {cuts_csv_path}")

    header = [cell.strip() for cell in rows[0]]
    process_numbers: list[int] = []
    for column_name in header[1:]:
        if not column_name.startswith("proc_"):
            continue
        suffix = column_name.removeprefix("proc_")
        if not suffix.isdigit():
            continue
        process_numbers.append(int(suffix))

    if not process_numbers:
        raise ValueError(f"No process columns found in {cuts_csv_path}")

    cutflow_by_process = {process_number: {} for process_number in process_numbers}
    for row in rows[1:]:
        if not row:
            continue
        label = row[0].strip()
        values = row[1 : 1 + len(process_numbers)]
        for process_number, value in zip(process_numbers, values):
            cutflow_by_process[process_number][label] = value.strip()

    return cutflow_by_process


def load_xsec_rows(xsec_csv_path: Path) -> dict[int, dict[str, str]]:
    with xsec_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {int(row["process"]): row for row in reader}


def build_output_rows(
    cutflow_by_process: dict[int, dict[str, str]],
    xsec_by_process: dict[int, dict[str, str]],
) -> list[dict[str, str]]:
    output_rows: list[dict[str, str]] = []
    total_effective_cross_section_pb = 0.0
    total_yield_450_fb = 0.0
    total_yield_3000_fb = 0.0

    for process_number in sorted(cutflow_by_process):
        if process_number not in xsec_by_process:
            raise ValueError(f"Missing xsec row for process {process_number}")

        cutflow_row = cutflow_by_process[process_number]
        xsec_row = xsec_by_process[process_number]

        total_events = int(cutflow_row["total_events"])
        passing_events = int(cutflow_row["pass_cut_iv_mll"])
        cross_section_pb = float(xsec_row["cross section pb"])

        if total_events <= 0:
            raise ValueError(f"Non-positive total event count for process {process_number}")

        effective_cross_section_pb = cross_section_pb * passing_events / total_events
        yield_450_fb = effective_cross_section_pb * LUMINOSITY_450_FB * 1000.0
        yield_3000_fb = effective_cross_section_pb * LUMINOSITY_3000_FB * 1000.0
        total_effective_cross_section_pb += effective_cross_section_pb
        total_yield_450_fb += yield_450_fb
        total_yield_3000_fb += yield_3000_fb

        output_rows.append(
            {
                "process": str(process_number),
                "xsec_pb": f"{effective_cross_section_pb:.6f}",
                "yield_450fb": f"{yield_450_fb:.6f}",
                "yield_3000fb": f"{yield_3000_fb:.6f}",
            }
        )

    output_rows.append(
        {
            "process": "total",
            "xsec_pb": f"{total_effective_cross_section_pb:.6f}",
            "yield_450fb": f"{total_yield_450_fb:.6f}",
            "yield_3000fb": f"{total_yield_3000_fb:.6f}",
        }
    )

    return output_rows


def write_output_csv(output_csv_path: Path, rows: list[dict[str, str]]) -> Path:
    with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "process",
                "xsec_pb",
                "yield_450fb",
                "yield_3000fb",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_csv_path.resolve()


def run_for_generated_dir(mass: str, tanphi: str, generated_dir: Path) -> Path:
    cuts_csv_path = build_cuts_csv_path(generated_dir, mass, tanphi)
    xsec_csv_path = build_xsec_csv_path(generated_dir, mass, tanphi)
    output_csv_path = build_output_csv_path(generated_dir, mass, tanphi)

    cutflow_by_process = load_cutflow_rows(cuts_csv_path)
    xsec_by_process = load_xsec_rows(xsec_csv_path)
    output_rows = build_output_rows(cutflow_by_process, xsec_by_process)
    return write_output_csv(output_csv_path, output_rows)


def main() -> int:
    args = parse_args()
    base_dir = args.base_dir.resolve()

    if args.mass is not None and args.tanphi is not None:
        output_path = run_for_generated_dir(
            args.mass,
            args.tanphi,
            build_generated_dir(base_dir, args.mass, args.tanphi),
        )
        print(f"Wrote {output_path}")
        return 0

    for mass, tanphi, generated_dir in discover_generated_dirs(base_dir):
        output_path = run_for_generated_dir(mass, tanphi, generated_dir)
        print(f"Wrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
