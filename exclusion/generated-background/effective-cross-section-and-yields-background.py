#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CUTS_CSV_PATH = SCRIPT_DIR / "cuts-background.csv"
XSEC_CSV_PATH = SCRIPT_DIR / "xsec-background.csv"
OUTPUT_CSV_PATH = SCRIPT_DIR / "effective-cross-section-and-yields-background.csv"
LUMINOSITY_450_FB = 450.0
LUMINOSITY_3000_FB = 3000.0


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


def main() -> int:
    cutflow_by_process = load_cutflow_rows(CUTS_CSV_PATH)
    xsec_by_process = load_xsec_rows(XSEC_CSV_PATH)
    output_rows = build_output_rows(cutflow_by_process, xsec_by_process)
    output_path = write_output_csv(OUTPUT_CSV_PATH, output_rows)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
