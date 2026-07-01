#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import awkward as ak
import vector


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_SCRIPT_PATH = SCRIPT_DIR / "compressor-and-cuts-background.py"
OUTPUT_PATH = SCRIPT_DIR / "cuts-background.csv"


def load_background_module():
    spec = importlib.util.spec_from_file_location(
        "compressor_and_cuts_background",
        SOURCE_SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {SOURCE_SCRIPT_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover_background_parquets(parquet_dir: Path) -> list[tuple[int, Path]]:
    parquets: list[tuple[int, Path]] = []
    for parquet_path in sorted(parquet_dir.glob("proc-*-background.parquet")):
        stem_parts = parquet_path.stem.split("-")
        if len(stem_parts) < 3 or stem_parts[0] != "proc":
            continue

        try:
            process_number = int(stem_parts[1])
        except ValueError:
            continue

        parquets.append((process_number, parquet_path.resolve()))

    if len(parquets) != 4:
        raise ValueError(
            f"Expected exactly 4 background parquet files in {parquet_dir}, found {len(parquets)}."
        )

    return parquets


def build_transposed_rows(cutflow_rows: list[dict[str, int]]) -> list[list[str | int]]:
    rows_by_process = {int(row["proc"]): row for row in cutflow_rows}
    process_numbers = sorted(rows_by_process)

    headers = ["process", *[f"proc_{process_number}" for process_number in process_numbers]]
    transposed_rows: list[list[str | int]] = [headers]

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
        transposed_rows.append([label, *raw_values])

    percentage_fields = [
        ("pass_cut_i_bjet_frac", "pass_cut_i_bjet"),
        ("pass_cut_ii_met_frac", "pass_cut_ii_met"),
        ("pass_cut_iii_same_sign_dilepton_frac", "pass_cut_iii_same_sign_dilepton"),
        ("pass_cut_iv_mll_frac", "pass_cut_iv_mll"),
    ]

    for label, field_name in percentage_fields:
        percentage_values = []
        for process_number in process_numbers:
            total_events = rows_by_process[process_number]["total_events"]
            value = rows_by_process[process_number][field_name]
            percentage = 0.0 if total_events == 0 else (value / total_events)
            percentage_values.append(f"{percentage:.6f}")
        transposed_rows.append([label, *percentage_values])

    return transposed_rows


def write_transposed_csv(output_path: Path, rows: list[list[str | int]]) -> Path:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)
    return output_path.resolve()


def main() -> int:
    vector.register_awkward()
    background_module = load_background_module()
    parquet_dir = SCRIPT_DIR / "parquets"
    parquet_entries = discover_background_parquets(parquet_dir)

    cutflow_rows = []
    for process_number, parquet_path in parquet_entries:
        events = ak.from_parquet(parquet_path)
        counts = background_module.summarize_cutflow_counts(events)
        cutflow_rows.append(background_module.build_cutflow_row(process_number, counts))

    cutflow_rows.sort(key=lambda row: row["proc"])
    output_path = write_transposed_csv(OUTPUT_PATH, build_transposed_rows(cutflow_rows))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
