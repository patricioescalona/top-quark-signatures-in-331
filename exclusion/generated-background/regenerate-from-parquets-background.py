#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import awkward as ak
import vector


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_SCRIPT_PATH = SCRIPT_DIR / "compressor-and-cuts-background.py"
DEFAULT_LABEL = "background"


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
        stem = parquet_path.stem
        parts = stem.split("-")
        if len(parts) < 3 or parts[0] != "proc":
            continue
        try:
            process_number = int(parts[1])
        except ValueError:
            continue
        parquets.append((process_number, parquet_path.resolve()))

    if not parquets:
        raise FileNotFoundError(f"No background parquet files found in {parquet_dir}")

    return parquets


def load_xsec_rows(xsec_csv_path: Path) -> list[dict[str, str]]:
    if not xsec_csv_path.is_file():
        raise FileNotFoundError(
            "Cross-section metadata is required to rebuild efficiencies and the "
            f"decay summary, but {xsec_csv_path} was not found."
        )

    with xsec_csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_decay_summary_from_xsec_rows(xsec_rows: list[dict[str, str]], label: str) -> str:
    lines = [f"sample: {label}", "", "Processes:"]
    for row in sorted(xsec_rows, key=lambda item: int(item["process"])):
        description = row["description"].strip()
        if description:
            for process_description in description.split(" ; "):
                lines.append(f"proc {row['process']}: {process_description}")
        else:
            lines.append(f"proc {row['process']}: no process description found in xsec csv")

    lines.extend(
        [
            "",
            "BR(a -> X):",
            "not applicable for this background sample",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    vector.register_awkward()
    background_module = load_background_module()

    output_dir = SCRIPT_DIR
    label = DEFAULT_LABEL
    parquet_dir = output_dir / "parquets"

    parquet_entries = discover_background_parquets(parquet_dir)
    cutflow_rows = []
    for process_number, parquet_path in parquet_entries:
        events = ak.from_parquet(parquet_path)
        counts = background_module.summarize_cutflow_counts(events)
        cutflow_rows.append(background_module.build_cutflow_row(process_number, counts))

    cutflow_rows.sort(key=lambda row: row["proc"])
    cuts_output_path = background_module.write_cutflow_csv(
        background_module.build_cuts_output_path(output_dir, label),
        cutflow_rows,
    )

    xsec_output_path = background_module.build_xsec_output_path(output_dir, label)
    xsec_rows = load_xsec_rows(xsec_output_path)
    xsec_rows.sort(key=lambda row: int(row["process"]))

    efficiencies_output_path = background_module.write_efficiencies_csv(
        background_module.build_efficiencies_output_path(output_dir, label),
        background_module.build_efficiency_rows_from_process_rows(cutflow_rows, xsec_rows),
    )

    decay_output_path = background_module.build_decay_output_path(output_dir, label)
    decay_output_path.write_text(
        build_decay_summary_from_xsec_rows(xsec_rows, label),
        encoding="utf-8",
    )
    decay_output_path = decay_output_path.resolve()

    print(f"Wrote {cuts_output_path}")
    print(f"Wrote {efficiencies_output_path}")
    print(f"Wrote {decay_output_path}")
    print(
        f"Reused metadata from {xsec_output_path.resolve()} to rebuild efficiencies "
        "and the decay summary."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
