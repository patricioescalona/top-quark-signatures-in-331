#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import awkward as ak
import numpy as np


PROC_PATTERN = re.compile(r"^proc-(\d+)-m-.*-tanphi-.*\.parquet$")
GENERATED_DIR_PATTERN = re.compile(r"^generated-m-(.+)-tanphi-(.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read generated parquet files and print a compact terminal summary."
        )
    )
    parser.add_argument("--mass", help="Mass label used in the folder name.")
    parser.add_argument(
        "--tanphi", help="Tanphi label used in the folder name."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Base exclusion directory. Default: exclusion/",
    )
    return parser.parse_args()


def format_value_for_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    sanitized = sanitized.strip("-")
    return sanitized or "value"


def build_generated_dir(base_dir: Path, mass: str, tanphi: str) -> Path:
    return base_dir / (
        f"generated-m-{format_value_for_filename(mass)}"
        f"-tanphi-{format_value_for_filename(tanphi)}"
    )


def discover_generated_dirs(base_dir: Path) -> list[tuple[str, str, Path]]:
    generated_dirs: list[tuple[str, str, Path]] = []
    for path in sorted(base_dir.glob("generated-m-*-tanphi-*")):
        if not path.is_dir():
            continue
        match = GENERATED_DIR_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        generated_dirs.append((match.group(1), match.group(2), path))

    if not generated_dirs:
        raise FileNotFoundError(f"No generated folders found in {base_dir}")

    return generated_dirs


def discover_parquets(parquet_dir: Path) -> list[tuple[int, Path]]:
    parquets: list[tuple[int, Path]] = []
    for path in parquet_dir.glob("proc-*.parquet"):
        match = PROC_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        parquets.append((int(match.group(1)), path))

    if not parquets:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

    return sorted(parquets, key=lambda item: item[0])


def summarize_parquet(process_number: int, parquet_path: Path) -> dict[str, float]:
    events = ak.from_parquet(parquet_path)
    return {
        "process": process_number,
        "events": len(events),
        "mean_met": float(np.mean(ak.to_numpy(events.met))),
        "mean_n_electrons": float(np.mean(ak.to_numpy(events.n_electrons))),
        "mean_n_muons": float(np.mean(ak.to_numpy(events.n_muons))),
        "mean_n_leptons": float(np.mean(ak.to_numpy(events.n_leptons))),
        "mean_n_jets": float(np.mean(ak.to_numpy(events.n_jets))),
        "mean_n_bjets": float(np.mean(ak.to_numpy(events.n_bjets))),
        "met_values": ak.to_numpy(events.met),
    }


def print_summary_table(rows: list[dict[str, float]]) -> None:
    print("Parquet summary:")
    print("proc | events | mean MET | <nlep> | <njets> | <nbjets>")
    for row in rows:
        print(
            f"{int(row['process']):>4} | "
            f"{int(row['events']):>6} | "
            f"{row['mean_met']:>8.3f} | "
            f"{row['mean_n_leptons']:>6.3f} | "
            f"{row['mean_n_jets']:>7.3f} | "
            f"{row['mean_n_bjets']:>8.3f}"
        )


def run_summary_for_generated_dir(mass: str, tanphi: str, generated_dir: Path) -> None:
    parquet_dir = generated_dir / "parquets"
    parquet_entries = discover_parquets(parquet_dir)
    rows = [summarize_parquet(process_number, parquet_path) for process_number, parquet_path in parquet_entries]
    print(f"mass = {mass}, tanphi = {tanphi}")
    print_summary_table(rows)
    print()


def main() -> int:
    args = parse_args()
    base_dir = args.base_dir.resolve()

    if (args.mass is None) != (args.tanphi is None):
        raise ValueError("Use both --mass and --tanphi together, or neither of them.")

    if args.mass is not None and args.tanphi is not None:
        run_summary_for_generated_dir(
            args.mass,
            args.tanphi,
            build_generated_dir(base_dir, args.mass, args.tanphi),
        )
        return 0

    for mass, tanphi, generated_dir in discover_generated_dirs(base_dir):
        run_summary_for_generated_dir(mass, tanphi, generated_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
