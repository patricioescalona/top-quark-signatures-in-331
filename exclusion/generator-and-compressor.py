#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
from pathlib import Path


DEFAULT_MG5_BIN = Path(
    os.environ.get("MG5_BIN_DIR", "/home/patricio/Documents/mg5amcnlo-3.x/bin")
)
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("tmp_run.mg5")
PROCESS_DIR_PATTERN = re.compile(r"^proc-(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write a MadGraph .mg5 command file that launches every generated "
            "proc-* directory in the local MG5 bin folder with the same number "
            "of events."
        )
    )
    parser.add_argument(
        "nevents",
        type=int,
        help="Number of events to generate in each discovered process directory.",
    )
    parser.add_argument(
        "--mass-pdg",
        type=int,
        default=36,
        help="PDG code whose mass entry will be updated. Default: 36",
    )
    parser.add_argument(
        "--mass",
        default="200",
        help="Mass value written through 'set mass <pdg> <value>'. Default: 200",
    )
    parser.add_argument(
        "--tanphi-index",
        type=int,
        default=1,
        help="Index used in the TANPHI block. Default: 1",
    )
    parser.add_argument(
        "--tanphi",
        default="60",
        help="TANPHI value written through 'set tanphi <index> <value>'. Default: 60",
    )
    parser.add_argument(
        "--ebeam1",
        default="7000",
        help="Beam-1 energy in GeV. Default: 7000",
    )
    parser.add_argument(
        "--ebeam2",
        default="7000",
        help="Beam-2 energy in GeV. Default: 7000",
    )
    parser.add_argument(
        "--mg5-bin",
        type=Path,
        default=DEFAULT_MG5_BIN,
        help="Path to the MadGraph bin directory that contains proc-* outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination .mg5 file. Default: exclusion/tmp_run.mg5",
    )
    parser.add_argument(
        "--write-only",
        action="store_true",
        help="Only write the .mg5 card without launching MadGraph.",
    )
    parser.add_argument(
        "--xsec-output",
        type=Path,
        default=None,
        help=(
            "Destination CSV file for cross section summaries. "
            "Default: exclusion/xsec-m-<mass>-tanphi-<tanphi>.csv"
        ),
    )
    args = parser.parse_args()

    if args.nevents <= 0:
        parser.error("nevents must be a positive integer.")
    if args.output.suffix != ".mg5":
        parser.error("--output must end with .mg5.")
    if args.xsec_output is None:
        args.xsec_output = build_default_xsec_output(args.mass, args.tanphi)
    if args.xsec_output.suffix != ".csv":
        parser.error("--xsec-output must end with .csv.")

    return args


def format_value_for_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    sanitized = sanitized.strip("-")
    return sanitized or "value"


def build_default_xsec_output(mass: str, tanphi: str) -> Path:
    filename = (
        f"xsec-m-{format_value_for_filename(mass)}"
        f"-tanphi-{format_value_for_filename(tanphi)}.csv"
    )
    return Path(__file__).resolve().with_name(filename)


def discover_process_directories(mg5_bin: Path) -> list[Path]:
    if not mg5_bin.exists():
        raise FileNotFoundError(f"MadGraph bin directory not found: {mg5_bin}")
    if not mg5_bin.is_dir():
        raise NotADirectoryError(f"MadGraph bin path is not a directory: {mg5_bin}")

    process_entries: list[tuple[int, str, Path]] = []
    for path in mg5_bin.iterdir():
        match = PROCESS_DIR_PATTERN.fullmatch(path.name)
        if match is None or not path.is_dir():
            continue
        if not (path / "Cards" / "proc_card_mg5.dat").is_file():
            continue

        process_entries.append((int(match.group(1)), path.name, path.resolve()))

    process_dirs = [
        path for _, _, path in sorted(process_entries, key=lambda item: (item[0], item[1]))
    ]
    if not process_dirs:
        raise FileNotFoundError(
            f"No generated proc-* directories were found inside {mg5_bin}."
        )

    return process_dirs


def build_mg5_card(
    process_dirs: list[Path],
    nevents: int,
    mass_pdg: int,
    mass: str,
    tanphi_index: int,
    tanphi: str,
    ebeam1: str,
    ebeam2: str,
) -> str:
    lines = [
        "set automatic_html_opening False --no_save",
        "set notification_center False --no_save",
        "",
    ]

    for process_dir in process_dirs:
        lines.extend(
            [
                f"launch {process_dir.name}",
                "shower=Pythia8",
                "detector=Delphes",
                f"set nevents {nevents}",
                f"set mass {mass_pdg} {mass}",
                "set width 36 Auto",
                f"set tanphi {tanphi_index} {tanphi}",
                f"set ebeam1 {ebeam1}",
                f"set ebeam2 {ebeam2}",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def write_card(output_path: Path, card_text: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(card_text, encoding="utf-8")
    return output_path.resolve()


def run_madgraph(mg5_bin: Path, card_path: Path) -> int:
    mg5_executable = mg5_bin / "mg5_aMC"
    if not mg5_executable.is_file():
        raise FileNotFoundError(f"MadGraph executable not found: {mg5_executable}")
    if not os.access(mg5_executable, os.X_OK):
        raise PermissionError(f"MadGraph executable is not executable: {mg5_executable}")

    completed = subprocess.run(
        [str(mg5_executable.resolve()), str(card_path.resolve())],
        cwd=str(mg5_bin),
        check=False,
    )
    return completed.returncode


def extract_process_description(process_dir: Path) -> str:
    proc_card_path = process_dir / "Cards" / "proc_card_mg5.dat"
    lines = proc_card_path.read_text(encoding="utf-8", errors="replace").splitlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("generate "):
            i += 1
            continue

        description = line[len("generate ") :].rstrip()
        while description.endswith("\\") and i + 1 < len(lines):
            description = description[:-1] + lines[i + 1].strip()
            i += 1
        return " ".join(description.split())

    raise ValueError(f"No 'generate' command found in {proc_card_path}")


def extract_requested_events_from_banner(banner_path: Path) -> str:
    content = banner_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"^\s*([0-9]+(?:\.[0-9]+)?)\s*=\s*nevents\b",
        content,
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"Could not find requested event count in {banner_path}")

    nevents = match.group(1)
    return nevents[:-2] if nevents.endswith(".0") else nevents


def extract_cross_section_from_banner(banner_path: Path) -> tuple[str, str, str]:
    content = banner_path.read_text(encoding="utf-8", errors="replace")

    generated_events_match = re.search(
        r"#\s*Number of Events\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        content,
    )
    if generated_events_match is not None:
        generated_events = generated_events_match.group(1)
        if generated_events.endswith(".0"):
            generated_events = generated_events[:-2]
    else:
        generated_events = extract_requested_events_from_banner(banner_path)

    init_match = re.search(
        r"<init>\s*\n.*\n\s*([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+[0-9.eE+-]+\s+\d+",
        content,
        flags=re.DOTALL,
    )
    if init_match is None:
        raise ValueError(f"Could not find cross section data in {banner_path}")

    return init_match.group(1), init_match.group(2), generated_events


def extract_latest_existing_banner(process_dir: Path) -> tuple[int, int, Path]:
    latest_banner: tuple[int, int, Path] | None = None
    for banner_path in process_dir.glob("Events/run_*/run_*_tag_*_banner.txt"):
        match = re.fullmatch(
            r"run_(\d+)_tag_(\d+)_banner\.txt",
            banner_path.name,
        )
        if match is None:
            continue

        candidate = (int(match.group(1)), int(match.group(2)), banner_path)
        if latest_banner is None or candidate[:2] > latest_banner[:2]:
            latest_banner = candidate

    if latest_banner is None:
        raise FileNotFoundError(f"No banner files found in {process_dir / 'Events'}")

    return latest_banner


def build_xsec_rows(process_dirs: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for process_dir in process_dirs:
        process_match = PROCESS_DIR_PATTERN.fullmatch(process_dir.name)
        if process_match is None:
            raise ValueError(f"Unexpected process directory name: {process_dir.name}")

        _, _, banner_path = extract_latest_existing_banner(process_dir)
        cross_section_pb, cross_section_error_pb, generated_events = (
            extract_cross_section_from_banner(banner_path)
        )
        rows.append(
            {
                "process": process_match.group(1),
                "cross section pb": cross_section_pb,
                "cross section error pb": cross_section_error_pb,
                "N events (generated)": generated_events,
                "description": extract_process_description(process_dir),
            }
        )

    return rows


def write_xsec_csv(output_path: Path, rows: list[dict[str, str]]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "process",
        "cross section pb",
        "cross section error pb",
        "N events (generated)",
        "description",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path.resolve()


def main() -> int:
    args = parse_args()
    process_dirs = discover_process_directories(args.mg5_bin.resolve())
    card_text = build_mg5_card(
        process_dirs,
        args.nevents,
        args.mass_pdg,
        args.mass,
        args.tanphi_index,
        args.tanphi,
        args.ebeam1,
        args.ebeam2,
    )
    output_path = write_card(args.output, card_text)

    print(
        f"Wrote {output_path} for {len(process_dirs)} processes "
        f"with {args.nevents} events each."
    )
    for process_dir in process_dirs:
        print(process_dir.name)

    if args.write_only:
        print("Skipping MadGraph launch because --write-only was requested.")
        return 0

    print(f"Launching MadGraph with {output_path}...")
    returncode = run_madgraph(args.mg5_bin.resolve(), output_path)
    if returncode != 0:
        print(f"MadGraph finished with a non-zero exit code: {returncode}")
        return returncode

    xsec_rows = build_xsec_rows(process_dirs)
    xsec_output_path = write_xsec_csv(args.xsec_output, xsec_rows)

    print("MadGraph finished successfully.")
    print(f"Wrote cross section summary to {xsec_output_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
