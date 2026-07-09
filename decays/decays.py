#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
from pathlib import Path


DEFAULT_MG5_BIN = Path(
    os.environ.get("MG5_BIN_DIR", "/home/patricio/Documents/mg5amcnlo-3.x/bin")
)
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("tmp_run.mg5")
NWA_THRESHOLD = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a simple MadGraph scan over all proc-* directories and extract "
            "the decay information for pseudoscalar PDG 36."
        )
    )
    parser.add_argument(
        "--nevents",
        type=int,
        default=2,
        help="Number of events to request in each process. Default: 2",
    )
    parser.add_argument(
        "--mass-pdg",
        type=int,
        default=36,
        help="PDG code whose mass entry will be updated. Default: 36",
    )
    parser.add_argument(
        "--tanphi-index",
        type=int,
        default=1,
        help="Index used in the TANPHI block. Default: 1",
    )
    parser.add_argument(
        "--width",
        default="Auto",
        help="Width value for 'set width 36 <value>'. Default: Auto",
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
        "--process-name",
        default="proc-decays",
        help="MadGraph process directory to launch. Default: proc-decays",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination .mg5 card. Default: decays/tmp_run.mg5",
    )
    parser.add_argument(
        "--decays-output",
        type=Path,
        default=None,
        help="Optional destination CSV file for the decay scan. Default: decays/decays.csv",
    )
    parser.add_argument(
        "--narrow-width-output",
        type=Path,
        default=None,
        help=(
            "Optional destination CSV file for the width/mass = 0.1 boundary curves. "
            "Default: decays/narrow-width.csv"
        ),
    )
    parser.add_argument(
        "--mass-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(100.0, 1000.0),
        help="Mass scan range as: --mass-range MIN MAX. Default: 100 1000",
    )
    parser.add_argument(
        "--mass-points",
        type=int,
        default=3,
        help="Number of mass points in the scan. Default: 3",
    )
    parser.add_argument(
        "--tanphi-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.1, 100.0),
        help="Tanphi scan range as: --tanphi-range MIN MAX. Default: 0.1 100",
    )
    parser.add_argument(
        "--tanphi-points",
        type=int,
        default=3,
        help="Number of tanphi points in the log scan. Default: 3",
    )
    parser.add_argument(
        "--write-only",
        action="store_true",
        help="Only write the .mg5 card without launching MadGraph.",
    )
    args = parser.parse_args()

    if args.nevents <= 0:
        parser.error("--nevents must be a positive integer.")
    if args.mass_points <= 0:
        parser.error("--mass-points must be a positive integer.")
    if args.tanphi_points <= 0:
        parser.error("--tanphi-points must be a positive integer.")
    args.mass_min, args.mass_max = args.mass_range
    args.tanphi_min, args.tanphi_max = args.tanphi_range
    if args.mass_min <= 0 or args.mass_max <= 0:
        parser.error("--mass-range values must be positive.")
    if args.tanphi_min <= 0 or args.tanphi_max <= 0:
        parser.error("--tanphi-range values must be positive.")
    if args.mass_min > args.mass_max:
        parser.error("--mass-range requires MIN <= MAX.")
    if args.tanphi_min > args.tanphi_max:
        parser.error("--tanphi-range requires MIN <= MAX.")
    if args.output.suffix != ".mg5":
        parser.error("--output must end with .mg5.")
    if args.decays_output is None:
        args.decays_output = build_default_decays_output()
    if args.decays_output.suffix != ".csv":
        parser.error("--decays-output must end with .csv.")
    if args.narrow_width_output is None:
        args.narrow_width_output = build_default_narrow_width_output()
    if args.narrow_width_output.suffix != ".csv":
        parser.error("--narrow-width-output must end with .csv.")

    return args


def build_default_decays_output() -> Path:
    return Path(__file__).resolve().with_name("decays.csv")


def build_default_narrow_width_output() -> Path:
    return Path(__file__).resolve().with_name("narrow-width.csv")


def format_scan_value(value: float) -> str:
    return f"{value:.12g}"


def linspace(start: float, stop: float, count: int) -> list[float]:
    if count == 1:
        return [start]
    step = (stop - start) / (count - 1)
    return [start + index * step for index in range(count)]


def logspace(start: float, stop: float, count: int) -> list[float]:
    if count == 1:
        return [start]
    log_start = math.log10(start)
    log_stop = math.log10(stop)
    return [10 ** value for value in linspace(log_start, log_stop, count)]


def build_process_dir(mg5_bin: Path, process_name: str = "proc-decays") -> Path:
    return mg5_bin / process_name


def validate_process_dir(process_dir: Path) -> None:
    if not process_dir.parent.exists():
        raise FileNotFoundError(f"MadGraph bin directory not found: {process_dir.parent}")
    if not process_dir.parent.is_dir():
        raise NotADirectoryError(
            f"MadGraph bin path is not a directory: {process_dir.parent}"
        )
    if not process_dir.is_dir():
        raise FileNotFoundError(f"Process directory not found: {process_dir}")
    if not (process_dir / "Cards" / "proc_card_mg5.dat").is_file():
        raise FileNotFoundError(
            f"Missing proc card in process directory: {process_dir / 'Cards' / 'proc_card_mg5.dat'}"
        )


def build_mg5_card(
    process_dir: Path,
    nevents: int,
    mass_pdg: int,
    mass: str,
    tanphi_index: int,
    tanphi: str,
    width: str,
    ebeam1: str,
    ebeam2: str,
) -> str:
    lines = [
        "set automatic_html_opening False --no_save",
        "set notification_center False --no_save",
        "",
    ]

    lines.extend(
        [
            f"launch {process_dir.name}",
            "shower=OFF",
            "detector=OFF",
            "analysis=OFF",
            f"set mass {mass_pdg} {mass}",
            f"set tanphi {tanphi_index} {tanphi}",
            f"set width 36 {width}",
            f"set nevents {nevents}",
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


def extract_latest_existing_banner(process_dir: Path) -> tuple[int, int, Path]:
    latest_banner: tuple[int, int, Path] | None = None
    for banner_path in process_dir.glob("Events/run_*/run_*_tag_*_banner.txt"):
        match = re.fullmatch(r"run_(\d+)_tag_(\d+)_banner\.txt", banner_path.name)
        if match is None:
            continue

        candidate = (int(match.group(1)), int(match.group(2)), banner_path)
        if latest_banner is None or candidate[:2] > latest_banner[:2]:
            latest_banner = candidate

    if latest_banner is None:
        raise FileNotFoundError(f"No banner files found in {process_dir / 'Events'}")

    return latest_banner


def wait_for_new_banner(
    process_dir: Path,
    previous_banner: tuple[int, int, Path] | None,
) -> Path:
    latest_banner = extract_latest_existing_banner(process_dir)
    if previous_banner is not None and latest_banner[:2] <= previous_banner[:2]:
        raise RuntimeError(
            "MadGraph finished but no newer banner was found for this scan point."
        )
    return latest_banner[2]


def extract_mass_from_banner(banner_path: Path, mass_pdg: int = 36) -> str:
    content = banner_path.read_text(encoding="utf-8", errors="replace")
    mass_block_match = re.search(
        r"BLOCK MASS\b(.*?)(?:^\s*BLOCK\b|^\s*DECAY\b|^\s*</MGParamCard>)",
        content,
        flags=re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    if mass_block_match is None:
        raise ValueError(f"Could not find MASS block in {banner_path}")

    mass_match = re.search(
        rf"^\s*{mass_pdg}\s+([0-9.eE+-]+)\b",
        mass_block_match.group(1),
        flags=re.MULTILINE,
    )
    if mass_match is None:
        raise ValueError(f"Could not find mass entry for PDG {mass_pdg} in {banner_path}")

    return mass_match.group(1)


def extract_tanphi_from_banner(banner_path: Path, tanphi_index: int = 1) -> str:
    content = banner_path.read_text(encoding="utf-8", errors="replace")
    tanphi_match = re.search(
        rf"^\s*{tanphi_index}\s+([0-9.eE+-]+)\s+#\s*tanphi\b",
        content,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if tanphi_match is None:
        raise ValueError(f"Could not find tanphi entry in {banner_path}")

    return tanphi_match.group(1)


def extract_decay_block(
    banner_path: Path, pseudoscalar_pdg: int = 36
) -> tuple[str, list[tuple[str, list[str]]]]:
    content = banner_path.read_text(encoding="utf-8", errors="replace")
    decay_match = re.search(
        rf"^\s*DECAY\s+{pseudoscalar_pdg}\s+([0-9.eE+-]+)\b.*?$"
        r"(.*?)(?=^\s*DECAY\b|^\s*</slha>|\Z)",
        content,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if decay_match is None:
        raise ValueError(
            f"Could not find DECAY block for pseudoscalar PDG {pseudoscalar_pdg} in {banner_path}"
        )

    width = decay_match.group(1)
    channels: list[tuple[str, list[str]]] = []
    for raw_line in decay_match.group(2).splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 4:
            continue

        br = parts[0]
        nda = int(parts[1])
        daughters = parts[2 : 2 + nda]
        if len(daughters) != nda:
            continue
        channels.append((br, daughters))

    return width, channels


PDG_LABELS = {
    -24: "W-",
    -16: "vt~",
    -15: "tau+",
    -14: "vm~",
    -13: "mu+",
    -12: "ve~",
    -11: "e+",
    -6: "t~",
    -5: "b~",
    -4: "c~",
    -3: "s~",
    -2: "u~",
    -1: "d~",
    1: "d",
    2: "u",
    3: "s",
    4: "c",
    5: "b",
    6: "t",
    11: "e-",
    12: "ve",
    13: "mu-",
    14: "vm",
    15: "tau-",
    16: "vt",
    21: "g",
    22: "gamma",
    23: "Z",
    24: "W+",
    25: "H",
    36: "AP",
}


def format_decay_channel(daughters: list[str]) -> str:
    labels = [PDG_LABELS.get(int(pid), pid) for pid in daughters]
    return " ".join(labels)


ORDINAL_LABELS = {
    1: "primary",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
}


def ordinal_label(index: int) -> str:
    return ORDINAL_LABELS.get(index, f"decay {index}")


def build_decay_summary_row(
    banner_path: Path,
    mass_pdg: int,
    tanphi_index: int,
) -> tuple[list[str], dict[str, str]]:
    mass = extract_mass_from_banner(banner_path, mass_pdg)
    tanphi = extract_tanphi_from_banner(banner_path, tanphi_index)
    width, channels = extract_decay_block(banner_path, mass_pdg)

    mass_value = float(mass)
    width_value = float(width)
    ordered_channels = sorted(channels, key=lambda item: float(item[0]), reverse=True)

    headers = ["mass", "tanphi", "width", "total width/mass"]
    row: dict[str, str] = {
        "mass": mass,
        "tanphi": tanphi,
        "width": width,
        "total width/mass": f"{width_value / mass_value:.12g}",
    }

    for index, (branching_ratio, daughters) in enumerate(ordered_channels[:4], start=1):
        label = ordinal_label(index)
        channel_key = f"{label} decay channel"
        branching_key = f"{label} decay branching"
        headers.extend([channel_key, branching_key])
        row[channel_key] = format_decay_channel(daughters)
        row[branching_key] = branching_ratio
    return headers, row


def merge_headers(existing_headers: list[str], new_headers: list[str]) -> list[str]:
    merged = list(existing_headers)
    for header in new_headers:
        if header not in merged:
            merged.append(header)
    return merged


def read_existing_rows(output_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not output_path.exists():
        return [], []

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return list(reader.fieldnames or []), rows


def write_decay_summary(
    output_path: Path,
    headers: list[str],
    rows: list[dict[str, str]],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path.resolve()


def extract_narrow_width_rows(
    rows: list[dict[str, str]],
    threshold: float = NWA_THRESHOLD,
) -> list[dict[str, str]]:
    grouped_points: dict[float, list[tuple[float, float]]] = {}
    for row in rows:
        if not row.get("mass") or not row.get("tanphi") or not row.get("total width/mass"):
            continue
        mass = float(row["mass"])
        tanphi = float(row["tanphi"])
        width_ratio = float(row["total width/mass"])
        grouped_points.setdefault(mass, []).append((tanphi, width_ratio))

    boundary_rows: list[dict[str, str]] = []
    for mass in sorted(grouped_points):
        points = sorted(grouped_points[mass], key=lambda item: item[0])
        crossings: list[float] = []
        for index in range(len(points) - 1):
            tanphi_1, width_ratio_1 = points[index]
            tanphi_2, width_ratio_2 = points[index + 1]
            delta_1 = width_ratio_1 - threshold
            delta_2 = width_ratio_2 - threshold

            if delta_1 == 0.0:
                crossings.append(tanphi_1)
            if delta_1 * delta_2 < 0.0:
                crossing_tanphi = tanphi_1 + (
                    (threshold - width_ratio_1) * (tanphi_2 - tanphi_1)
                    / (width_ratio_2 - width_ratio_1)
                )
                crossings.append(crossing_tanphi)
        if points and points[-1][1] == threshold:
            crossings.append(points[-1][0])

        unique_crossings: list[float] = []
        for value in crossings:
            if not unique_crossings or not math.isclose(
                value, unique_crossings[-1], rel_tol=1e-9, abs_tol=1e-12
            ):
                unique_crossings.append(value)

        low_branch = unique_crossings[0] if unique_crossings else None
        high_branch = unique_crossings[1] if len(unique_crossings) > 1 else None
        boundary_rows.append(
            {
                "mass": format_scan_value(mass),
                "tanphi_low": "" if low_branch is None else format_scan_value(low_branch),
                "tanphi_high": "" if high_branch is None else format_scan_value(high_branch),
            }
        )

    return boundary_rows


def write_narrow_width_summary(
    output_path: Path,
    rows: list[dict[str, str]],
) -> Path:
    headers = ["mass", "tanphi_low", "tanphi_high"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path.resolve()


def main() -> int:
    args = parse_args()
    process_dir = build_process_dir(args.mg5_bin, args.process_name)

    mass_values = linspace(args.mass_min, args.mass_max, args.mass_points)
    tanphi_values = logspace(args.tanphi_min, args.tanphi_max, args.tanphi_points)

    existing_headers, existing_rows = read_existing_rows(args.decays_output)
    all_headers = list(existing_headers)
    all_rows = list(existing_rows)

    if all_rows:
        narrow_width_rows = extract_narrow_width_rows(all_rows)
        narrow_width_output = write_narrow_width_summary(
            args.narrow_width_output,
            narrow_width_rows,
        )
        print(f"Updated narrow-width summary at {narrow_width_output}")

    if args.write_only:
        preview_mass = format_scan_value(mass_values[0])
        preview_tanphi = format_scan_value(tanphi_values[0])
        card_text = build_mg5_card(
            process_dir=process_dir,
            nevents=args.nevents,
            mass_pdg=args.mass_pdg,
            mass=preview_mass,
            tanphi_index=args.tanphi_index,
            tanphi=preview_tanphi,
            width=args.width,
            ebeam1=args.ebeam1,
            ebeam2=args.ebeam2,
        )
        card_path = write_card(args.output, card_text)
        print(f"Wrote preview MadGraph card to {card_path}")
        return 0

    validate_process_dir(process_dir)

    for mass_value in mass_values:
        for tanphi_value in tanphi_values:
            mass = format_scan_value(mass_value)
            tanphi = format_scan_value(tanphi_value)
            previous_banner: tuple[int, int, Path] | None = None
            try:
                previous_banner = extract_latest_existing_banner(process_dir)
            except FileNotFoundError:
                previous_banner = None

            card_text = build_mg5_card(
                process_dir=process_dir,
                nevents=args.nevents,
                mass_pdg=args.mass_pdg,
                mass=mass,
                tanphi_index=args.tanphi_index,
                tanphi=tanphi,
                width=args.width,
                ebeam1=args.ebeam1,
                ebeam2=args.ebeam2,
            )
            card_path = write_card(args.output, card_text)
            print(f"Running point mass={mass}, tanphi={tanphi} with card {card_path}")

            return_code = run_madgraph(args.mg5_bin, card_path)
            if return_code != 0:
                print(f"MadGraph exited with code {return_code} for mass={mass}, tanphi={tanphi}")
                return return_code

            banner_path = wait_for_new_banner(process_dir, previous_banner)
            headers, row = build_decay_summary_row(
                banner_path=banner_path,
                mass_pdg=args.mass_pdg,
                tanphi_index=args.tanphi_index,
            )
            all_headers = merge_headers(all_headers, headers)
            all_rows.append(row)
            decays_output = write_decay_summary(args.decays_output, all_headers, all_rows)
            narrow_width_rows = extract_narrow_width_rows(all_rows)
            narrow_width_output = write_narrow_width_summary(
                args.narrow_width_output,
                narrow_width_rows,
            )
            print(f"Saved row to {decays_output}")
            print(f"Updated narrow-width summary at {narrow_width_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
