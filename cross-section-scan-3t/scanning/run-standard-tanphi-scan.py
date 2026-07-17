from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCAN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = SCAN_ROOT / "results"
DEFAULT_MG5_BIN = Path(
    os.environ.get("MG5_BIN_DIR", "/home/patricio/Documents/mg5amcnlo-3.x/bin")
)
DEFAULT_PROCESS = "xs-signal-bsm-pp-tttbar"
DEFAULT_SCAN_SCRIPT = Path(__file__).with_name("cpodd-mass-vs-cs.py")
DEFAULT_TANPHI_GRID = [
    ("001", "0.01"),
    ("01", "0.1"),
    ("1", "1"),
    ("10", "10"),
    ("100", "100"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the standard tanphi scan grid for the local pp -> tt t~ process "
            "and save one output table per tanphi value."
        )
    )
    parser.add_argument(
        "--mg5-bin",
        type=Path,
        default=DEFAULT_MG5_BIN,
        help="Path to the local MadGraph bin directory.",
    )
    parser.add_argument(
        "--process",
        default=DEFAULT_PROCESS,
        help="Generated process directory name inside the MadGraph bin directory.",
    )
    parser.add_argument(
        "--scan-script",
        type=Path,
        default=DEFAULT_SCAN_SCRIPT,
        help="Path to the single-scan script.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory where the summary files will be saved.",
    )
    parser.add_argument(
        "--output-prefix",
        default="bsm-tttbar",
        help="Prefix used to build output names like <prefix>-tanphi-001.csv.",
    )
    parser.add_argument(
        "--pdg",
        type=int,
        default=36,
        help="PDG code of the particle whose mass will be scanned.",
    )
    parser.add_argument(
        "--ebeam1",
        default="7000",
        help="Beam-1 energy in GeV written into the run card.",
    )
    parser.add_argument(
        "--ebeam2",
        default="7000",
        help="Beam-2 energy in GeV written into the run card.",
    )
    parser.add_argument(
        "--nevents",
        type=int,
        default=10000,
        help="Number of events requested from MadGraph at each scan point.",
    )
    parser.add_argument(
        "--output-format",
        choices=("csv", "tsv", "json"),
        default="csv",
        help="Format for the summary files.",
    )
    parser.add_argument(
        "--masses",
        nargs="+",
        help="Explicit list of masses, for example: --masses 200 400 600",
    )
    parser.add_argument(
        "--mass-start",
        default="200",
        help="Starting mass for a regular scan grid.",
    )
    parser.add_argument(
        "--mass-stop",
        default="1400",
        help="Final mass for a regular scan grid.",
    )
    parser.add_argument(
        "--mass-step",
        default="100",
        help="Mass spacing for a regular scan grid.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without launching MadGraph.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first failed tanphi scan.",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace, tanphi_token: str, tanphi_value: str) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(args.scan_script.expanduser().resolve()),
        "--mg5-bin",
        str(args.mg5_bin.expanduser().resolve()),
        "--process",
        args.process,
        "--pdg",
        str(args.pdg),
        "--tanphi",
        tanphi_value,
        "--ebeam1",
        args.ebeam1,
        "--ebeam2",
        args.ebeam2,
        "--nevents",
        str(args.nevents),
        "--output-name",
        f"{args.output_prefix}-tanphi-{tanphi_token}",
        "--output-format",
        args.output_format,
        "--results-dir",
        str(args.results_dir.expanduser().resolve()),
    ]

    if args.masses:
        command.extend(["--masses", *args.masses])
    else:
        command.extend(
            [
                "--mass-start",
                args.mass_start,
                "--mass-stop",
                args.mass_stop,
                "--mass-step",
                args.mass_step,
            ]
        )

    if args.dry_run:
        command.append("--dry-run")
    if args.stop_on_error:
        command.append("--stop-on-error")

    return command


def main() -> int:
    args = parse_args()

    for tanphi_token, tanphi_value in DEFAULT_TANPHI_GRID:
        command = build_command(args, tanphi_token, tanphi_value)
        print(
            f"Running tanphi={tanphi_value} -> {args.output_prefix}-tanphi-{tanphi_token}",
            flush=True,
        )
        print(" ".join(command), flush=True)

        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            print(
                f"Scan failed for tanphi={tanphi_value} with return code {completed.returncode}",
                flush=True,
            )
            if args.stop_on_error:
                return completed.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
