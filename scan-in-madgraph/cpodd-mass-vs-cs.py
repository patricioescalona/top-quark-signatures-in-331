from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_MG5_BIN = Path(
    os.environ.get("MG5_BIN_DIR", "/home/patricio/Documents/mg5amcnlo-3.x/bin")
)
DEFAULT_PROCESS = "varI-BM3-tt"
DEFAULT_PDG = 36


@dataclass
class ScanResult:
    mass: str
    run_name: str | None
    cross_section_pb: float | None
    cross_section_error_pb: float | None
    status: str
    return_code: int
    note: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a mass parameter in an already-generated local MadGraph process "
            "directory and save only the final summary table in this repository."
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
        "--pdg",
        type=int,
        default=DEFAULT_PDG,
        help="PDG code of the particle whose mass will be scanned.",
    )
    parser.add_argument(
        "--output-name",
        default="top-pseudoscalar-scan",
        help="Base name for the summary file written in results/.",
    )
    parser.add_argument(
        "--output-format",
        choices=("csv", "tsv", "json"),
        default="csv",
        help="Format for the summary file.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory where the summary file will be saved.",
    )
    parser.add_argument(
        "--masses",
        nargs="+",
        help="Explicit list of masses, for example: --masses 500 750 1000",
    )
    parser.add_argument(
        "--mass-start",
        default="500",
        help="Starting mass for a regular scan grid.",
    )
    parser.add_argument(
        "--mass-stop",
        default="1500",
        help="Final mass for a regular scan grid.",
    )
    parser.add_argument(
        "--mass-step",
        default="500",
        help="Mass spacing for a regular scan grid.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved configuration without launching MadGraph.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the scan at the first failed mass point.",
    )
    return parser.parse_args()


def parse_decimal(raw: str) -> Decimal:
    try:
        return Decimal(str(raw))
    except InvalidOperation as exc:
        raise ValueError(f"Invalid mass value: {raw}") from exc


def format_decimal(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def resolve_masses(args: argparse.Namespace) -> list[Decimal]:
    if args.masses:
        masses = [parse_decimal(raw) for raw in args.masses]
    else:
        start = parse_decimal(args.mass_start)
        stop = parse_decimal(args.mass_stop)
        step = parse_decimal(args.mass_step)

        if step <= 0:
            raise ValueError("--mass-step must be positive.")
        if stop < start:
            raise ValueError("--mass-stop must be greater than or equal to --mass-start.")

        masses = []
        current = start
        epsilon = step / Decimal("1000000")
        while current <= stop + epsilon:
            masses.append(current)
            current += step

    if not masses:
        raise ValueError("No mass points were provided.")
    return masses


def normalize_output_stem(name: str, output_format: str) -> str:
    stem = Path(name).name
    suffix = f".{output_format}"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem or "scan"


def list_run_directories(events_dir: Path) -> set[str]:
    if not events_dir.exists():
        return set()
    return {path.name for path in events_dir.glob("run_*") if path.is_dir()}


def run_index(run_name: str) -> int:
    try:
        return int(run_name.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def detect_new_run(before: set[str], after: set[str]) -> str | None:
    new_runs = sorted(after - before, key=run_index)
    return new_runs[-1] if new_runs else None


def mtime_ns(path: Path) -> int | None:
    if not path.exists():
        return None
    return path.stat().st_mtime_ns


def build_card(process: str, pdg: int, mass: str) -> str:
    return f"""set automatic_html_opening False --no_save
launch {process}
set mass {pdg} {mass}
done
"""


def run_mg5(mg5_bin: Path, process: str, pdg: int, mass: str) -> subprocess.CompletedProcess[str]:
    mg5_executable = mg5_bin / "mg5_aMC"
    with tempfile.TemporaryDirectory(prefix="mg5-scan-") as tmpdir:
        card_path = Path(tmpdir) / f"{process}_{mass}.mg5"
        card_path.write_text(build_card(process, pdg, mass), encoding="utf-8")
        return subprocess.run(
            [str(mg5_executable), str(card_path)],
            cwd=str(mg5_bin),
            capture_output=True,
            text=True,
            check=False,
        )


def parse_results_dat(results_dat: Path) -> tuple[float | None, float | None]:
    if not results_dat.exists():
        return None, None

    lines = results_dat.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return None, None

    fields = lines[0].split()
    if len(fields) < 2:
        return None, None

    return float(fields[0]), float(fields[1])


def parse_results_html(results_html: Path) -> tuple[float | None, float | None]:
    if not results_html.exists():
        return None, None

    text = results_html.read_text(encoding="utf-8", errors="replace")
    marker = "<b>s="
    start = text.find(marker)
    if start == -1:
        return None, None

    snippet = text[start : start + 200]
    snippet = snippet.replace("&#177;", "±").replace("&#177", "±")
    snippet = snippet.replace("</b>", "")
    snippet = snippet.replace("<b>", "")
    snippet = snippet.replace("(pb)", "")
    snippet = snippet.split("<", 1)[0]
    snippet = snippet.replace("s=", "").strip()

    if "±" not in snippet:
        return None, None

    cross_section, error = [part.strip() for part in snippet.split("±", 1)]
    return float(cross_section), float(error)


def write_summary(
    results: list[ScanResult], output_path: Path, output_format: str
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]

    if output_format == "json":
        output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return

    delimiter = "," if output_format == "csv" else "\t"
    fieldnames = list(ScanResult.__dataclass_fields__.keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def validate_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    mg5_bin = args.mg5_bin.expanduser().resolve()
    mg5_executable = mg5_bin / "mg5_aMC"
    process_dir = mg5_bin / args.process

    if not mg5_executable.exists():
        raise FileNotFoundError(f"MadGraph executable not found: {mg5_executable}")
    if not process_dir.exists():
        raise FileNotFoundError(
            "Generated process directory not found: "
            f"{process_dir}\nCreate it locally in MadGraph first, then rerun the scan."
        )

    return mg5_bin, process_dir


def main() -> int:
    args = parse_args()

    try:
        masses = resolve_masses(args)
        mg5_bin, process_dir = validate_paths(args)
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return 1

    output_stem = normalize_output_stem(args.output_name, args.output_format)
    results_dir = args.results_dir.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / f"{output_stem}.{args.output_format}"

    if args.dry_run:
        print(f"MadGraph bin : {mg5_bin}")
        print(f"Process dir  : {process_dir}")
        print(f"Results file : {summary_path}")
        print(f"Mass points  : {', '.join(format_decimal(mass) for mass in masses)}")
        return 0

    events_dir = process_dir / "Events"
    results_dat = process_dir / "SubProcesses" / "results.dat"
    results: list[ScanResult] = []

    for mass in masses:
        mass_text = format_decimal(mass)
        before_runs = list_run_directories(events_dir)
        results_before = mtime_ns(results_dat)

        completed = run_mg5(mg5_bin, args.process, args.pdg, mass_text)

        after_runs = list_run_directories(events_dir)
        run_name = detect_new_run(before_runs, after_runs)

        cross_section = None
        cross_section_error = None

        if mtime_ns(results_dat) != results_before:
            cross_section, cross_section_error = parse_results_dat(results_dat)

        if (cross_section is None or cross_section_error is None) and run_name:
            html_results = process_dir / "HTML" / run_name / "results.html"
            cross_section, cross_section_error = parse_results_html(html_results)

        notes = []
        if run_name is None:
            notes.append("No new run_* directory was detected.")
        if completed.returncode != 0:
            notes.append("MadGraph exited with a non-zero return code.")
        if cross_section is None or cross_section_error is None:
            notes.append("Could not parse the cross section and its error.")

        status = "ok"
        if completed.returncode != 0:
            status = "failed"
        elif cross_section is None or cross_section_error is None:
            status = "incomplete"

        result = ScanResult(
            mass=mass_text,
            run_name=run_name,
            cross_section_pb=cross_section,
            cross_section_error_pb=cross_section_error,
            status=status,
            return_code=completed.returncode,
            note=" ".join(notes) if notes else None,
        )
        results.append(result)
        write_summary(results, summary_path, args.output_format)

        if cross_section is not None and cross_section_error is not None:
            print(
                f"m={mass_text} -> xs={cross_section} pb +/- {cross_section_error} pb"
                f" [{status}]"
            )
        else:
            print(f"m={mass_text} -> xs unavailable [{status}]")

        if status != "ok":
            stdout_tail = completed.stdout.strip().splitlines()[-5:]
            stderr_tail = completed.stderr.strip().splitlines()[-5:]
            for line in stdout_tail + stderr_tail:
                if line.strip():
                    print(line)

        if args.stop_on_error and status != "ok":
            break

    print(f"Saved summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
