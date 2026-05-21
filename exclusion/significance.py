#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
GENERATED_DIR_PATTERN = re.compile(r"^generated-m-(.+)-tanphi-(.+)$")
DEFAULT_BACKGROUND_DIR = BASE_DIR / "generated-background"
DEFAULT_LUMINOSITY_FB = 130.0
STAGE_LABELS = [
    "Before cuts",
    "After cut I",
    "After cut II",
    "After cut III",
    "After cut IV",
]


@dataclass
class StageSummary:
    label: str
    cross_section_pb: float
    yield_events: float
    efficiency_percent: float


@dataclass
class SampleSummary:
    label: str
    generated_dir: Path
    stages: list[StageSummary]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-signal Asimov significances against the generated background "
            "and write significance.txt inside each generated signal folder."
        )
    )
    parser.add_argument(
        "--mass",
        help="Mass label used in generated-m-<mass>-tanphi-<tanphi>.",
    )
    parser.add_argument(
        "--tanphi",
        help="Tanphi label used in generated-m-<mass>-tanphi-<tanphi>.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_DIR,
        help="Base exclusion directory. Default: exclusion/",
    )
    parser.add_argument(
        "--background-dir",
        type=Path,
        default=DEFAULT_BACKGROUND_DIR,
        help="Background directory containing xsec-background.csv and efficiencies-background.csv.",
    )
    parser.add_argument(
        "--luminosity",
        type=float,
        default=DEFAULT_LUMINOSITY_FB,
        help="Integrated luminosity in fb^-1 used for the Asimov significance.",
    )
    args = parser.parse_args()

    if (args.mass is None) != (args.tanphi is None):
        parser.error("Use both --mass and --tanphi together, or neither of them.")
    if args.luminosity <= 0.0:
        parser.error("--luminosity must be positive.")

    return args


def format_value_for_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    sanitized = sanitized.strip("-")
    return sanitized or "value"


def build_generated_dir(base_dir: Path, mass: str, tanphi: str) -> Path:
    return base_dir / (
        f"generated-m-{format_value_for_filename(mass)}"
        f"-tanphi-{format_value_for_filename(tanphi)}"
    )


def build_xsec_csv_path(generated_dir: Path, mass: str, tanphi: str) -> Path:
    return generated_dir / (
        f"xsec-m-{format_value_for_filename(mass)}"
        f"-tanphi-{format_value_for_filename(tanphi)}.csv"
    )


def build_efficiencies_csv_path(generated_dir: Path, mass: str, tanphi: str) -> Path:
    return generated_dir / (
        f"efficiencies-m-{format_value_for_filename(mass)}"
        f"-tanphi-{format_value_for_filename(tanphi)}.csv"
    )


def build_significance_output_path(generated_dir: Path) -> Path:
    return generated_dir / "significance.txt"


def discover_generated_dirs(base_dir: Path) -> list[tuple[str, str, Path]]:
    generated_dirs: list[tuple[str, str, Path]] = []
    for path in sorted(base_dir.glob("generated-m-*-tanphi-*")):
        if not path.is_dir():
            continue
        match = GENERATED_DIR_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        generated_dirs.append((match.group(1), match.group(2), path.resolve()))

    if not generated_dirs:
        raise FileNotFoundError(f"No generated signal folders found in {base_dir}")

    return generated_dirs


def load_total_cross_section_pb(xsec_csv_path: Path) -> float:
    if not xsec_csv_path.is_file():
        raise FileNotFoundError(f"Cross-section CSV not found: {xsec_csv_path}")

    total_cross_section_pb = 0.0
    with xsec_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total_cross_section_pb += float(row["cross section pb"])

    if total_cross_section_pb <= 0.0:
        raise ValueError(f"Total cross section must be positive in {xsec_csv_path}")

    return total_cross_section_pb


def load_efficiency_rows(efficiencies_csv_path: Path) -> list[dict[str, float | str]]:
    if not efficiencies_csv_path.is_file():
        raise FileNotFoundError(f"Efficiencies CSV not found: {efficiencies_csv_path}")

    rows: list[dict[str, float | str]] = []
    with efficiencies_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "cut": row["cut"],
                    "efficiency": float(row["efficiency"]),
                    "cross_section_pb": float(row["cross_section_pb"]),
                }
            )

    expected_cuts = ["I", "II", "III", "IV"]
    found_cuts = [str(row["cut"]) for row in rows]
    if found_cuts != expected_cuts:
        raise ValueError(
            f"Expected cuts {expected_cuts} in {efficiencies_csv_path}, found {found_cuts}."
        )

    return rows


def build_sample_summary(
    label: str,
    generated_dir: Path,
    total_cross_section_pb: float,
    efficiencies_rows: list[dict[str, float | str]],
    luminosity_fb: float,
) -> SampleSummary:
    stages = [
        StageSummary(
            label=STAGE_LABELS[0],
            cross_section_pb=total_cross_section_pb,
            yield_events=total_cross_section_pb * luminosity_fb * 1000.0,
            efficiency_percent=100.0,
        )
    ]

    for stage_label, row in zip(STAGE_LABELS[1:], efficiencies_rows):
        cross_section_pb = float(row["cross_section_pb"])
        efficiency = float(row["efficiency"]) * 100.0
        stages.append(
            StageSummary(
                label=stage_label,
                cross_section_pb=cross_section_pb,
                yield_events=cross_section_pb * luminosity_fb * 1000.0,
                efficiency_percent=efficiency,
            )
        )

    return SampleSummary(label=label, generated_dir=generated_dir, stages=stages)


def load_signal_sample(
    mass: str,
    tanphi: str,
    generated_dir: Path,
    luminosity_fb: float,
) -> SampleSummary:
    return build_sample_summary(
        label=f"m={mass}, tanphi={tanphi}",
        generated_dir=generated_dir,
        total_cross_section_pb=load_total_cross_section_pb(
            build_xsec_csv_path(generated_dir, mass, tanphi)
        ),
        efficiencies_rows=load_efficiency_rows(
            build_efficiencies_csv_path(generated_dir, mass, tanphi)
        ),
        luminosity_fb=luminosity_fb,
    )


def load_background_sample(background_dir: Path, luminosity_fb: float) -> SampleSummary:
    return build_sample_summary(
        label="background",
        generated_dir=background_dir,
        total_cross_section_pb=load_total_cross_section_pb(background_dir / "xsec-background.csv"),
        efficiencies_rows=load_efficiency_rows(background_dir / "efficiencies-background.csv"),
        luminosity_fb=luminosity_fb,
    )


def asimov_significance(signal_events: float, background_events: float) -> float:
    if signal_events <= 0.0:
        return 0.0
    if background_events <= 0.0:
        return math.inf

    radicand = 2.0 * (
        (signal_events + background_events) * math.log1p(signal_events / background_events)
        - signal_events
    )
    if radicand < 0.0 and abs(radicand) < 1e-12:
        radicand = 0.0

    return math.sqrt(radicand)


def format_significance(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.6f}"


def format_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    columns = list(zip(*([headers] + rows)))
    widths = [max(len(str(value)) for value in column) for column in columns]

    def format_row(row: tuple[str, ...]) -> str:
        return " | ".join(str(value).ljust(width) for value, width in zip(row, widths))

    lines = [format_row(headers), "-+-".join("-" * width for width in widths)]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def build_significance_text(
    signal_sample: SampleSummary,
    background_sample: SampleSummary,
    luminosity_fb: float,
) -> str:
    efficiency_rows = [
        (
            signal_stage.label,
            f"{signal_stage.efficiency_percent:.2f}%",
            f"{background_stage.efficiency_percent:.2f}%",
        )
        for signal_stage, background_stage in zip(
            signal_sample.stages[1:], background_sample.stages[1:]
        )
    ]

    significance_rows = []
    for signal_stage, background_stage in zip(signal_sample.stages, background_sample.stages):
        significance_rows.append(
            (
                signal_stage.label,
                f"{signal_stage.yield_events:.6f}",
                f"{background_stage.yield_events:.6f}",
                format_significance(
                    asimov_significance(
                        signal_stage.yield_events,
                        background_stage.yield_events,
                    )
                ),
            )
        )

    lines = [
        f"Luminosity: {luminosity_fb:g} fb^-1",
        f"Signal sample: {signal_sample.label}",
        f"Background sample: {background_sample.label}",
        "",
        "Cut efficiencies (cumulative)",
        format_table(
            ("cut", "signal", "background"),
            efficiency_rows,
        ),
        "",
        "Asimov significance",
        format_table(
            ("stage", "signal yield [events]", "background yield [events]", "Z_A"),
            significance_rows,
        ),
        "",
    ]
    return "\n".join(lines)


def write_significance_file(
    output_path: Path,
    signal_sample: SampleSummary,
    background_sample: SampleSummary,
    luminosity_fb: float,
) -> Path:
    output_path.write_text(
        build_significance_text(signal_sample, background_sample, luminosity_fb),
        encoding="utf-8",
    )
    return output_path.resolve()


def main() -> int:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    background_dir = args.background_dir.resolve()
    background_sample = load_background_sample(background_dir, args.luminosity)

    if args.mass is not None and args.tanphi is not None:
        signal_dir = build_generated_dir(base_dir, args.mass, args.tanphi).resolve()
        signal_sample = load_signal_sample(
            args.mass, args.tanphi, signal_dir, args.luminosity
        )
        output_path = write_significance_file(
            build_significance_output_path(signal_dir),
            signal_sample,
            background_sample,
            args.luminosity,
        )
        print(f"Wrote {output_path}")
        return 0

    for mass, tanphi, generated_dir in discover_generated_dirs(base_dir):
        signal_sample = load_signal_sample(mass, tanphi, generated_dir, args.luminosity)
        output_path = write_significance_file(
            build_significance_output_path(generated_dir),
            signal_sample,
            background_sample,
            args.luminosity,
        )
        print(f"Wrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
