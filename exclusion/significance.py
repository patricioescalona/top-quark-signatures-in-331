#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

from generated_signal_paths import (
    build_generated_dir,
    discover_generated_dirs,
    format_value_for_filename,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_BACKGROUND_DIR = BASE_DIR / "generated-background"
DEFAULT_LUMINOSITIES_FB = (450.0, 3000.0)
BACKGROUND_EFFECTIVE_YIELDS_CSV = "effective-cross-section-and-yields.csv"
STAGE_LABELS = [
    "Before cuts",
    "After cut I",
    "After cut II",
    "After cut III",
    "After cut IV",
]
YIELD_COLUMN_PATTERN = re.compile(
    r"^yield_(?P<luminosity>[0-9]+(?:\.[0-9]+)?)_?fb$",
    flags=re.IGNORECASE,
)


@dataclass
class StageSummary:
    label: str
    cross_section_pb: float
    efficiency_percent: float
    event_yields: dict[float, float] | None = None


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
        help=(
            "Background directory containing the background tables. "
            "It supports the legacy xsec/efficiencies CSV files and the newer "
            f"{BACKGROUND_EFFECTIVE_YIELDS_CSV} summary."
        ),
    )
    parser.add_argument(
        "--luminosities",
        type=float,
        nargs="+",
        default=list(DEFAULT_LUMINOSITIES_FB),
        help=(
            "Integrated luminosities in fb^-1 used for the Asimov significance. "
            "Default: 450 3000"
        ),
    )
    args = parser.parse_args()

    if (args.mass is None) != (args.tanphi is None):
        parser.error("Use both --mass and --tanphi together, or neither of them.")
    if not args.luminosities:
        parser.error("Provide at least one luminosity.")
    if any(luminosity <= 0.0 for luminosity in args.luminosities):
        parser.error("All luminosities must be positive.")

    return args


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


def build_summary_output_path(base_dir: Path) -> Path:
    return base_dir / "significance-summary.csv"

def load_total_cross_section_pb(xsec_csv_path: Path) -> float:
    if not xsec_csv_path.is_file():
        raise FileNotFoundError(f"Cross-section CSV not found: {xsec_csv_path}")

    total_cross_section_pb = 0.0
    process_count = 0
    with xsec_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            process_count += 1
            total_cross_section_pb += float(row["cross section pb"])

    if process_count == 0:
        raise ValueError(f"No cross-section rows found in {xsec_csv_path}")
    if total_cross_section_pb <= 0.0:
        raise ValueError(f"Total cross section must be positive in {xsec_csv_path}")

    return total_cross_section_pb


def load_process_count(xsec_csv_path: Path) -> int:
    if not xsec_csv_path.is_file():
        raise FileNotFoundError(f"Cross-section CSV not found: {xsec_csv_path}")

    process_count = 0
    with xsec_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for _ in reader:
            process_count += 1

    if process_count == 0:
        raise ValueError(f"No cross-section rows found in {xsec_csv_path}")

    return process_count


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
                    "event_yields": parse_yield_columns(row),
                }
            )

    expected_cuts = ["I", "II", "III", "IV"]
    found_cuts = [str(row["cut"]) for row in rows]
    if found_cuts != expected_cuts:
        raise ValueError(
            f"Expected cuts {expected_cuts} in {efficiencies_csv_path}, found {found_cuts}."
        )

    return rows


def parse_yield_columns(row: dict[str, str]) -> dict[float, float]:
    event_yields: dict[float, float] = {}
    for column_name, raw_value in row.items():
        if raw_value in (None, ""):
            continue

        match = YIELD_COLUMN_PATTERN.fullmatch(column_name.strip())
        if match is None:
            continue

        event_yields[float(match.group("luminosity"))] = float(raw_value)

    return event_yields


def build_sample_summary(
    label: str,
    generated_dir: Path,
    total_cross_section_pb: float,
    efficiencies_rows: list[dict[str, float | str]],
) -> SampleSummary:
    stages = [
        StageSummary(
            label=STAGE_LABELS[0],
            cross_section_pb=total_cross_section_pb,
            efficiency_percent=100.0,
            event_yields=None,
        )
    ]

    for stage_label, row in zip(STAGE_LABELS[1:], efficiencies_rows):
        cross_section_pb = float(row["cross_section_pb"])
        efficiency = float(row["efficiency"]) * 100.0
        event_yields = {
            float(luminosity_fb): float(yield_value)
            for luminosity_fb, yield_value in dict(row["event_yields"]).items()
        }
        stages.append(
            StageSummary(
                label=stage_label,
                cross_section_pb=cross_section_pb,
                efficiency_percent=efficiency,
                event_yields=event_yields or None,
            )
        )

    return SampleSummary(label=label, generated_dir=generated_dir, stages=stages)


def load_signal_sample(
    mass: str,
    tanphi: str,
    generated_dir: Path,
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
    )


def load_background_sample(background_dir: Path) -> SampleSummary:
    xsec_csv_path = background_dir / "xsec-background.csv"
    efficiencies_csv_path = background_dir / "efficiencies-background.csv"
    effective_yields_csv_path = background_dir / BACKGROUND_EFFECTIVE_YIELDS_CSV
    process_count = load_process_count(xsec_csv_path)

    sample = build_sample_summary(
        label=f"background ({process_count} processes)",
        generated_dir=background_dir,
        total_cross_section_pb=load_total_cross_section_pb(xsec_csv_path),
        efficiencies_rows=load_efficiency_rows(efficiencies_csv_path),
    )

    if effective_yields_csv_path.is_file():
        total_effective_cross_section_pb, total_event_yields = (
            load_total_effective_background_data(effective_yields_csv_path)
        )
        sample.stages[-1] = StageSummary(
            label=sample.stages[-1].label,
            cross_section_pb=total_effective_cross_section_pb,
            efficiency_percent=sample.stages[-1].efficiency_percent,
            event_yields=total_event_yields or None,
        )

    return sample


def load_total_effective_background_data(
    effective_yields_csv_path: Path,
) -> tuple[float, dict[float, float]]:
    total_cross_section_pb = 0.0
    total_event_yields: dict[float, float] = {}
    process_count = 0

    with effective_yields_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            process_count += 1
            total_cross_section_pb += float(row["xsec_pb"])
            for luminosity_fb, yield_value in parse_yield_columns(row).items():
                total_event_yields[luminosity_fb] = (
                    total_event_yields.get(luminosity_fb, 0.0) + yield_value
                )

    if process_count == 0:
        raise ValueError(f"No background rows found in {effective_yields_csv_path}")
    if total_cross_section_pb <= 0.0:
        raise ValueError(
            f"Total effective background cross section must be positive in {effective_yields_csv_path}"
        )

    return total_cross_section_pb, total_event_yields


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


def format_luminosity_label(luminosity_fb: float) -> str:
    return f"{luminosity_fb:g} fb^-1"


def yield_from_cross_section(cross_section_pb: float, luminosity_fb: float) -> float:
    return cross_section_pb * luminosity_fb * 1000.0


def format_efficiency_value(value: float) -> str:
    return f"{value:.6e}"


def stage_yield(stage: StageSummary, luminosity_fb: float) -> float:
    if stage.event_yields is not None and luminosity_fb in stage.event_yields:
        return stage.event_yields[luminosity_fb]
    return yield_from_cross_section(stage.cross_section_pb, luminosity_fb)


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
    luminosities_fb: list[float],
) -> str:
    efficiency_rows = [
        (
            signal_stage.label,
            format_efficiency_value(signal_stage.efficiency_percent / 100.0),
            format_efficiency_value(background_stage.efficiency_percent / 100.0),
        )
        for signal_stage, background_stage in zip(
            signal_sample.stages[1:], background_sample.stages[1:]
        )
    ]

    lines = [
        "Luminosities: " + ", ".join(format_luminosity_label(luminosity) for luminosity in luminosities_fb),
        f"Signal sample: {signal_sample.label}",
        f"Background sample: {background_sample.label}",
        "",
        "Cut efficiencies (cumulative)",
        format_table(
            ("cut", "signal", "background"),
            efficiency_rows,
        ),
        "",
    ]

    for luminosity_fb in luminosities_fb:
        significance_rows = []
        for signal_stage, background_stage in zip(signal_sample.stages, background_sample.stages):
            signal_yield = stage_yield(signal_stage, luminosity_fb)
            background_yield = stage_yield(background_stage, luminosity_fb)
            significance_rows.append(
                (
                    signal_stage.label,
                    f"{signal_yield:.6f}",
                    f"{background_yield:.6f}",
                    format_significance(
                        asimov_significance(
                            signal_yield,
                            background_yield,
                        )
                    ),
                )
            )

        lines.extend(
            [
                f"Asimov significance ({format_luminosity_label(luminosity_fb)})",
                format_table(
                    ("stage", "signal yield [events]", "background yield [events]", "Z_A"),
                    significance_rows,
                ),
                "",
            ]
        )

    return "\n".join(lines)


def write_significance_file(
    output_path: Path,
    signal_sample: SampleSummary,
    background_sample: SampleSummary,
    luminosities_fb: list[float],
) -> Path:
    output_path.write_text(
        build_significance_text(signal_sample, background_sample, luminosities_fb),
        encoding="utf-8",
    )
    return output_path.resolve()


def build_summary_table(
    rows: list[tuple[str, str, SampleSummary, SampleSummary]],
    luminosities_fb: list[float],
) -> list[list[str]]:
    headers = ["m", "tanphi"] + [f"Z_A_{luminosity:g}fb" for luminosity in luminosities_fb]
    table_rows = [headers]
    for mass, tanphi, signal_sample, background_sample in rows:
        final_signal_stage = signal_sample.stages[-1]
        final_background_stage = background_sample.stages[-1]
        values = [mass, tanphi]
        for luminosity_fb in luminosities_fb:
            signal_yield = stage_yield(final_signal_stage, luminosity_fb)
            background_yield = stage_yield(final_background_stage, luminosity_fb)
            values.append(
                format_significance(
                    asimov_significance(signal_yield, background_yield)
                )
            )
        table_rows.append(values)
    return table_rows


def write_summary_file(
    output_path: Path,
    rows: list[tuple[str, str, SampleSummary, SampleSummary]],
    luminosities_fb: list[float],
) -> Path:
    table_rows = build_summary_table(rows, luminosities_fb)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(table_rows)
    return output_path.resolve()


def main() -> int:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    background_dir = args.background_dir.resolve()
    background_sample = load_background_sample(background_dir)
    if args.mass is not None and args.tanphi is not None:
        targets = [
            (
                args.mass,
                args.tanphi,
                build_generated_dir(base_dir, args.mass, args.tanphi).resolve(),
            )
        ]
    else:
        targets = list(discover_generated_dirs(base_dir))

    summary_rows: list[tuple[str, str, SampleSummary, SampleSummary]] = []
    for mass, tanphi, generated_dir in targets:
        signal_sample = load_signal_sample(mass, tanphi, generated_dir)
        output_path = write_significance_file(
            build_significance_output_path(generated_dir),
            signal_sample,
            background_sample,
            args.luminosities,
        )
        print(f"Wrote {output_path}")
        summary_rows.append((mass, tanphi, signal_sample, background_sample))

    summary_output_path = write_summary_file(
        build_summary_output_path(base_dir),
        summary_rows,
        args.luminosities,
    )
    print(f"Wrote {summary_output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
