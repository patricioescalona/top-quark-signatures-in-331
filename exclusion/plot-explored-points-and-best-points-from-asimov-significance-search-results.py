#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

MPLCONFIGDIR = Path(__file__).resolve().parent / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SEARCH_RESULTS_DIRECTORY = SCRIPT_DIR / "asimov-significance-search-results"
DEFAULT_DECAYS_SCAN_CSV_PATH = SCRIPT_DIR.parent / "decays" / "decays.csv"
DEFAULT_OUTPUT_PLOT_PATH = (
    DEFAULT_SEARCH_RESULTS_DIRECTORY
    / "explored-points-and-best-points-from-asimov-significance-search-results.png"
)


@dataclass(frozen=True)
class SearchConfiguration:
    integrated_luminosity_in_fb: float
    target_asimov_significance: float
    number_of_events_per_process: int


@dataclass(frozen=True)
class SearchSummary:
    summary_path: Path
    evaluation_history_csv_path: Path
    mass_value: float
    best_tanphi_value: float
    best_asimov_significance: float
    configuration: SearchConfiguration
    stopping_reason: str


@dataclass(frozen=True)
class HistoryPoint:
    mass_value: float
    tanphi_value: float
    final_asimov_significance: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot explored tanphi search points and the best points from "
            "asimov-significance-search-results, using the decays scan to match "
            "the parameter-space axis ranges."
        )
    )
    parser.add_argument(
        "--search-results-directory",
        type=Path,
        default=DEFAULT_SEARCH_RESULTS_DIRECTORY,
        help=(
            "Directory containing asimov-significance-search-result-*.txt and "
            "asimov-significance-search-history-*.csv. "
            "Default: exclusion/asimov-significance-search-results/"
        ),
    )
    parser.add_argument(
        "--decays-scan-csv-path",
        type=Path,
        default=DEFAULT_DECAYS_SCAN_CSV_PATH,
        help="CSV file used to match the axis limits from decays/. Default: decays/decays.csv",
    )
    parser.add_argument(
        "--output-plot-path",
        type=Path,
        default=DEFAULT_OUTPUT_PLOT_PATH,
        help=(
            "Output PNG file for the explored/best-point plot. Default: "
            "exclusion/asimov-significance-search-results/"
            "explored-points-and-best-points-from-asimov-significance-search-results.png"
        ),
    )
    parser.add_argument(
        "--integrated-luminosity-in-fb",
        type=float,
        default=None,
        help="Optional filter on the integrated luminosity stored in the search summary.",
    )
    parser.add_argument(
        "--target-asimov-significance",
        type=float,
        default=None,
        help="Optional filter on the target Asimov significance stored in the search summary.",
    )
    parser.add_argument(
        "--number-of-events-per-process",
        type=int,
        default=None,
        help="Optional filter on the generated event count stored in the search summary.",
    )
    parser.add_argument(
        "--allow-mixed-search-configurations",
        action="store_true",
        help=(
            "Allow plotting search results with different luminosities, targets, or "
            "numbers of events per process in the same figure."
        ),
    )
    args = parser.parse_args()

    if (
        args.integrated_luminosity_in_fb is not None
        and args.integrated_luminosity_in_fb <= 0.0
    ):
        parser.error("--integrated-luminosity-in-fb must be positive.")
    if (
        args.target_asimov_significance is not None
        and args.target_asimov_significance <= 0.0
    ):
        parser.error("--target-asimov-significance must be positive.")
    if (
        args.number_of_events_per_process is not None
        and args.number_of_events_per_process <= 0
    ):
        parser.error("--number-of-events-per-process must be a positive integer.")
    if args.output_plot_path.suffix.lower() != ".png":
        parser.error("--output-plot-path must end with .png.")

    return args


def parse_search_summary(summary_path: Path) -> SearchSummary:
    lines = summary_path.read_text(encoding="utf-8").splitlines()
    current_section = ""
    values_by_key: dict[tuple[str, str], str] = {}

    for raw_line in lines:
        stripped_line = raw_line.strip()
        if not stripped_line:
            continue
        if stripped_line in {
            "Seed evaluation",
            "Bracketing interval",
            "Best evaluation found",
            "Search accounting",
        }:
            current_section = stripped_line
            continue
        if ":" not in stripped_line:
            continue

        key, value = stripped_line.split(":", 1)
        values_by_key[(current_section, key.strip())] = value.strip()

    try:
        mass_value = float(values_by_key[("", "mass")])
        integrated_luminosity_in_fb = float(
            values_by_key[("", "integrated luminosity [fb^-1]")]
        )
        number_of_events_per_process = int(
            values_by_key[("", "number of events per process")]
        )
        target_asimov_significance = float(
            values_by_key[("", "target Asimov significance")]
        )
        best_tanphi_value = float(values_by_key[("Best evaluation found", "tanphi")])
        best_asimov_significance = float(values_by_key[("Best evaluation found", "Z_A")])
        evaluation_history_csv_path = Path(
            values_by_key[("Search accounting", "evaluation history CSV")]
        )
        stopping_reason = values_by_key.get(
            ("Search accounting", "stopping reason"),
            "n/a",
        )
    except KeyError as error:
        raise ValueError(f"Missing expected field in {summary_path}: {error}") from error

    return SearchSummary(
        summary_path=summary_path.resolve(),
        evaluation_history_csv_path=evaluation_history_csv_path.resolve(),
        mass_value=mass_value,
        best_tanphi_value=best_tanphi_value,
        best_asimov_significance=best_asimov_significance,
        configuration=SearchConfiguration(
            integrated_luminosity_in_fb=integrated_luminosity_in_fb,
            target_asimov_significance=target_asimov_significance,
            number_of_events_per_process=number_of_events_per_process,
        ),
        stopping_reason=stopping_reason,
    )


def load_history_points(evaluation_history_csv_path: Path) -> list[HistoryPoint]:
    with evaluation_history_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            HistoryPoint(
                mass_value=float(row["mass"]),
                tanphi_value=float(row["tanphi"]),
                final_asimov_significance=float(row["final_asimov_significance"]),
            )
            for row in reader
        ]


def load_decays_scan_points(decays_scan_csv_path: Path) -> tuple[list[float], list[float]]:
    with decays_scan_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        masses: list[float] = []
        tanphis: list[float] = []
        for row in reader:
            masses.append(float(row["mass"]))
            tanphis.append(float(row["tanphi"]))

    if not masses or not tanphis:
        raise ValueError(f"No decays scan points found in {decays_scan_csv_path}")

    return masses, tanphis


def configuration_matches_filters(
    configuration: SearchConfiguration,
    args: argparse.Namespace,
) -> bool:
    if (
        args.integrated_luminosity_in_fb is not None
        and not math.isclose(
            configuration.integrated_luminosity_in_fb,
            args.integrated_luminosity_in_fb,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        return False
    if (
        args.target_asimov_significance is not None
        and not math.isclose(
            configuration.target_asimov_significance,
            args.target_asimov_significance,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        return False
    if (
        args.number_of_events_per_process is not None
        and configuration.number_of_events_per_process != args.number_of_events_per_process
    ):
        return False
    return True


def format_configuration(configuration: SearchConfiguration) -> str:
    return (
        f"lumi={configuration.integrated_luminosity_in_fb:g} fb^-1, "
        f"target Z_A={configuration.target_asimov_significance:.6g}, "
        f"events/proc={configuration.number_of_events_per_process}"
    )


def ensure_configuration_consistency(
    summaries: list[SearchSummary],
    allow_mixed_search_configurations: bool,
) -> None:
    distinct_configurations = {
        (
            summary.configuration.integrated_luminosity_in_fb,
            summary.configuration.target_asimov_significance,
            summary.configuration.number_of_events_per_process,
        )
        for summary in summaries
    }
    if len(distinct_configurations) <= 1 or allow_mixed_search_configurations:
        return

    formatted_configurations = "\n".join(
        f"- {format_configuration(SearchConfiguration(*configuration))}"
        for configuration in sorted(distinct_configurations)
    )
    raise ValueError(
        "Multiple search configurations were found in the selected summaries. "
        "Use the luminosity/target/nevents filters, or pass "
        "--allow-mixed-search-configurations.\n"
        f"{formatted_configurations}"
    )


def build_plot_title(summaries: list[SearchSummary]) -> str:
    distinct_configurations = {
        (
            summary.configuration.integrated_luminosity_in_fb,
            summary.configuration.target_asimov_significance,
            summary.configuration.number_of_events_per_process,
        )
        for summary in summaries
    }
    if len(distinct_configurations) == 1:
        configuration = summaries[0].configuration
        return (
            "Explored and Best Search Points\n"
            f"{configuration.integrated_luminosity_in_fb:g} fb^-1, "
            f"target Z_A={configuration.target_asimov_significance:.6g}, "
            f"{configuration.number_of_events_per_process} events/proc"
        )
    return "Explored and Best Search Points"


def plot_explored_and_best_points(
    decays_scan_masses: list[float],
    decays_scan_tanphis: list[float],
    histories_by_summary: list[tuple[SearchSummary, list[HistoryPoint]]],
    output_plot_path: Path,
) -> Path:
    explored_masses = [
        history_point.mass_value
        for _, history_points in histories_by_summary
        for history_point in history_points
    ]
    explored_tanphis = [
        history_point.tanphi_value
        for _, history_points in histories_by_summary
        for history_point in history_points
    ]
    best_summaries_sorted = sorted(
        (summary for summary, _ in histories_by_summary),
        key=lambda summary: (summary.mass_value, summary.best_tanphi_value),
    )
    best_masses = [summary.mass_value for summary in best_summaries_sorted]
    best_tanphis = [summary.best_tanphi_value for summary in best_summaries_sorted]

    fig, ax = plt.subplots(1, 1, figsize=(7, 5), constrained_layout=True)
    ax.set_yscale("log")
    # Match the decays/ parameter-space plots by autoscaling on the full scan.
    ax.scatter(decays_scan_masses, decays_scan_tanphis, s=0, alpha=0)
    ax.scatter(
        explored_masses,
        explored_tanphis,
        s=60,
        facecolors="white",
        edgecolors="#7f8c8d",
        linewidths=0.8,
        alpha=0.9,
        zorder=2,
    )
    ax.scatter(
        best_masses,
        best_tanphis,
        s=90,
        facecolors="#d35400",
        edgecolors="black",
        linewidths=0.8,
        zorder=4,
    )
    if len(best_masses) >= 2:
        ax.plot(
            best_masses,
            best_tanphis,
            color="#b30000",
            linewidth=2.0,
            zorder=3,
        )

    ax.set_title(build_plot_title([summary for summary, _ in histories_by_summary]))
    ax.set_xlabel("mass")
    ax.set_ylabel("tanphi")
    ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.4)

    legend_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            label="Explored points",
            markerfacecolor="white",
            markeredgecolor="#7f8c8d",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            label="Best points",
            markerfacecolor="#d35400",
            markeredgecolor="black",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            color="#b30000",
            linewidth=2.0,
            label="Best-point curve",
        ),
    ]
    ax.legend(handles=legend_items, loc="best", frameon=True)

    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot_path, dpi=200)
    plt.close(fig)
    return output_plot_path.resolve()


def main() -> int:
    args = parse_args()
    search_results_directory = args.search_results_directory.resolve()
    decays_scan_csv_path = args.decays_scan_csv_path.resolve()
    output_plot_path = args.output_plot_path.resolve()

    summary_paths = sorted(
        search_results_directory.glob("asimov-significance-search-result-*.txt")
    )
    if not summary_paths:
        raise FileNotFoundError(
            "No asimov-significance-search-result-*.txt files were found in "
            f"{search_results_directory}"
        )

    summaries = [
        parse_search_summary(summary_path)
        for summary_path in summary_paths
    ]
    summaries = [
        summary
        for summary in summaries
        if configuration_matches_filters(summary.configuration, args)
    ]
    if not summaries:
        raise ValueError(
            "No search summaries remained after applying the requested filters."
        )

    ensure_configuration_consistency(
        summaries,
        args.allow_mixed_search_configurations,
    )

    decays_scan_masses, decays_scan_tanphis = load_decays_scan_points(decays_scan_csv_path)
    histories_by_summary = [
        (summary, load_history_points(summary.evaluation_history_csv_path))
        for summary in summaries
    ]
    written_output_path = plot_explored_and_best_points(
        decays_scan_masses,
        decays_scan_tanphis,
        histories_by_summary,
        output_plot_path,
    )
    print(f"Wrote plot to {written_output_path}")
    print(f"Loaded {len(summaries)} search summary file(s).")
    print(f"Used axis ranges from {decays_scan_csv_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
