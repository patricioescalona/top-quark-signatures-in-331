#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cuts
import significance
from generated_signal_paths import build_generated_dir, format_value_for_filename


MAXIMUM_TANPHI_IN_DECREASING_REGIME = 1.0
DEFAULT_MINIMUM_TANPHI_IN_DECREASING_REGIME = 1.0e-2
DEFAULT_SEARCH_RESULTS_DIRECTORY = SCRIPT_DIR / "asimov-significance-search-results"


def load_module_from_script_path(module_name: str, script_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


GENERATOR_AND_COMPRESSOR_MODULE = load_module_from_script_path(
    "generator_and_compressor_for_tanphi_search",
    SCRIPT_DIR / "generator-and-compressor.py",
)


@dataclass(frozen=True)
class GeneratedSignalPointEvaluation:
    mass_label: str
    tanphi_label: str
    tanphi_value: float
    integrated_luminosity_in_fb: float
    number_of_events_per_process: int
    final_signal_yield: float
    final_background_yield: float
    final_asimov_significance: float
    generated_signal_point_directory: Path
    significance_output_path: Path
    reused_existing_generated_signal_point: bool
    reran_generator_and_compressor: bool
    reran_cuts: bool


@dataclass(frozen=True)
class PreparedGeneratedSignalPoint:
    generated_signal_point_directory: Path
    reused_existing_generated_signal_point: bool
    reran_generator_and_compressor: bool
    reran_cuts: bool


@dataclass(frozen=True)
class TanphiSearchResult:
    target_asimov_significance: float
    seed_evaluation: GeneratedSignalPointEvaluation
    lower_bracketing_evaluation: GeneratedSignalPointEvaluation
    upper_bracketing_evaluation: GeneratedSignalPointEvaluation
    best_evaluation: GeneratedSignalPointEvaluation
    number_of_bracketing_iterations: int
    number_of_bisection_iterations: int
    stopping_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate or reuse signal points in the tanphi < 1 regime, run the cutflow, "
            "write a single-luminosity significance.txt, and optionally search for the "
            "tanphi value that reaches a target Asimov significance."
        )
    )
    parser.add_argument(
        "--mass",
        required=True,
        help="Mass label passed to the generated signal point directory and MadGraph card.",
    )
    parser.add_argument(
        "--integrated-luminosity-in-fb",
        type=float,
        required=True,
        choices=[450.0, 3000.0],
        help="Single luminosity used for the Asimov significance. Allowed values: 450 or 3000.",
    )
    parser.add_argument(
        "--tanphi-seed",
        type=float,
        required=True,
        help="Starting tanphi value used in the decreasing-tanphi regime search.",
    )
    parser.add_argument(
        "--target-asimov-significance",
        type=float,
        default=None,
        help=(
            "Optional target Asimov significance. If omitted, the script only evaluates "
            "the tanphi seed."
        ),
    )
    parser.add_argument(
        "--number-of-events-per-process",
        type=int,
        default=10000,
        help="Number of generated events requested for each proc-* directory. Default: 10000",
    )
    parser.add_argument(
        "--tanphi-growth-factor-during-bracketing",
        type=float,
        default=2.0,
        help="Multiplicative factor used while expanding the tanphi bracket. Default: 2.0",
    )
    parser.add_argument(
        "--relative-tolerance-on-tanphi",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--absolute-tolerance-on-target-asimov-significance",
        type=float,
        default=1e-1,
        help="Absolute stopping criterion on |Z_A - target|. Default: 1e-1",
    )
    parser.add_argument(
        "--maximum-number-of-bracketing-iterations",
        type=int,
        default=25,
        help="Maximum number of multiplicative bracket-expansion steps. Default: 25",
    )
    parser.add_argument(
        "--maximum-number-of-bisection-iterations",
        type=int,
        default=25,
        help="Maximum number of logarithmic bisection steps. Default: 25",
    )
    parser.add_argument(
        "--minimum-tanphi-in-decreasing-regime",
        type=float,
        default=DEFAULT_MINIMUM_TANPHI_IN_DECREASING_REGIME,
        help="Smallest tanphi value allowed while expanding the lower bracket. Default: 1e-2",
    )
    parser.add_argument(
        "--madgraph-bin-directory",
        type=Path,
        default=GENERATOR_AND_COMPRESSOR_MODULE.DEFAULT_MG5_BIN,
        help="MadGraph bin directory that contains proc-* and mg5_aMC.",
    )
    parser.add_argument(
        "--madgraph-command-file-output-path",
        type=Path,
        default=GENERATOR_AND_COMPRESSOR_MODULE.DEFAULT_OUTPUT,
        help="Output .mg5 command file used for the generation step.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Base exclusion directory. Default: exclusion/",
    )
    parser.add_argument(
        "--background-dir",
        type=Path,
        default=significance.DEFAULT_BACKGROUND_DIR,
        help="Background directory used by significance.py.",
    )
    parser.add_argument(
        "--search-results-directory",
        type=Path,
        default=DEFAULT_SEARCH_RESULTS_DIRECTORY,
        help="Directory where the search history CSV and summary TXT are written.",
    )
    parser.add_argument(
        "--mass-pdg",
        type=int,
        default=36,
        help="PDG code whose mass entry is updated in the MadGraph card. Default: 36",
    )
    parser.add_argument(
        "--tanphi-index",
        type=int,
        default=1,
        help="Index used in the TANPHI block of the MadGraph card. Default: 1",
    )
    parser.add_argument(
        "--beam-1-energy-in-gev",
        default="7000",
        help="Beam-1 energy written to the MadGraph card. Default: 7000",
    )
    parser.add_argument(
        "--beam-2-energy-in-gev",
        default="7000",
        help="Beam-2 energy written to the MadGraph card. Default: 7000",
    )
    parser.add_argument(
        "--regenerate-existing-generated-signal-point-if-number-of-events-does-not-match",
        action="store_true",
        help=(
            "If the existing xsec summary reports a different number of generated events, "
            "rerun generator-and-compressor.py instead of raising an error."
        ),
    )
    args = parser.parse_args()

    if args.tanphi_seed <= 0.0 or args.tanphi_seed >= MAXIMUM_TANPHI_IN_DECREASING_REGIME:
        parser.error("--tanphi-seed must satisfy 0 < tanphi < 1 in the decreasing regime.")
    if args.number_of_events_per_process <= 0:
        parser.error("--number-of-events-per-process must be a positive integer.")
    if args.target_asimov_significance is not None and args.target_asimov_significance <= 0.0:
        parser.error("--target-asimov-significance must be positive.")
    if args.tanphi_growth_factor_during_bracketing <= 1.0:
        parser.error("--tanphi-growth-factor-during-bracketing must be greater than 1.")
    if args.absolute_tolerance_on_target_asimov_significance <= 0.0:
        parser.error("--absolute-tolerance-on-target-asimov-significance must be positive.")
    if args.maximum_number_of_bracketing_iterations <= 0:
        parser.error("--maximum-number-of-bracketing-iterations must be positive.")
    if args.maximum_number_of_bisection_iterations <= 0:
        parser.error("--maximum-number-of-bisection-iterations must be positive.")
    if (
        args.minimum_tanphi_in_decreasing_regime <= 0.0
        or args.minimum_tanphi_in_decreasing_regime
        >= MAXIMUM_TANPHI_IN_DECREASING_REGIME
    ):
        parser.error("--minimum-tanphi-in-decreasing-regime must satisfy 0 < tanphi < 1.")
    if args.madgraph_command_file_output_path.suffix != ".mg5":
        parser.error("--madgraph-command-file-output-path must end with .mg5.")

    return args


def format_tanphi_label(value: float) -> str:
    return f"{value:.12g}"


def build_search_output_stem(
    mass_label: str,
    integrated_luminosity_in_fb: float,
    target_asimov_significance: float,
    tanphi_seed: float,
    number_of_events_per_process: int,
) -> str:
    return (
        "m-"
        f"{format_value_for_filename(mass_label)}"
        "-lumi-"
        f"{format_value_for_filename(f'{integrated_luminosity_in_fb:g}')}"
        "fb-target-"
        f"{format_value_for_filename(f'{target_asimov_significance:.12g}')}"
        "-tanphi-seed-"
        f"{format_value_for_filename(f'{tanphi_seed:.12g}')}"
        "-nevents-"
        f"{number_of_events_per_process}"
    )


def build_search_history_output_path(
    output_directory: Path,
    mass_label: str,
    integrated_luminosity_in_fb: float,
    target_asimov_significance: float,
    tanphi_seed: float,
    number_of_events_per_process: int,
) -> Path:
    return output_directory / (
        "asimov-significance-search-history-"
        + build_search_output_stem(
            mass_label,
            integrated_luminosity_in_fb,
            target_asimov_significance,
            tanphi_seed,
            number_of_events_per_process,
        )
        + ".csv"
    )


def build_search_summary_output_path(
    output_directory: Path,
    mass_label: str,
    integrated_luminosity_in_fb: float,
    target_asimov_significance: float,
    tanphi_seed: float,
    number_of_events_per_process: int,
) -> Path:
    return output_directory / (
        "asimov-significance-search-result-"
        + build_search_output_stem(
            mass_label,
            integrated_luminosity_in_fb,
            target_asimov_significance,
            tanphi_seed,
            number_of_events_per_process,
        )
        + ".txt"
    )


def load_requested_number_of_events_per_process_from_cross_section_summary(
    cross_section_summary_csv_path: Path,
) -> int:
    if not cross_section_summary_csv_path.is_file():
        raise FileNotFoundError(f"Cross-section summary not found: {cross_section_summary_csv_path}")

    requested_event_counts: set[int] = set()
    with cross_section_summary_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            requested_event_counts.add(int(round(float(row["N events (generated)"]))))

    if not requested_event_counts:
        raise ValueError(f"No rows found in {cross_section_summary_csv_path}")
    if len(requested_event_counts) != 1:
        raise ValueError(
            "Expected a unique number of generated events per process in "
            f"{cross_section_summary_csv_path}, found {sorted(requested_event_counts)}."
        )

    return next(iter(requested_event_counts))


def significance_is_within_target_tolerance(
    asimov_significance_value: float,
    target_asimov_significance: float,
    absolute_tolerance_on_target_asimov_significance: float,
) -> bool:
    return (
        abs(asimov_significance_value - target_asimov_significance)
        <= absolute_tolerance_on_target_asimov_significance
    )


def select_better_evaluation_for_target(
    current_best_evaluation: GeneratedSignalPointEvaluation,
    candidate_evaluation: GeneratedSignalPointEvaluation,
    target_asimov_significance: float,
) -> GeneratedSignalPointEvaluation:
    current_distance = abs(
        current_best_evaluation.final_asimov_significance - target_asimov_significance
    )
    candidate_distance = abs(
        candidate_evaluation.final_asimov_significance - target_asimov_significance
    )
    if candidate_distance < current_distance:
        return candidate_evaluation
    return current_best_evaluation


class DecreasingTanphiGeneratedSignalPointEvaluator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.background_sample = significance.load_background_sample(
            args.background_dir.resolve()
        )
        self.cached_evaluations_by_tanphi_label: dict[str, GeneratedSignalPointEvaluation] = {}
        self.evaluation_history: list[GeneratedSignalPointEvaluation] = []
        self.discovered_process_directories: list[Path] | None = None

    def evaluate_generated_signal_point_at_tanphi(
        self,
        tanphi_value: float,
    ) -> GeneratedSignalPointEvaluation:
        tanphi_label = format_tanphi_label(tanphi_value)
        if tanphi_label in self.cached_evaluations_by_tanphi_label:
            return self.cached_evaluations_by_tanphi_label[tanphi_label]

        prepared_generated_signal_point = (
            self.prepare_generated_signal_point_for_requested_number_of_events_per_process(
                tanphi_label
            )
        )
        evaluation = self.compute_final_asimov_significance_for_generated_signal_point_at_selected_luminosity(
            tanphi_label,
            prepared_generated_signal_point,
        )
        self.cached_evaluations_by_tanphi_label[tanphi_label] = evaluation
        self.evaluation_history.append(evaluation)
        self.print_evaluation_summary(evaluation)
        return evaluation

    def prepare_generated_signal_point_for_requested_number_of_events_per_process(
        self,
        tanphi_label: str,
    ) -> PreparedGeneratedSignalPoint:
        generated_signal_point_directory = build_generated_dir(
            self.args.base_dir.resolve(),
            self.args.mass,
            tanphi_label,
        )
        cross_section_summary_csv_path = cuts.build_xsec_csv_path(
            generated_signal_point_directory,
            self.args.mass,
            tanphi_label,
        )
        cutflow_csv_path = cuts.build_output_csv_path(
            generated_signal_point_directory,
            self.args.mass,
            tanphi_label,
        )
        efficiencies_csv_path = cuts.build_efficiencies_csv_path(
            generated_signal_point_directory,
            self.args.mass,
            tanphi_label,
        )

        has_cross_section_summary = cross_section_summary_csv_path.is_file()
        has_complete_compressed_event_parquets = (
            self.generated_signal_point_has_complete_compressed_event_parquets(
                generated_signal_point_directory
            )
        )
        requested_number_of_events_matches_existing_point = True

        if has_cross_section_summary:
            existing_number_of_events_per_process = (
                load_requested_number_of_events_per_process_from_cross_section_summary(
                    cross_section_summary_csv_path
                )
            )
            requested_number_of_events_matches_existing_point = (
                existing_number_of_events_per_process
                == self.args.number_of_events_per_process
            )
            if (
                not requested_number_of_events_matches_existing_point
                and has_complete_compressed_event_parquets
                and not self.args.regenerate_existing_generated_signal_point_if_number_of_events_does_not_match
            ):
                raise ValueError(
                    "The existing generated signal point "
                    f"{generated_signal_point_directory} was produced with "
                    f"{existing_number_of_events_per_process} events per process, but "
                    f"--number-of-events-per-process requested "
                    f"{self.args.number_of_events_per_process}. "
                    "Use "
                    "--regenerate-existing-generated-signal-point-if-number-of-events-does-not-match "
                    "to overwrite it."
                )

        must_rerun_generator_and_compressor = (
            not has_cross_section_summary
            or not has_complete_compressed_event_parquets
            or (
                has_cross_section_summary
                and not requested_number_of_events_matches_existing_point
                and self.args.regenerate_existing_generated_signal_point_if_number_of_events_does_not_match
            )
        )

        if must_rerun_generator_and_compressor:
            self.run_generator_and_compressor_for_generated_signal_point(tanphi_label)

        must_rerun_cuts = (
            must_rerun_generator_and_compressor
            or not cutflow_csv_path.is_file()
            or not efficiencies_csv_path.is_file()
        )
        if must_rerun_cuts:
            cuts.run_cutflow_for_generated_dir(
                self.args.mass,
                tanphi_label,
                generated_signal_point_directory,
            )

        return PreparedGeneratedSignalPoint(
            generated_signal_point_directory=generated_signal_point_directory,
            reused_existing_generated_signal_point=not must_rerun_generator_and_compressor,
            reran_generator_and_compressor=must_rerun_generator_and_compressor,
            reran_cuts=must_rerun_cuts,
        )

    def generated_signal_point_has_complete_compressed_event_parquets(
        self,
        generated_signal_point_directory: Path,
    ) -> bool:
        try:
            cuts.discover_parquets(generated_signal_point_directory / "parquets")
        except (FileNotFoundError, ValueError):
            return False
        return True

    def run_generator_and_compressor_for_generated_signal_point(self, tanphi_label: str) -> None:
        if self.discovered_process_directories is None:
            self.discovered_process_directories = (
                GENERATOR_AND_COMPRESSOR_MODULE.discover_process_directories(
                    self.args.madgraph_bin_directory.resolve()
                )
            )

        generator_arguments = argparse.Namespace(
            nevents=self.args.number_of_events_per_process,
            mass_pdg=self.args.mass_pdg,
            mass=self.args.mass,
            mass_range=None,
            mass_points=1,
            tanphi_index=self.args.tanphi_index,
            tanphi=tanphi_label,
            tanphi_range=None,
            tanphi_points=1,
            ebeam1=self.args.beam_1_energy_in_gev,
            ebeam2=self.args.beam_2_energy_in_gev,
            mg5_bin=self.args.madgraph_bin_directory,
            output=self.args.madgraph_command_file_output_path,
            write_only=False,
            xsec_output=None,
        )
        return_code = GENERATOR_AND_COMPRESSOR_MODULE.run_single_point(
            self.discovered_process_directories,
            generator_arguments,
            self.args.mass,
            tanphi_label,
            None,
        )
        if return_code != 0:
            raise RuntimeError(
                "generator-and-compressor.py finished with a non-zero exit code "
                f"for mass={self.args.mass}, tanphi={tanphi_label}: {return_code}"
            )

    def compute_final_asimov_significance_for_generated_signal_point_at_selected_luminosity(
        self,
        tanphi_label: str,
        prepared_generated_signal_point: PreparedGeneratedSignalPoint,
    ) -> GeneratedSignalPointEvaluation:
        generated_signal_point_directory = (
            prepared_generated_signal_point.generated_signal_point_directory
        )
        signal_sample = significance.load_signal_sample(
            self.args.mass,
            tanphi_label,
            generated_signal_point_directory,
        )
        significance_output_path = significance.write_significance_file(
            significance.build_significance_output_path(generated_signal_point_directory),
            signal_sample,
            self.background_sample,
            [self.args.integrated_luminosity_in_fb],
        )

        final_signal_stage = signal_sample.stages[-1]
        final_background_stage = self.background_sample.stages[-1]
        final_signal_yield = significance.stage_yield(
            final_signal_stage,
            self.args.integrated_luminosity_in_fb,
        )
        final_background_yield = significance.stage_yield(
            final_background_stage,
            self.args.integrated_luminosity_in_fb,
        )
        final_asimov_significance = significance.asimov_significance(
            final_signal_yield,
            final_background_yield,
        )

        return GeneratedSignalPointEvaluation(
            mass_label=self.args.mass,
            tanphi_label=tanphi_label,
            tanphi_value=float(tanphi_label),
            integrated_luminosity_in_fb=self.args.integrated_luminosity_in_fb,
            number_of_events_per_process=self.args.number_of_events_per_process,
            final_signal_yield=final_signal_yield,
            final_background_yield=final_background_yield,
            final_asimov_significance=final_asimov_significance,
            generated_signal_point_directory=generated_signal_point_directory.resolve(),
            significance_output_path=significance_output_path,
            reused_existing_generated_signal_point=(
                prepared_generated_signal_point.reused_existing_generated_signal_point
            ),
            reran_generator_and_compressor=(
                prepared_generated_signal_point.reran_generator_and_compressor
            ),
            reran_cuts=prepared_generated_signal_point.reran_cuts,
        )

    def print_evaluation_summary(self, evaluation: GeneratedSignalPointEvaluation) -> None:
        print(
            "Evaluated generated signal point: "
            f"mass={evaluation.mass_label}, "
            f"tanphi={evaluation.tanphi_label}, "
            f"luminosity={evaluation.integrated_luminosity_in_fb:g} fb^-1, "
            f"events/proc={evaluation.number_of_events_per_process}, "
            f"Z_A={evaluation.final_asimov_significance:.6f}, "
            f"signal_yield={evaluation.final_signal_yield:.6f}, "
            f"background_yield={evaluation.final_background_yield:.6f}"
        )
        print(f"  generated_dir: {evaluation.generated_signal_point_directory}")
        print(f"  significance_txt: {evaluation.significance_output_path}")
        print(
            "  actions: "
            f"reused_existing_generated_signal_point="
            f"{evaluation.reused_existing_generated_signal_point}, "
            f"reran_generator_and_compressor={evaluation.reran_generator_and_compressor}, "
            f"reran_cuts={evaluation.reran_cuts}"
        )


def find_tanphi_bracketing_interval_for_target_asimov_significance_in_decreasing_regime(
    evaluator: DecreasingTanphiGeneratedSignalPointEvaluator,
    seed_evaluation: GeneratedSignalPointEvaluation,
    args: argparse.Namespace,
) -> tuple[
    GeneratedSignalPointEvaluation,
    GeneratedSignalPointEvaluation,
    int,
]:
    target_asimov_significance = args.target_asimov_significance
    if target_asimov_significance is None:
        raise ValueError("Target significance is required to build a bracketing interval.")

    if significance_is_within_target_tolerance(
        seed_evaluation.final_asimov_significance,
        target_asimov_significance,
        args.absolute_tolerance_on_target_asimov_significance,
    ):
        return seed_evaluation, seed_evaluation, 0

    if seed_evaluation.final_asimov_significance > target_asimov_significance:
        lower_bracketing_evaluation = seed_evaluation
        for iteration_index in range(1, args.maximum_number_of_bracketing_iterations + 1):
            candidate_tanphi_value = min(
                lower_bracketing_evaluation.tanphi_value
                * args.tanphi_growth_factor_during_bracketing,
                MAXIMUM_TANPHI_IN_DECREASING_REGIME,
            )
            if candidate_tanphi_value <= lower_bracketing_evaluation.tanphi_value:
                break

            upper_bracketing_evaluation = evaluator.evaluate_generated_signal_point_at_tanphi(
                candidate_tanphi_value
            )
            if significance_is_within_target_tolerance(
                upper_bracketing_evaluation.final_asimov_significance,
                target_asimov_significance,
                args.absolute_tolerance_on_target_asimov_significance,
            ):
                return upper_bracketing_evaluation, upper_bracketing_evaluation, iteration_index
            if upper_bracketing_evaluation.final_asimov_significance <= target_asimov_significance:
                return lower_bracketing_evaluation, upper_bracketing_evaluation, iteration_index
            lower_bracketing_evaluation = upper_bracketing_evaluation

        raise ValueError(
            "Could not bracket the target Asimov significance from below before reaching "
            f"tanphi={MAXIMUM_TANPHI_IN_DECREASING_REGIME:.12g}."
        )

    upper_bracketing_evaluation = seed_evaluation
    for iteration_index in range(1, args.maximum_number_of_bracketing_iterations + 1):
        candidate_tanphi_value = (
            upper_bracketing_evaluation.tanphi_value
            / args.tanphi_growth_factor_during_bracketing
        )
        if candidate_tanphi_value <= args.minimum_tanphi_in_decreasing_regime:
            lower_bracketing_evaluation = evaluator.evaluate_generated_signal_point_at_tanphi(
                args.minimum_tanphi_in_decreasing_regime
            )
            if significance_is_within_target_tolerance(
                lower_bracketing_evaluation.final_asimov_significance,
                target_asimov_significance,
                args.absolute_tolerance_on_target_asimov_significance,
            ):
                return lower_bracketing_evaluation, lower_bracketing_evaluation, iteration_index
            if lower_bracketing_evaluation.final_asimov_significance < target_asimov_significance:
                raise ValueError(
                    "Even the minimum tanphi allowed in the decreasing regime stays below "
                    "the target Asimov significance, so no solution was found."
                )
            return lower_bracketing_evaluation, upper_bracketing_evaluation, iteration_index

        lower_bracketing_evaluation = evaluator.evaluate_generated_signal_point_at_tanphi(
            candidate_tanphi_value
        )
        if significance_is_within_target_tolerance(
            lower_bracketing_evaluation.final_asimov_significance,
            target_asimov_significance,
            args.absolute_tolerance_on_target_asimov_significance,
        ):
            return lower_bracketing_evaluation, lower_bracketing_evaluation, iteration_index
        if lower_bracketing_evaluation.final_asimov_significance >= target_asimov_significance:
            return lower_bracketing_evaluation, upper_bracketing_evaluation, iteration_index
        upper_bracketing_evaluation = lower_bracketing_evaluation

    raise ValueError(
        "Could not bracket the target Asimov significance from above before exhausting "
        "the allowed number of bracketing iterations."
    )


def refine_tanphi_with_logarithmic_bisection_for_target_asimov_significance_in_decreasing_regime(
    evaluator: DecreasingTanphiGeneratedSignalPointEvaluator,
    lower_bracketing_evaluation: GeneratedSignalPointEvaluation,
    upper_bracketing_evaluation: GeneratedSignalPointEvaluation,
    args: argparse.Namespace,
) -> tuple[GeneratedSignalPointEvaluation, int, str]:
    target_asimov_significance = args.target_asimov_significance
    if target_asimov_significance is None:
        raise ValueError("Target significance is required for bisection.")

    best_evaluation = select_better_evaluation_for_target(
        lower_bracketing_evaluation,
        upper_bracketing_evaluation,
        target_asimov_significance,
    )
    if significance_is_within_target_tolerance(
        best_evaluation.final_asimov_significance,
        target_asimov_significance,
        args.absolute_tolerance_on_target_asimov_significance,
    ):
        return (
            best_evaluation,
            0,
            "A bracketing endpoint already satisfied the absolute target-significance tolerance.",
        )
    if lower_bracketing_evaluation.tanphi_value == upper_bracketing_evaluation.tanphi_value:
        return (
            best_evaluation,
            0,
            "A bracketing evaluation satisfied the absolute target-significance tolerance.",
        )

    bisection_iteration_count = 0
    for bisection_iteration_count in range(1, args.maximum_number_of_bisection_iterations + 1):
        midpoint_tanphi_value = math.sqrt(
            lower_bracketing_evaluation.tanphi_value
            * upper_bracketing_evaluation.tanphi_value
        )
        if (
            midpoint_tanphi_value == lower_bracketing_evaluation.tanphi_value
            or midpoint_tanphi_value == upper_bracketing_evaluation.tanphi_value
        ):
            return (
                best_evaluation,
                bisection_iteration_count,
                "The logarithmic midpoint collapsed onto a bracket endpoint before reaching the target-significance tolerance.",
            )

        midpoint_evaluation = evaluator.evaluate_generated_signal_point_at_tanphi(
            midpoint_tanphi_value
        )
        best_evaluation = select_better_evaluation_for_target(
            best_evaluation,
            midpoint_evaluation,
            target_asimov_significance,
        )
        if significance_is_within_target_tolerance(
            midpoint_evaluation.final_asimov_significance,
            target_asimov_significance,
            args.absolute_tolerance_on_target_asimov_significance,
        ):
            return (
                midpoint_evaluation,
                bisection_iteration_count,
                "A midpoint evaluation satisfied the absolute target-significance tolerance.",
            )

        if midpoint_evaluation.final_asimov_significance < target_asimov_significance:
            upper_bracketing_evaluation = midpoint_evaluation
        else:
            lower_bracketing_evaluation = midpoint_evaluation

    return (
        best_evaluation,
        bisection_iteration_count,
        "The maximum number of bisection iterations was reached before any evaluation satisfied the absolute target-significance tolerance.",
    )


def write_search_history_csv(
    output_path: Path,
    evaluation_history: list[GeneratedSignalPointEvaluation],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "evaluation_index",
                "mass",
                "tanphi",
                "integrated_luminosity_in_fb",
                "number_of_events_per_process",
                "final_signal_yield",
                "final_background_yield",
                "final_asimov_significance",
                "generated_signal_point_directory",
                "significance_output_path",
                "reused_existing_generated_signal_point",
                "reran_generator_and_compressor",
                "reran_cuts",
            ],
        )
        writer.writeheader()
        for evaluation_index, evaluation in enumerate(evaluation_history, start=1):
            writer.writerow(
                {
                    "evaluation_index": evaluation_index,
                    "mass": evaluation.mass_label,
                    "tanphi": evaluation.tanphi_label,
                    "integrated_luminosity_in_fb": (
                        f"{evaluation.integrated_luminosity_in_fb:g}"
                    ),
                    "number_of_events_per_process": evaluation.number_of_events_per_process,
                    "final_signal_yield": f"{evaluation.final_signal_yield:.6f}",
                    "final_background_yield": f"{evaluation.final_background_yield:.6f}",
                    "final_asimov_significance": (
                        f"{evaluation.final_asimov_significance:.6f}"
                    ),
                    "generated_signal_point_directory": str(
                        evaluation.generated_signal_point_directory
                    ),
                    "significance_output_path": str(evaluation.significance_output_path),
                    "reused_existing_generated_signal_point": (
                        str(evaluation.reused_existing_generated_signal_point)
                    ),
                    "reran_generator_and_compressor": str(
                        evaluation.reran_generator_and_compressor
                    ),
                    "reran_cuts": str(evaluation.reran_cuts),
                }
            )
    return output_path.resolve()


def write_search_summary_text(
    output_path: Path,
    search_result: TanphiSearchResult,
    evaluation_history_csv_path: Path,
    number_of_unique_tanphi_evaluations: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Decreasing-tanphi Asimov significance search",
        f"mass: {search_result.best_evaluation.mass_label}",
        f"integrated luminosity [fb^-1]: {search_result.best_evaluation.integrated_luminosity_in_fb:g}",
        f"number of events per process: {search_result.best_evaluation.number_of_events_per_process}",
        f"target Asimov significance: {search_result.target_asimov_significance:.6f}",
        "",
        "Seed evaluation",
        f"tanphi: {search_result.seed_evaluation.tanphi_label}",
        f"Z_A: {search_result.seed_evaluation.final_asimov_significance:.6f}",
        "",
        "Bracketing interval",
        f"lower tanphi: {search_result.lower_bracketing_evaluation.tanphi_label}",
        f"lower Z_A: {search_result.lower_bracketing_evaluation.final_asimov_significance:.6f}",
        f"upper tanphi: {search_result.upper_bracketing_evaluation.tanphi_label}",
        f"upper Z_A: {search_result.upper_bracketing_evaluation.final_asimov_significance:.6f}",
        "",
        "Best evaluation found",
        f"tanphi: {search_result.best_evaluation.tanphi_label}",
        f"Z_A: {search_result.best_evaluation.final_asimov_significance:.6f}",
        f"signal yield: {search_result.best_evaluation.final_signal_yield:.6f}",
        f"background yield: {search_result.best_evaluation.final_background_yield:.6f}",
        f"generated signal point directory: {search_result.best_evaluation.generated_signal_point_directory}",
        f"significance output path: {search_result.best_evaluation.significance_output_path}",
        "",
        "Search accounting",
        f"number of bracketing iterations: {search_result.number_of_bracketing_iterations}",
        f"number of bisection iterations: {search_result.number_of_bisection_iterations}",
        f"stopping reason: {search_result.stopping_reason}",
        f"number of unique tanphi evaluations: {number_of_unique_tanphi_evaluations}",
        f"evaluation history CSV: {evaluation_history_csv_path}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path.resolve()


def print_final_single_point_summary(
    evaluation: GeneratedSignalPointEvaluation,
) -> None:
    print("")
    print("Final single-point evaluation")
    print(f"  mass={evaluation.mass_label}")
    print(f"  tanphi={evaluation.tanphi_label}")
    print(f"  luminosity={evaluation.integrated_luminosity_in_fb:g} fb^-1")
    print(f"  events/proc={evaluation.number_of_events_per_process}")
    print(f"  final signal yield={evaluation.final_signal_yield:.6f}")
    print(f"  final background yield={evaluation.final_background_yield:.6f}")
    print(f"  final Z_A={evaluation.final_asimov_significance:.6f}")
    print(f"  significance.txt={evaluation.significance_output_path}")


def print_final_search_summary(
    search_result: TanphiSearchResult,
    evaluation_history_csv_path: Path,
    search_summary_output_path: Path,
) -> None:
    print("")
    print("Final tanphi search result")
    print(f"  target Z_A={search_result.target_asimov_significance:.6f}")
    print(f"  best tanphi={search_result.best_evaluation.tanphi_label}")
    print(f"  best Z_A={search_result.best_evaluation.final_asimov_significance:.6f}")
    print(f"  stopping reason={search_result.stopping_reason}")
    print(
        f"  lower bracket=({search_result.lower_bracketing_evaluation.tanphi_label}, "
        f"{search_result.lower_bracketing_evaluation.final_asimov_significance:.6f})"
    )
    print(
        f"  upper bracket=({search_result.upper_bracketing_evaluation.tanphi_label}, "
        f"{search_result.upper_bracketing_evaluation.final_asimov_significance:.6f})"
    )
    print(f"  search history CSV={evaluation_history_csv_path}")
    print(f"  search summary TXT={search_summary_output_path}")


def main() -> int:
    args = parse_args()
    vector.register_awkward()

    evaluator = DecreasingTanphiGeneratedSignalPointEvaluator(args)
    seed_evaluation = evaluator.evaluate_generated_signal_point_at_tanphi(args.tanphi_seed)

    if args.target_asimov_significance is None:
        print_final_single_point_summary(seed_evaluation)
        return 0

    (
        lower_bracketing_evaluation,
        upper_bracketing_evaluation,
        number_of_bracketing_iterations,
    ) = find_tanphi_bracketing_interval_for_target_asimov_significance_in_decreasing_regime(
        evaluator,
        seed_evaluation,
        args,
    )
    best_evaluation, number_of_bisection_iterations, stopping_reason = (
        refine_tanphi_with_logarithmic_bisection_for_target_asimov_significance_in_decreasing_regime(
            evaluator,
            lower_bracketing_evaluation,
            upper_bracketing_evaluation,
            args,
        )
    )

    search_result = TanphiSearchResult(
        target_asimov_significance=args.target_asimov_significance,
        seed_evaluation=seed_evaluation,
        lower_bracketing_evaluation=lower_bracketing_evaluation,
        upper_bracketing_evaluation=upper_bracketing_evaluation,
        best_evaluation=best_evaluation,
        number_of_bracketing_iterations=number_of_bracketing_iterations,
        number_of_bisection_iterations=number_of_bisection_iterations,
        stopping_reason=stopping_reason,
    )
    evaluation_history_csv_path = write_search_history_csv(
        build_search_history_output_path(
            args.search_results_directory.resolve(),
            args.mass,
            args.integrated_luminosity_in_fb,
            args.target_asimov_significance,
            args.tanphi_seed,
            args.number_of_events_per_process,
        ),
        evaluator.evaluation_history,
    )
    search_summary_output_path = write_search_summary_text(
        build_search_summary_output_path(
            args.search_results_directory.resolve(),
            args.mass,
            args.integrated_luminosity_in_fb,
            args.target_asimov_significance,
            args.tanphi_seed,
            args.number_of_events_per_process,
        ),
        search_result,
        evaluation_history_csv_path,
        len(evaluator.evaluation_history),
    )
    print_final_search_summary(
        search_result,
        evaluation_history_csv_path,
        search_summary_output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
