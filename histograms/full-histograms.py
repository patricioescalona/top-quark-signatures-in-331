import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import textwrap
import traceback

import awkward as ak
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import uproot


ROOT_FILENAME = "tag_1_delphes_events.root"
TREE_NAME = "Delphes;1"
DEFAULT_RUN_DIRS = ("run_tt", "run_at", "run_aa")
SAMPLE_COLORS = ("red", "green", "blue")
COLOR_NAMES = {
    "red": "Red",
    "green": "Green",
    "blue": "Blue",
}
PLOT_SPECS = [
    ("leading_bjet_pt", "Leading b-jet pT [GeV]"),
    ("subleading_bjet_pt", "Subleading b-jet pT [GeV]"),
    ("leading_lepton_pt", "Leading lepton pT [GeV]"),
    ("subleading_lepton_pt", "Subleading lepton pT [GeV]"),
    ("leading_non_bjet_pt", "Leading non-b jet pT [GeV]"),
    ("subleading_non_bjet_pt", "Subleading non-b jet pT [GeV]"),
    ("missing_et", "Missing transverse momentum [GeV]"),
]
PDG_LABELS = {
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
    23: "z",
    24: "w+",
    25: "h",
    36: "a",
}


@dataclass
class SampleResult:
    run_dir: Path
    label: str
    color: str
    total_events: int
    run_info: dict | None
    event_weight: float
    counts: dict[str, int]
    fractions: dict[str, float]
    histograms: dict[str, np.ndarray]
    selected_yield: float


def parse_mode(value):
    normalized_value = value.strip().lower()
    mode_map = {
        "1": "mc",
        "mc": "mc",
        "raw": "mc",
        "events": "mc",
        "2": "xsec",
        "xsec": "xsec",
        "cross-section": "xsec",
        "cross_section": "xsec",
        "crosssection": "xsec",
        "3": "lumi",
        "lumi": "lumi",
        "luminosity": "lumi",
    }

    if normalized_value not in mode_map:
        raise argparse.ArgumentTypeError(
            "Mode must be one of: 1/mc, 2/xsec, or 3/lumi."
        )

    return mode_map[normalized_value]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Apply the same Delphes-level event selection to several samples and "
            "produce stacked kinematic histograms showing the contribution of each process."
        )
    )
    parser.add_argument(
        "--mode",
        type=parse_mode,
        default="mc",
        help=(
            "Histogram mode: 1 or mc for raw Monte Carlo events, "
            "2 or xsec for cross-section weighting, "
            "3 or lumi for luminosity scaling. Default: mc."
        ),
    )
    parser.add_argument(
        "--luminosity",
        type=float,
        default=130.0,
        help=(
            "Integrated luminosity in fb^-1 used when --mode is 3 or lumi. "
            "Default: 130."
        ),
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=list(DEFAULT_RUN_DIRS),
        help=(
            "Run directories to combine. Default: run_tt run_at run_aa. "
            "Up to three directories are supported to keep the RGB palette."
        ),
    )
    parser.add_argument(
        "--output",
        default="combined_selected_kinematics.png",
        help="Output image filename. Default: combined_selected_kinematics.png",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Draw the histogram y-axes on a logarithmic scale.",
    )
    args = parser.parse_args()

    if args.luminosity <= 0:
        parser.error("--luminosity must be positive.")
    if not args.runs:
        parser.error("At least one run directory must be provided with --runs.")
    if len(args.runs) > len(SAMPLE_COLORS):
        parser.error("At most three run directories are supported because the plot uses RGB colors.")

    return args


def count_and_fraction(mask, total_events):
    count = int(ak.sum(mask))
    fraction = 100 * count / total_events if total_events else 0.0
    return count, fraction


def pdg_to_label(pdg_id):
    label = PDG_LABELS.get(abs(pdg_id), str(abs(pdg_id)))
    if pdg_id < 0:
        if label.endswith("+"):
            return f"{label[:-1]}-"
        if label.endswith("-"):
            return f"{label[:-1]}+"
        return f"{label}~"
    return label


def find_banner_file(run_dir):
    banner_files = sorted(Path(run_dir).glob("*_banner.txt"))
    return banner_files[0] if banner_files else None


def extract_process_text(banner_text):
    lines = banner_text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("generate "):
            continue

        process_parts = [stripped[len("generate ") :].rstrip("\\").strip()]
        while stripped.endswith("\\") and index + 1 < len(lines):
            index += 1
            stripped = lines[index].strip()
            process_parts.append(stripped.rstrip("\\").strip())

        return " ".join(part for part in process_parts if part)

    return None


def read_run_info(run_dir):
    banner_path = find_banner_file(run_dir)
    if banner_path is None:
        return None

    banner_text = banner_path.read_text(encoding="utf-8", errors="ignore")

    mass_match = re.search(r"^\s*36\s+([^\s#]+)\s+#\s*ma\b", banner_text, re.MULTILINE)
    tanphi_match = re.search(r"^\s*1\s+([^\s#]+)\s+#\s*tanphi\b", banner_text, re.MULTILINE)
    decay_match = re.search(r"^DECAY\s+36\s+([^\s#]+)\s+#.*$", banner_text, re.MULTILINE)
    cross_section_match = re.search(
        r"Integrated weight \(pb\)\s*:\s*([^\s#]+)", banner_text
    )

    branching_ratios = []
    if decay_match:
        decay_start = decay_match.end()
        started_decay_table = False
        for line in banner_text[decay_start:].splitlines():
            stripped = line.strip()
            if not stripped:
                if started_decay_table:
                    break
                continue

            if stripped.startswith("<") or stripped.startswith("DECAY"):
                break

            parts = stripped.split()
            if len(parts) < 4:
                continue

            started_decay_table = True
            branching_ratio = float(parts[0])
            n_children = int(parts[1])
            daughters = [int(pdg_id) for pdg_id in parts[2 : 2 + n_children]]
            channel = " ".join(pdg_to_label(pdg_id) for pdg_id in daughters)
            branching_ratios.append((branching_ratio, channel))

    return {
        "process": extract_process_text(banner_text),
        "mass": float(mass_match.group(1)) if mass_match else None,
        "tanphi": float(tanphi_match.group(1)) if tanphi_match else None,
        "width": float(decay_match.group(1)) if decay_match else None,
        "cross_section_pb": float(cross_section_match.group(1)) if cross_section_match else None,
        "branching_ratios": branching_ratios,
    }


def build_normalization(mode, luminosity):
    if mode == "mc":
        return {
            "ylabel": "Monte Carlo Events",
            "mode_label": "Monte Carlo events",
            "figure_title": "Selected-event kinematics (stacked by process)",
            "luminosity_fb": None,
            "yield_units": "events",
        }

    if mode == "xsec":
        return {
            "ylabel": "Cross section [pb]",
            "mode_label": "Cross-section weighted",
            "figure_title": "Selected-event kinematics (stacked, cross-section weighted)",
            "luminosity_fb": None,
            "yield_units": "pb",
        }

    return {
        "ylabel": f"Expected events (L = {luminosity:g} fb^-1)",
        "mode_label": "Luminosity scaled",
        "figure_title": f"Selected-event kinematics (stacked, L = {luminosity:g} fb^-1)",
        "luminosity_fb": luminosity,
        "yield_units": "events",
    }


def build_event_weight(mode, total_events, run_info, luminosity, run_dir):
    if total_events <= 0:
        raise SystemExit(f"No events were found in {run_dir / ROOT_FILENAME}.")

    if mode == "mc":
        return 1.0

    cross_section_pb = run_info["cross_section_pb"] if run_info else None
    if cross_section_pb is None:
        raise SystemExit(
            f"Could not find the run cross section in {run_dir}. "
            "Modes 2 and 3 require a *_banner.txt file with the integrated weight."
        )

    if mode == "xsec":
        return cross_section_pb / total_events

    return cross_section_pb * luminosity * 1000.0 / total_events


def sample_label_from_dir(run_dir):
    name = Path(run_dir).name
    if name.startswith("run_"):
        return name[4:]
    return name


def analyze_sample(run_dir, color, mode, luminosity):
    run_path = Path(run_dir)
    if not run_path.is_dir():
        raise SystemExit(f"Run directory not found: {run_path}")

    root_path = run_path / ROOT_FILENAME
    if not root_path.is_file():
        raise SystemExit(f"ROOT file not found: {root_path}")

    run_info = read_run_info(run_path)

    branches = [
        "Jet/Jet.BTag",
        "Jet/Jet.PT",
        "Electron/Electron.Charge",
        "Electron/Electron.PT",
        "Muon/Muon.Charge",
        "Muon/Muon.PT",
        "MissingET/MissingET.MET",
    ]

    with uproot.open(root_path) as root_file:
        tree = root_file[TREE_NAME]
        total_events = tree.num_entries
        arrays = tree.arrays(branches, library="ak")

    event_weight = build_event_weight(mode, total_events, run_info, luminosity, run_path)

    jet_is_btag = arrays["Jet/Jet.BTag"] > 0
    jet_pt = arrays["Jet/Jet.PT"]

    electron_charge = arrays["Electron/Electron.Charge"]
    electron_pt = arrays["Electron/Electron.PT"]
    muon_charge = arrays["Muon/Muon.Charge"]
    muon_pt = arrays["Muon/Muon.PT"]

    missing_et = ak.to_numpy(ak.fill_none(ak.firsts(arrays["MissingET/MissingET.MET"]), 0.0))

    all_lepton_charge = ak.concatenate([electron_charge, muon_charge], axis=1)
    all_lepton_pt = ak.concatenate([electron_pt, muon_pt], axis=1)

    n_electrons = ak.num(electron_charge, axis=1)
    n_muons = ak.num(muon_charge, axis=1)
    n_leptons = ak.num(all_lepton_charge, axis=1)

    padded_charge = ak.pad_none(all_lepton_charge, 2)
    charge_1 = ak.fill_none(padded_charge[:, 0], 0)
    charge_2 = ak.fill_none(padded_charge[:, 1], 0)

    exactly_two_btag_jets = ak.sum(jet_is_btag, axis=1) == 2
    exactly_two_leptons = n_leptons == 2
    same_sign_leptons = exactly_two_leptons & (charge_1 * charge_2 > 0)
    selected_events = exactly_two_btag_jets & same_sign_leptons

    count_two_btag_jets, percent_two_btag_jets = count_and_fraction(
        exactly_two_btag_jets, total_events
    )
    same_sign_count, same_sign_percent = count_and_fraction(same_sign_leptons, total_events)
    selected_count, selected_percent = count_and_fraction(selected_events, total_events)

    negative_pair = same_sign_leptons & (charge_1 < 0)
    positive_pair = same_sign_leptons & (charge_1 > 0)

    ee_minus = negative_pair & (n_electrons == 2)
    emu_minus = negative_pair & (n_electrons == 1) & (n_muons == 1)
    mumu_minus = negative_pair & (n_muons == 2)

    ee_plus = positive_pair & (n_electrons == 2)
    emu_plus = positive_pair & (n_electrons == 1) & (n_muons == 1)
    mumu_plus = positive_pair & (n_muons == 2)

    selected_missing_et = missing_et[selected_events]

    selected_jet_pt = jet_pt[selected_events]
    selected_jet_is_btag = jet_is_btag[selected_events]

    bjet_pt = ak.sort(selected_jet_pt[selected_jet_is_btag], axis=1, ascending=False)
    leading_bjet_pt = ak.to_numpy(bjet_pt[:, 0])
    subleading_bjet_pt = ak.to_numpy(bjet_pt[:, 1])

    selected_lepton_pt = ak.sort(all_lepton_pt[selected_events], axis=1, ascending=False)
    leading_lepton_pt = ak.to_numpy(selected_lepton_pt[:, 0])
    subleading_lepton_pt = ak.to_numpy(selected_lepton_pt[:, 1])

    non_bjet_pt = ak.sort(selected_jet_pt[~selected_jet_is_btag], axis=1, ascending=False)
    has_non_bjet_1 = ak.num(non_bjet_pt, axis=1) >= 1
    has_non_bjet_2 = ak.num(non_bjet_pt, axis=1) >= 2
    leading_non_bjet_pt = ak.to_numpy(non_bjet_pt[has_non_bjet_1][:, 0])
    subleading_non_bjet_pt = ak.to_numpy(non_bjet_pt[has_non_bjet_2][:, 1])

    counts = {
        "two_btag": count_two_btag_jets,
        "same_sign": same_sign_count,
        "selected": selected_count,
        "ee_minus": int(ak.sum(ee_minus)),
        "emu_minus": int(ak.sum(emu_minus)),
        "mumu_minus": int(ak.sum(mumu_minus)),
        "ee_plus": int(ak.sum(ee_plus)),
        "emu_plus": int(ak.sum(emu_plus)),
        "mumu_plus": int(ak.sum(mumu_plus)),
        "non_bjet_ge1": int(ak.sum(has_non_bjet_1)),
        "non_bjet_ge2": int(ak.sum(has_non_bjet_2)),
    }
    fractions = {
        "two_btag": percent_two_btag_jets,
        "same_sign": same_sign_percent,
        "selected": selected_percent,
    }
    histograms = {
        "leading_bjet_pt": leading_bjet_pt,
        "subleading_bjet_pt": subleading_bjet_pt,
        "leading_lepton_pt": leading_lepton_pt,
        "subleading_lepton_pt": subleading_lepton_pt,
        "leading_non_bjet_pt": leading_non_bjet_pt,
        "subleading_non_bjet_pt": subleading_non_bjet_pt,
        "missing_et": selected_missing_et,
    }

    return SampleResult(
        run_dir=run_path,
        label=sample_label_from_dir(run_path),
        color=color,
        total_events=total_events,
        run_info=run_info,
        event_weight=event_weight,
        counts=counts,
        fractions=fractions,
        histograms=histograms,
        selected_yield=selected_count * event_weight,
    )


def collect_common_value(samples, key):
    values = []
    for sample in samples:
        if sample.run_info is None:
            return None
        values.append(sample.run_info.get(key))

    first_value = values[0]
    if all(value == first_value for value in values):
        return first_value
    return None


def collect_common_branching_ratios(samples):
    branching_ratios = []
    for sample in samples:
        if sample.run_info is None:
            return None
        branching_ratios.append(sample.run_info.get("branching_ratios", []))

    first_value = branching_ratios[0]
    if all(value == first_value for value in branching_ratios):
        return first_value
    return None


def format_cross_section(sample):
    if sample.run_info is None or sample.run_info["cross_section_pb"] is None:
        return "unavailable"
    return f"{sample.run_info['cross_section_pb']:.9g} pb"


def format_process(sample):
    process = sample.run_info["process"] if sample.run_info else None
    if not process:
        return "unavailable"
    return textwrap.shorten(process, width=72, placeholder="...")


def compute_hist_bins(data_list, n_bins=30):
    non_empty = [values for values in data_list if len(values) > 0]
    if not non_empty:
        return None

    combined_values = np.concatenate(non_empty)
    if np.allclose(combined_values.min(), combined_values.max()):
        center = combined_values[0]
        delta = 1.0 if center == 0 else max(abs(center) * 0.05, 1.0)
        return np.linspace(center - delta, center + delta, n_bins + 1)

    return np.histogram_bin_edges(combined_values, bins=n_bins)


def make_stacked_hist(ax, samples, plot_key, xlabel, ylabel, log_y=False):
    values_list = [sample.histograms[plot_key] for sample in samples]
    bins = compute_hist_bins(values_list)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if bins is None:
        ax.text(
            0.5,
            0.5,
            "No selected events",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    weights_list = [
        np.full(len(values), sample.event_weight, dtype=float)
        for sample, values in zip(samples, values_list)
    ]
    total_counts = np.sum(
        [
            np.histogram(values, bins=bins, weights=weights)[0]
            for values, weights in zip(values_list, weights_list)
        ],
        axis=0,
    )

    ax.hist(
        values_list,
        bins=bins,
        weights=weights_list,
        stacked=True,
        color=[sample.color for sample in samples],
        edgecolor="black",
        linewidth=0.7,
    )

    if log_y and np.any(total_counts > 0):
        min_positive = np.min(total_counts[total_counts > 0])
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(min_positive * 0.5, np.finfo(float).tiny))


def build_summary_text(samples, normalization, mode):
    total_selected_events = sum(sample.counts["selected"] for sample in samples)
    total_selected_yield = sum(sample.selected_yield for sample in samples)
    shared_mass = collect_common_value(samples, "mass")
    shared_tanphi = collect_common_value(samples, "tanphi")
    shared_branching_ratios = collect_common_branching_ratios(samples)

    lines = [
        "Combined run information",
        f"Processes: {', '.join(sample.label for sample in samples)}",
        f"Normalization: {normalization['mode_label']}",
        f"Selected events (raw sum): {total_selected_events}",
    ]

    if mode != "mc":
        lines.append(
            f"Selected yield (sum): {total_selected_yield:.6g} {normalization['yield_units']}"
        )

    if normalization["luminosity_fb"] is not None:
        lines.append(f"Luminosity: {normalization['luminosity_fb']:g} fb^-1")

    if shared_mass is not None or shared_tanphi is not None or shared_branching_ratios:
        lines.append("")
        lines.append("Shared pseudoscalar information")
        lines.append(
            f"m_a: {shared_mass:g} GeV" if shared_mass is not None else "m_a: unavailable"
        )
        lines.append(
            f"tanphi: {shared_tanphi:g}" if shared_tanphi is not None else "tanphi: unavailable"
        )

        if shared_branching_ratios:
            lines.append("BR(a -> X):")
            for branching_ratio, channel in shared_branching_ratios:
                lines.append(f"  {channel}: {100 * branching_ratio:.3f}%")

    for sample in samples:
        lines.extend(
            [
                "",
                f"{COLOR_NAMES[sample.color]} | {sample.label}",
                f"Process: {format_process(sample)}",
                f"Events / xsec: {sample.total_events} / {format_cross_section(sample)}",
                (
                    f"2b / SSll / both: "
                    f"{sample.counts['two_btag']} ({sample.fractions['two_btag']:.2f}%) / "
                    f"{sample.counts['same_sign']} ({sample.fractions['same_sign']:.2f}%) / "
                    f"{sample.counts['selected']} ({sample.fractions['selected']:.2f}%)"
                ),
                (
                    f"SS-: ee={sample.counts['ee_minus']}, emu={sample.counts['emu_minus']}, "
                    f"mumu={sample.counts['mumu_minus']}"
                ),
                (
                    f"SS+: ee={sample.counts['ee_plus']}, emu={sample.counts['emu_plus']}, "
                    f"mumu={sample.counts['mumu_plus']}"
                ),
                (
                    f"non-b jets >=1 / >=2: "
                    f"{sample.counts['non_bjet_ge1']} / {sample.counts['non_bjet_ge2']}"
                ),
            ]
        )

        if mode != "mc":
            lines.append(f"Per-event weight: {sample.event_weight:.9g}")
            lines.append(
                f"Selected yield: {sample.selected_yield:.6g} {normalization['yield_units']}"
            )

    return "\n".join(lines)


def print_sample_report(sample, normalization, mode):
    print(f"[{sample.label}]")
    print(f"  Directory: {sample.run_dir}")
    print(f"  Process: {sample.run_info['process'] if sample.run_info else 'unavailable'}")
    print(f"  Events: {sample.total_events}")
    print(f"  Cross section: {format_cross_section(sample)}")
    print(
        "  2 b-tagged jets / 2 same-sign leptons / both: "
        f"{sample.counts['two_btag']} ({sample.fractions['two_btag']:.2f}%) / "
        f"{sample.counts['same_sign']} ({sample.fractions['same_sign']:.2f}%) / "
        f"{sample.counts['selected']} ({sample.fractions['selected']:.2f}%)"
    )
    print(
        "  SS- channels (ee, emu, mumu): "
        f"{sample.counts['ee_minus']}, {sample.counts['emu_minus']}, {sample.counts['mumu_minus']}"
    )
    print(
        "  SS+ channels (ee, emu, mumu): "
        f"{sample.counts['ee_plus']}, {sample.counts['emu_plus']}, {sample.counts['mumu_plus']}"
    )
    print(
        "  Events with >=1 non-b jet / >=2 non-b jets: "
        f"{sample.counts['non_bjet_ge1']} / {sample.counts['non_bjet_ge2']}"
    )
    if mode != "mc":
        print(f"  Per-event weight: {sample.event_weight:.9g}")
        print(
            f"  Selected yield: {sample.selected_yield:.6g} {normalization['yield_units']}"
        )
    print()


def plot_samples(samples, normalization, mode, output, log_y=False):
    fig, axes = plt.subplots(4, 2, figsize=(14, 20))

    plot_axes = [
        axes[0, 0],
        axes[0, 1],
        axes[1, 0],
        axes[1, 1],
        axes[2, 0],
        axes[2, 1],
        axes[3, 0],
    ]

    for ax, (plot_key, xlabel) in zip(plot_axes, PLOT_SPECS):
        make_stacked_hist(
            ax,
            samples,
            plot_key,
            xlabel,
            normalization["ylabel"],
            log_y=log_y,
        )

    summary_text = build_summary_text(samples, normalization, mode)
    axes[3, 1].axis("off")
    axes[3, 1].text(
        0.0,
        1.0,
        summary_text,
        ha="left",
        va="top",
        fontsize=6.8,
        linespacing=1.05,
        family="monospace",
        transform=axes[3, 1].transAxes,
        bbox=dict(facecolor="white", alpha=0.92, edgecolor="black"),
        wrap=True,
    )

    fig.suptitle(normalization["figure_title"], fontsize=15)
    fig.legend(
        handles=[Patch(facecolor=sample.color, edgecolor="black", label=sample.label) for sample in samples],
        loc="upper center",
        ncol=len(samples),
        frameon=True,
        bbox_to_anchor=(0.5, 0.955),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    args = parse_arguments()
    normalization = build_normalization(args.mode, args.luminosity)
    samples = [
        analyze_sample(run_dir, color, args.mode, args.luminosity)
        for run_dir, color in zip(args.runs, SAMPLE_COLORS)
    ]

    print(f"Histogram normalization: {normalization['mode_label']}")
    if normalization["luminosity_fb"] is not None:
        print(f"Luminosity: {normalization['luminosity_fb']:g} fb^-1")
    print(f"Y-axis scale: {'logarithmic' if args.log_y else 'linear'}")
    print()

    for sample in samples:
        print_sample_report(sample, normalization, args.mode)

    total_selected_events = sum(sample.counts["selected"] for sample in samples)
    print(f"Combined selected events (raw sum): {total_selected_events}")
    if args.mode != "mc":
        total_selected_yield = sum(sample.selected_yield for sample in samples)
        print(
            f"Combined selected yield: {total_selected_yield:.6g} "
            f"{normalization['yield_units']}"
        )
    print()

    plot_samples(samples, normalization, args.mode, args.output, log_y=args.log_y)
    print(f"Saved histograms to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
