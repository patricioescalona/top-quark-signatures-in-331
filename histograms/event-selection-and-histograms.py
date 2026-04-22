import argparse
from pathlib import Path
import re

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot


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
            "Apply the Delphes-level event selection and produce kinematic "
            "histograms with optional normalization by cross section or luminosity."
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
        "--output",
        default="selected_kinematics.png",
        help="Output image filename. Default: selected_kinematics.png",
    )
    args = parser.parse_args()

    if args.luminosity <= 0:
        parser.error("--luminosity must be positive.")

    return args


def count_and_fraction(mask, total_events):
    count = int(ak.sum(mask))
    fraction = 100 * count / total_events if total_events else 0.0
    return count, fraction


def make_hist(ax, values, xlabel, ylabel, event_weight=None):
    hist_kwargs = {"bins": 30, "edgecolor": "black"}
    if event_weight is not None:
        hist_kwargs["weights"] = np.full(len(values), event_weight)

    ax.hist(values, **hist_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def pdg_to_label(pdg_id):
    label = PDG_LABELS.get(abs(pdg_id), str(abs(pdg_id)))
    if pdg_id < 0:
        if label.endswith("+"):
            return f"{label[:-1]}-"
        if label.endswith("-"):
            return f"{label[:-1]}+"
        return f"{label}~"
    return label


def find_banner_file():
    banner_files = sorted(Path(".").glob("*_banner.txt"))
    return banner_files[0] if banner_files else None


def read_run_info():
    banner_path = find_banner_file()
    if banner_path is None:
        return None

    banner_text = banner_path.read_text(encoding="utf-8", errors="ignore")

    process_match = re.search(r"^\s*generate\s+(.+)$", banner_text, re.MULTILINE)
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
        "process": process_match.group(1).strip() if process_match else None,
        "mass": float(mass_match.group(1)) if mass_match else None,
        "tanphi": float(tanphi_match.group(1)) if tanphi_match else None,
        "width": float(decay_match.group(1)) if decay_match else None,
        "cross_section_pb": float(cross_section_match.group(1)) if cross_section_match else None,
        "branching_ratios": branching_ratios,
    }


def build_normalization(mode, total_events, run_info, luminosity):
    if total_events <= 0:
        raise SystemExit("No events were found in tag_1_delphes_events.root.")

    cross_section_pb = run_info["cross_section_pb"] if run_info else None

    if mode == "mc":
        return {
            "event_weight": None,
            "ylabel": "Monte Carlo Events",
            "mode_label": "Monte Carlo events",
            "figure_title": "Selected-event kinematics",
            "luminosity_fb": None,
        }

    if cross_section_pb is None:
        raise SystemExit(
            "Could not find the run cross section in *_banner.txt. "
            "Modes 2 and 3 require it."
        )

    if mode == "xsec":
        return {
            "event_weight": cross_section_pb / total_events,
            "ylabel": "Cross section [pb]",
            "mode_label": "Cross-section weighted",
            "figure_title": "Selected-event kinematics (cross-section weighted)",
            "luminosity_fb": None,
        }

    return {
        "event_weight": cross_section_pb * luminosity * 1000.0 / total_events,
        "ylabel": f"Expected events (L = {luminosity:g} fb^-1)",
        "mode_label": "Luminosity scaled",
        "figure_title": f"Selected-event kinematics (L = {luminosity:g} fb^-1)",
        "luminosity_fb": luminosity,
    }


def build_summary_text(total_events, run_info, normalization, selection_lines):
    run_info_lines = [
        "Run information",
        f"Number of events: {total_events}",
        f"Normalization: {normalization['mode_label']}",
    ]

    if run_info is None:
        run_info_lines.append("Process: unavailable")
        run_info_lines.append("Cross section: unavailable")
        run_info_lines.append("Pseudoscalar mass: unavailable")
        run_info_lines.append("tanphi: unavailable")
        run_info_lines.append("BR(a -> X): unavailable")
    else:
        process_text = run_info["process"] or "unavailable"
        cross_section_text = (
            f"{run_info['cross_section_pb']:.9g} pb"
            if run_info["cross_section_pb"] is not None
            else "unavailable"
        )
        mass_text = f"{run_info['mass']:g} GeV" if run_info["mass"] is not None else "unavailable"
        tanphi_text = f"{run_info['tanphi']:g}" if run_info["tanphi"] is not None else "unavailable"

        run_info_lines.append(f"Process: {process_text}")
        run_info_lines.append(f"Cross section: {cross_section_text}")
        run_info_lines.append(f"Pseudoscalar mass: {mass_text}")
        run_info_lines.append(f"tanphi: {tanphi_text}")

        if normalization["luminosity_fb"] is not None:
            run_info_lines.append(f"Luminosity: {normalization['luminosity_fb']:g} fb^-1")

        run_info_lines.append("BR(a -> X):")
        if run_info["branching_ratios"]:
            for branching_ratio, channel in run_info["branching_ratios"]:
                run_info_lines.append(f"  {channel}: {100 * branching_ratio:.3f}%")
        else:
            run_info_lines.append("  unavailable")

    return "\n".join(run_info_lines + ["", "Selection details"] + selection_lines)


def main():
    args = parse_arguments()

    root_file = uproot.open("tag_1_delphes_events.root")
    tree = root_file["Delphes;1"]
    total_events = tree.num_entries
    run_info = read_run_info()
    normalization = build_normalization(args.mode, total_events, run_info, args.luminosity)

    branches = [
        "Jet/Jet.BTag",
        "Jet/Jet.PT",
        "Electron/Electron.Charge",
        "Electron/Electron.PT",
        "Muon/Muon.Charge",
        "Muon/Muon.PT",
        "MissingET/MissingET.MET",
    ]
    arrays = tree.arrays(branches, library="ak")

    jet_is_btag = arrays["Jet/Jet.BTag"] > 0
    jet_pt = arrays["Jet/Jet.PT"]

    electron_charge = arrays["Electron/Electron.Charge"]
    electron_pt = arrays["Electron/Electron.PT"]
    muon_charge = arrays["Muon/Muon.Charge"]
    muon_pt = arrays["Muon/Muon.PT"]

    missing_et = ak.to_numpy(ak.firsts(arrays["MissingET/MissingET.MET"]))

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

    selection_lines = [
        f"Total events: {total_events}",
        f"2 b-tagged jets: {count_two_btag_jets} ({percent_two_btag_jets:.2f}%)",
        f"2 same-sign leptons: {same_sign_count} ({same_sign_percent:.2f}%)",
        f"Both filters: {selected_count} ({selected_percent:.2f}%)",
        "",
        f"e- e-  : {int(ak.sum(ee_minus))}",
        f"e- mu- : {int(ak.sum(emu_minus))}",
        f"mu- mu-: {int(ak.sum(mumu_minus))}",
        f"e+ e+  : {int(ak.sum(ee_plus))}",
        f"e+ mu+ : {int(ak.sum(emu_plus))}",
        f"mu+ mu+: {int(ak.sum(mumu_plus))}",
    ]

    print(
        f"From a total of {total_events} events, there are {count_two_btag_jets} "
        "satisfying the applied filter: exactly two b-tagged jets at Delphes level."
    )
    print(f"Percentage: {percent_two_btag_jets:.2f}%")
    print()

    print(
        f"From a total of {total_events} events, there are {same_sign_count} "
        "satisfying the applied filter: exactly two same-charge reconstructed leptons "
        "(electrons or muons) at Delphes level."
    )
    print(f"Percentage: {same_sign_percent:.2f}%")
    print()

    print("Breakdown by same-sign dilepton channel:")
    print(f"e- e-  : {int(ak.sum(ee_minus))}")
    print(f"e- mu- : {int(ak.sum(emu_minus))}")
    print(f"mu- mu-: {int(ak.sum(mumu_minus))}")
    print(f"e+ e+  : {int(ak.sum(ee_plus))}")
    print(f"e+ mu+ : {int(ak.sum(emu_plus))}")
    print(f"mu+ mu+: {int(ak.sum(mumu_plus))}")
    print()

    print(
        f"From a total of {total_events} events, there are {selected_count} "
        "satisfying both applied filters: exactly two b-tagged jets and exactly two "
        "same-charge reconstructed leptons at Delphes level."
    )
    print(f"Percentage: {selected_percent:.2f}%")

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

    selection_lines.extend(
        [
            "",
            f"Events with >=1 non-b jet: {int(ak.sum(has_non_bjet_1))}",
            f"Events with >=2 non-b jets: {int(ak.sum(has_non_bjet_2))}",
        ]
    )
    summary_text = build_summary_text(total_events, run_info, normalization, selection_lines)

    fig, axes = plt.subplots(4, 2, figsize=(12, 18))

    plot_items = [
        (axes[0, 0], leading_bjet_pt, "Leading b-jet pT [GeV]"),
        (axes[0, 1], subleading_bjet_pt, "Subleading b-jet pT [GeV]"),
        (axes[1, 0], leading_lepton_pt, "Leading lepton pT [GeV]"),
        (axes[1, 1], subleading_lepton_pt, "Subleading lepton pT [GeV]"),
        (axes[2, 0], leading_non_bjet_pt, "Leading non-b jet pT [GeV]"),
        (axes[2, 1], subleading_non_bjet_pt, "Subleading non-b jet pT [GeV]"),
        (axes[3, 0], selected_missing_et, "Missing transverse momentum [GeV]"),
    ]

    for ax, values, xlabel in plot_items:
        make_hist(ax, values, xlabel, normalization["ylabel"], normalization["event_weight"])

    fig.suptitle(normalization["figure_title"], fontsize=14)
    axes[3, 1].axis("off")
    axes[3, 1].text(
        0.0,
        1.0,
        summary_text,
        ha="left",
        va="top",
        fontsize=8,
        linespacing=1.1,
        family="monospace",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="black"),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.output, dpi=150)

    print()
    print(f"Histogram normalization: {normalization['mode_label']}")
    if run_info and run_info["cross_section_pb"] is not None and normalization["event_weight"] is not None:
        print(f"Cross section used for weighting: {run_info['cross_section_pb']:.9g} pb")
    if normalization["luminosity_fb"] is not None:
        print(f"Luminosity: {normalization['luminosity_fb']:g} fb^-1")
    if normalization["event_weight"] is not None:
        print(f"Per-event histogram weight: {normalization['event_weight']:.9g}")
    print()
    print(f"Saved histograms to {args.output}")


if __name__ == "__main__":
    main()
