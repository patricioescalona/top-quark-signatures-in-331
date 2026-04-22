import argparse  # Read command-line options such as the normalization mode and output name.
from pathlib import Path  # Work with file paths in a clean and platform-independent way.
import re  # Search the banner text with regular expressions.

import awkward as ak  # Handle event data where each event can contain a different number of objects.
import matplotlib.pyplot as plt  # Create the histogram figure.
import numpy as np  # Provide standard numerical arrays and helper functions.
import uproot  # Read ROOT files directly in Python.


PDG_LABELS = {  # Translate PDG particle ID numbers into short human-readable labels.
    1: "d",  # Down quark.
    2: "u",  # Up quark.
    3: "s",  # Strange quark.
    4: "c",  # Charm quark.
    5: "b",  # Bottom quark.
    6: "t",  # Top quark.
    11: "e-",  # Electron.
    12: "ve",  # Electron neutrino.
    13: "mu-",  # Muon.
    14: "vm",  # Muon neutrino.
    15: "tau-",  # Tau lepton.
    16: "vt",  # Tau neutrino.
    21: "g",  # Gluon.
    22: "gamma",  # Photon.
    23: "z",  # Z boson.
    24: "w+",  # W boson with positive charge.
    25: "h",  # Higgs boson.
    36: "a",  # Pseudoscalar a in this model.
}  # End of the PDG-label lookup table.


def parse_mode(value):  # Convert the user input for --mode into one standard internal label.
    normalized_value = value.strip().lower()  # Remove spaces and ignore upper/lower-case differences.
    mode_map = {  # Allow several equivalent names for each normalization mode.
        "1": "mc",  # Numeric shortcut for raw event counts.
        "mc": "mc",  # Standard name for raw Monte Carlo counts.
        "raw": "mc",  # Alternative name for raw Monte Carlo counts.
        "events": "mc",  # Another alternative for raw event counts.
        "2": "xsec",  # Numeric shortcut for cross-section weighting.
        "xsec": "xsec",  # Standard name for cross-section weighting.
        "cross-section": "xsec",  # Hyphenated alternative spelling.
        "cross_section": "xsec",  # Underscore alternative spelling.
        "crosssection": "xsec",  # Compact alternative spelling.
        "3": "lumi",  # Numeric shortcut for luminosity scaling.
        "lumi": "lumi",  # Standard name for luminosity scaling.
        "luminosity": "lumi",  # Full-word alternative spelling.
    }  # End of the accepted mode names.

    if normalized_value not in mode_map:  # Stop early if the user passed an unsupported mode.
        raise argparse.ArgumentTypeError(  # Tell argparse to print a clean error message.
            "Mode must be one of: 1/mc, 2/xsec, or 3/lumi."  # Explain the allowed choices.
        )  # End the error construction.

    return mode_map[normalized_value]  # Return the standard label used by the rest of the script.


def parse_arguments():  # Define and read the command-line arguments for the script.
    parser = argparse.ArgumentParser(  # Create the top-level argument parser.
        description=(  # Provide the help text shown by --help.
            "Apply the Delphes-level event selection and produce kinematic "  # First half of the help description.
            "histograms with optional normalization by cross section or luminosity."  # Second half of the help description.
        )  # End the description string.
    )  # Finish creating the parser.
    parser.add_argument(  # Add the option that chooses the histogram normalization mode.
        "--mode",  # Name of the command-line flag.
        type=parse_mode,  # Convert user input into one of the internal mode labels.
        default="mc",  # Use raw event counts if the user does not choose a mode.
        help=(  # Help text shown for this option.
            "Histogram mode: 1 or mc for raw Monte Carlo events, "  # Explain mode 1.
            "2 or xsec for cross-section weighting, "  # Explain mode 2.
            "3 or lumi for luminosity scaling. Default: mc."  # Explain mode 3 and the default choice.
        ),  # End the help text.
    )  # Finish adding the --mode option.
    parser.add_argument(  # Add the option for the integrated luminosity value.
        "--luminosity",  # Name of the command-line flag.
        type=float,  # Convert the input string into a floating-point number.
        default=130.0,  # Use 130 inverse femtobarns by default.
        help=(  # Help text shown for this option.
            "Integrated luminosity in fb^-1 used when --mode is 3 or lumi. "  # Explain when this value matters.
            "Default: 130."  # State the default value.
        ),  # End the help text.
    )  # Finish adding the --luminosity option.
    parser.add_argument(  # Add the option for the output image filename.
        "--output",  # Name of the command-line flag.
        default="selected_kinematics.png",  # Use this filename if the user does not specify another one.
        help="Output image filename. Default: selected_kinematics.png",  # Explain what this argument controls.
    )  # Finish adding the --output option.
    args = parser.parse_args()  # Read the command-line inputs and store them in an object.

    if args.luminosity <= 0:  # Check that the luminosity is physically sensible.
        parser.error("--luminosity must be positive.")  # Stop with a user-friendly error message.

    return args  # Give the validated argument object back to the caller.


def count_and_fraction(mask, total_events):  # Count how many events pass a condition and convert that to a percentage.
    count = int(ak.sum(mask))  # Sum a boolean mask, where True counts as 1 and False counts as 0.
    fraction = 100 * count / total_events if total_events else 0.0  # Avoid division by zero when there are no events.
    return count, fraction  # Return both the event count and the percentage.


def make_hist(ax, values, xlabel, ylabel, event_weight=None):  # Draw one histogram on a given subplot axis.
    hist_kwargs = {"bins": 30, "edgecolor": "black"}  # Use 30 bins and black edges for readability.
    if event_weight is not None:  # Only apply weights in the weighted histogram modes.
        hist_kwargs["weights"] = np.full(len(values), event_weight)  # Give each entry the same per-event weight.

    ax.hist(values, **hist_kwargs)  # Draw the histogram with the chosen settings.
    ax.set_xlabel(xlabel)  # Label the horizontal axis with the plotted observable.
    ax.set_ylabel(ylabel)  # Label the vertical axis with the chosen normalization.


def pdg_to_label(pdg_id):  # Convert one PDG ID into a particle label, including antiparticles.
    label = PDG_LABELS.get(abs(pdg_id), str(abs(pdg_id)))  # Look up the absolute PDG code, or fall back to the number itself.
    if pdg_id < 0:  # Negative PDG IDs usually correspond to antiparticles.
        if label.endswith("+"):  # If the particle label ends with a plus sign, flip the charge.
            return f"{label[:-1]}-"  # Replace the plus sign with a minus sign.
        if label.endswith("-"):  # If the particle label ends with a minus sign, flip the charge.
            return f"{label[:-1]}+"  # Replace the minus sign with a plus sign.
        return f"{label}~"  # For neutral antiparticles, append a tilde.
    return label  # For positive PDG IDs, return the label unchanged.


def find_banner_file():  # Search the current directory for a MadGraph banner file.
    banner_files = sorted(Path(".").glob("*_banner.txt"))  # Collect matching banner filenames in sorted order.
    return banner_files[0] if banner_files else None  # Return the first match, or None if no banner exists.


def read_run_info():  # Extract process metadata, parameters, and branching ratios from the banner file.
    banner_path = find_banner_file()  # Locate the banner file if it exists.
    if banner_path is None:  # Handle the case where no banner file is available.
        return None  # Signal that run information could not be read.

    banner_text = banner_path.read_text(encoding="utf-8", errors="ignore")  # Read the banner text while tolerating unusual characters.

    process_match = re.search(r"^\s*generate\s+(.+)$", banner_text, re.MULTILINE)  # Find the generated hard-scattering process.
    mass_match = re.search(r"^\s*36\s+([^\s#]+)\s+#\s*ma\b", banner_text, re.MULTILINE)  # Find the pseudoscalar mass parameter ma.
    tanphi_match = re.search(r"^\s*1\s+([^\s#]+)\s+#\s*tanphi\b", banner_text, re.MULTILINE)  # Find the tanphi model parameter.
    decay_match = re.search(r"^DECAY\s+36\s+([^\s#]+)\s+#.*$", banner_text, re.MULTILINE)  # Find the width entry for particle 36.
    cross_section_match = re.search(  # Search for the integrated event weight reported in pb.
        r"Integrated weight \(pb\)\s*:\s*([^\s#]+)", banner_text  # Capture the numerical cross section value.
    )  # End the cross-section search.

    branching_ratios = []  # Store each decay channel as a pair of branching ratio and readable label.
    if decay_match:  # Only try to read decay channels if the DECAY line was found.
        decay_start = decay_match.end()  # Start reading just after the DECAY line.
        started_decay_table = False  # Track whether we have entered the actual list of decay channels.
        for line in banner_text[decay_start:].splitlines():  # Loop over each line after the DECAY header.
            stripped = line.strip()  # Remove surrounding spaces so pattern checks are simpler.
            if not stripped:  # Blank lines may appear before or after the table.
                if started_decay_table:  # If the table already started, a blank line means it is over.
                    break  # Stop reading the decay table.
                continue  # Otherwise ignore leading blank lines.

            if stripped.startswith("<") or stripped.startswith("DECAY"):  # Stop if a new block begins.
                break  # Leave the decay-table loop.

            parts = stripped.split()  # Split the line into columns.
            if len(parts) < 4:  # Skip malformed or incomplete lines.
                continue  # Move on to the next line.

            started_decay_table = True  # Mark that we are now inside the decay table.
            branching_ratio = float(parts[0])  # First column is the branching ratio.
            n_children = int(parts[1])  # Second column is the number of daughter particles.
            daughters = [int(pdg_id) for pdg_id in parts[2 : 2 + n_children]]  # Read the daughter PDG IDs.
            channel = " ".join(pdg_to_label(pdg_id) for pdg_id in daughters)  # Convert the daughter IDs into a readable final state.
            branching_ratios.append((branching_ratio, channel))  # Save this decay channel.

    return {  # Package all extracted run information into one dictionary.
        "process": process_match.group(1).strip() if process_match else None,  # Hard process string, if found.
        "mass": float(mass_match.group(1)) if mass_match else None,  # Pseudoscalar mass in GeV, if found.
        "tanphi": float(tanphi_match.group(1)) if tanphi_match else None,  # tanphi parameter, if found.
        "width": float(decay_match.group(1)) if decay_match else None,  # Total decay width, if found.
        "cross_section_pb": float(cross_section_match.group(1)) if cross_section_match else None,  # Cross section in pb, if found.
        "branching_ratios": branching_ratios,  # List of decay channels and branching ratios.
    }  # Return the run-information dictionary.


def build_normalization(mode, total_events, run_info, luminosity):  # Decide how the histograms should be normalized.
    if total_events <= 0:  # There is nothing sensible to plot if the sample has no events.
        raise SystemExit("No events were found in tag_1_delphes_events.root.")  # Stop with a clear message.

    cross_section_pb = run_info["cross_section_pb"] if run_info else None  # Read the cross section from the banner information if available.

    if mode == "mc":  # Raw Monte Carlo mode keeps each event weight equal to 1.
        return {  # Return the settings needed for unweighted histograms.
            "event_weight": None,  # None means do not pass any explicit weights to matplotlib.
            "ylabel": "Monte Carlo Events",  # Label the y-axis as an event count.
            "mode_label": "Monte Carlo events",  # Short description used in printed summaries.
            "figure_title": "Selected-event kinematics",  # Title for the figure.
            "luminosity_fb": None,  # No luminosity value is needed in this mode.
        }  # End the raw-event normalization settings.

    if cross_section_pb is None:  # Weighted modes require the cross section from the banner file.
        raise SystemExit(  # Stop if that information is missing.
            "Could not find the run cross section in *_banner.txt. "  # Explain what could not be read.
            "Modes 2 and 3 require it."  # Explain why this is a problem.
        )  # End the error message.

    if mode == "xsec":  # Cross-section mode converts the histogram integral to pb.
        return {  # Return the settings for cross-section-weighted histograms.
            "event_weight": cross_section_pb / total_events,  # Each event gets an equal share of the total cross section.
            "ylabel": "Cross section [pb]",  # Label the y-axis in pb.
            "mode_label": "Cross-section weighted",  # Short description used in printed summaries.
            "figure_title": "Selected-event kinematics (cross-section weighted)",  # Title for the figure.
            "luminosity_fb": None,  # Luminosity is not part of this normalization.
        }  # End the cross-section normalization settings.

    return {  # If the mode is not mc or xsec, it must be luminosity scaling.
        "event_weight": cross_section_pb * luminosity * 1000.0 / total_events,  # Convert pb times fb^-1 into an expected event count per Monte Carlo event.
        "ylabel": f"Expected events (L = {luminosity:g} fb^-1)",  # Label the y-axis as an expected event yield.
        "mode_label": "Luminosity scaled",  # Short description used in printed summaries.
        "figure_title": f"Selected-event kinematics (L = {luminosity:g} fb^-1)",  # Title for the figure.
        "luminosity_fb": luminosity,  # Store the chosen luminosity for later reporting.
    }  # End the luminosity-scaling settings.


def build_summary_text(total_events, run_info, normalization, selection_lines):  # Build the text block printed inside the figure.
    run_info_lines = [  # Start with the general run information section.
        "Run information",  # Section title.
        f"Number of events: {total_events}",  # Total number of events in the ROOT tree.
        f"Normalization: {normalization['mode_label']}",  # State which histogram normalization was used.
    ]  # End the initial summary lines.

    if run_info is None:  # Handle the case where the banner file was missing.
        run_info_lines.append("Process: unavailable")  # The process string could not be read.
        run_info_lines.append("Cross section: unavailable")  # The cross section could not be read.
        run_info_lines.append("Pseudoscalar mass: unavailable")  # The pseudoscalar mass could not be read.
        run_info_lines.append("tanphi: unavailable")  # The tanphi parameter could not be read.
        run_info_lines.append("BR(a -> X): unavailable")  # The branching-ratio table could not be read.
    else:  # If the banner exists, format the extracted information nicely.
        process_text = run_info["process"] or "unavailable"  # Use the process string, or a fallback if it is missing.
        cross_section_text = (  # Format the cross section if available.
            f"{run_info['cross_section_pb']:.9g} pb"  # Show the value in pb with compact formatting.
            if run_info["cross_section_pb"] is not None  # Only do this when the value exists.
            else "unavailable"  # Otherwise show a fallback message.
        )  # End the cross-section formatting.
        mass_text = f"{run_info['mass']:g} GeV" if run_info["mass"] is not None else "unavailable"  # Format the pseudoscalar mass.
        tanphi_text = f"{run_info['tanphi']:g}" if run_info["tanphi"] is not None else "unavailable"  # Format tanphi.

        run_info_lines.append(f"Process: {process_text}")  # Add the process string to the figure summary.
        run_info_lines.append(f"Cross section: {cross_section_text}")  # Add the cross section to the figure summary.
        run_info_lines.append(f"Pseudoscalar mass: {mass_text}")  # Add the pseudoscalar mass to the figure summary.
        run_info_lines.append(f"tanphi: {tanphi_text}")  # Add tanphi to the figure summary.

        if normalization["luminosity_fb"] is not None:  # Only show luminosity if that mode uses it.
            run_info_lines.append(f"Luminosity: {normalization['luminosity_fb']:g} fb^-1")  # Add the chosen luminosity.

        run_info_lines.append("BR(a -> X):")  # Start the branching-ratio subsection.
        if run_info["branching_ratios"]:  # Check whether at least one decay channel was found.
            for branching_ratio, channel in run_info["branching_ratios"]:  # Loop over each decay channel.
                run_info_lines.append(f"  {channel}: {100 * branching_ratio:.3f}%")  # Add the channel and its branching fraction in percent.
        else:  # If no channels were read, say so explicitly.
            run_info_lines.append("  unavailable")  # Fallback text for missing branching ratios.

    return "\n".join(run_info_lines + ["", "Selection details"] + selection_lines)  # Combine everything into one multi-line string.


def main():  # Run the full analysis workflow from input reading to histogram saving.
    args = parse_arguments()  # Read the command-line settings chosen by the user.

    root_file = uproot.open("tag_1_delphes_events.root")  # Open the Delphes ROOT file.
    tree = root_file["Delphes;1"]  # Access the Delphes TTree inside the ROOT file.
    total_events = tree.num_entries  # Count the total number of events in the tree.
    run_info = read_run_info()  # Read process information and cross section from the banner file.
    normalization = build_normalization(args.mode, total_events, run_info, args.luminosity)  # Prepare the histogram normalization settings.

    branches = [  # List the ROOT branches needed for the event selection and plots.
        "Jet/Jet.BTag",  # Jet b-tag information.
        "Jet/Jet.PT",  # Jet transverse momentum.
        "Electron/Electron.Charge",  # Electron charge.
        "Electron/Electron.PT",  # Electron transverse momentum.
        "Muon/Muon.Charge",  # Muon charge.
        "Muon/Muon.PT",  # Muon transverse momentum.
        "MissingET/MissingET.MET",  # Missing transverse energy.
    ]  # End the branch list.
    arrays = tree.arrays(branches, library="ak")  # Read the selected branches as awkward arrays.

    jet_is_btag = arrays["Jet/Jet.BTag"] > 0  # Treat positive b-tag values as tagged b-jets.
    jet_pt = arrays["Jet/Jet.PT"]  # Store all jet transverse momenta.

    electron_charge = arrays["Electron/Electron.Charge"]  # Store electron charges event by event.
    electron_pt = arrays["Electron/Electron.PT"]  # Store electron transverse momenta event by event.
    muon_charge = arrays["Muon/Muon.Charge"]  # Store muon charges event by event.
    muon_pt = arrays["Muon/Muon.PT"]  # Store muon transverse momenta event by event.

    missing_et = ak.to_numpy(ak.firsts(arrays["MissingET/MissingET.MET"]))  # Take the first MET entry in each event and convert it to a NumPy array.

    all_lepton_charge = ak.concatenate([electron_charge, muon_charge], axis=1)  # Merge electrons and muons into one per-event charge list.
    all_lepton_pt = ak.concatenate([electron_pt, muon_pt], axis=1)  # Merge electrons and muons into one per-event pT list.

    n_electrons = ak.num(electron_charge, axis=1)  # Count reconstructed electrons in each event.
    n_muons = ak.num(muon_charge, axis=1)  # Count reconstructed muons in each event.
    n_leptons = ak.num(all_lepton_charge, axis=1)  # Count total reconstructed leptons in each event.

    padded_charge = ak.pad_none(all_lepton_charge, 2)  # Ensure each event has at least two charge slots, filling missing ones with None.
    charge_1 = ak.fill_none(padded_charge[:, 0], 0)  # Read the first lepton charge, using 0 if no lepton exists.
    charge_2 = ak.fill_none(padded_charge[:, 1], 0)  # Read the second lepton charge, using 0 if no lepton exists.

    exactly_two_btag_jets = ak.sum(jet_is_btag, axis=1) == 2  # Select events with exactly two b-tagged jets.
    exactly_two_leptons = n_leptons == 2  # Select events with exactly two reconstructed leptons.
    same_sign_leptons = exactly_two_leptons & (charge_1 * charge_2 > 0)  # Require the two leptons to have the same electric charge.
    selected_events = exactly_two_btag_jets & same_sign_leptons  # Final event selection is the overlap of both requirements.

    count_two_btag_jets, percent_two_btag_jets = count_and_fraction(  # Count events with exactly two b-tagged jets.
        exactly_two_btag_jets, total_events  # Pass the b-jet selection mask and the total number of events.
    )  # Finish computing the two-b-tag count and percentage.
    same_sign_count, same_sign_percent = count_and_fraction(same_sign_leptons, total_events)  # Count events with exactly two same-sign leptons.
    selected_count, selected_percent = count_and_fraction(selected_events, total_events)  # Count events passing the full selection.

    negative_pair = same_sign_leptons & (charge_1 < 0)  # Same-sign events where both leptons are negatively charged.
    positive_pair = same_sign_leptons & (charge_1 > 0)  # Same-sign events where both leptons are positively charged.

    ee_minus = negative_pair & (n_electrons == 2)  # Two-electron negative channel.
    emu_minus = negative_pair & (n_electrons == 1) & (n_muons == 1)  # Electron-muon negative channel.
    mumu_minus = negative_pair & (n_muons == 2)  # Two-muon negative channel.

    ee_plus = positive_pair & (n_electrons == 2)  # Two-electron positive channel.
    emu_plus = positive_pair & (n_electrons == 1) & (n_muons == 1)  # Electron-muon positive channel.
    mumu_plus = positive_pair & (n_muons == 2)  # Two-muon positive channel.

    selection_lines = [  # Build the text lines that summarize the selection yields.
        f"Total events: {total_events}",  # Total number of events before cuts.
        f"2 b-tagged jets: {count_two_btag_jets} ({percent_two_btag_jets:.2f}%)",  # Yield after the two-b-jet requirement.
        f"2 same-sign leptons: {same_sign_count} ({same_sign_percent:.2f}%)",  # Yield after the same-sign dilepton requirement.
        f"Both filters: {selected_count} ({selected_percent:.2f}%)",  # Yield after applying both requirements together.
        "",  # Blank line for readability.
        f"e- e-  : {int(ak.sum(ee_minus))}",  # Number of negative same-sign dielectron events.
        f"e- mu- : {int(ak.sum(emu_minus))}",  # Number of negative same-sign electron-muon events.
        f"mu- mu-: {int(ak.sum(mumu_minus))}",  # Number of negative same-sign dimuon events.
        f"e+ e+  : {int(ak.sum(ee_plus))}",  # Number of positive same-sign dielectron events.
        f"e+ mu+ : {int(ak.sum(emu_plus))}",  # Number of positive same-sign electron-muon events.
        f"mu+ mu+: {int(ak.sum(mumu_plus))}",  # Number of positive same-sign dimuon events.
    ]  # End the selection-summary list.

    print(  # Print a human-readable summary of the two-b-tag selection.
        f"From a total of {total_events} events, there are {count_two_btag_jets} "  # First part of the sentence with counts.
        "satisfying the applied filter: exactly two b-tagged jets at Delphes level."  # Explain the selection criterion.
    )  # End the first printed summary sentence.
    print(f"Percentage: {percent_two_btag_jets:.2f}%")  # Print the percentage for the two-b-tag selection.
    print()  # Add a blank line in the terminal output.

    print(  # Print a human-readable summary of the same-sign dilepton selection.
        f"From a total of {total_events} events, there are {same_sign_count} "  # First part of the sentence with counts.
        "satisfying the applied filter: exactly two same-charge reconstructed leptons "  # Explain the required lepton multiplicity and charge.
        "(electrons or muons) at Delphes level."  # Clarify which leptons are included.
    )  # End the second printed summary sentence.
    print(f"Percentage: {same_sign_percent:.2f}%")  # Print the percentage for the same-sign selection.
    print()  # Add a blank line in the terminal output.

    print("Breakdown by same-sign dilepton channel:")  # Introduce the channel-by-channel breakdown.
    print(f"e- e-  : {int(ak.sum(ee_minus))}")  # Print the negative dielectron count.
    print(f"e- mu- : {int(ak.sum(emu_minus))}")  # Print the negative electron-muon count.
    print(f"mu- mu-: {int(ak.sum(mumu_minus))}")  # Print the negative dimuon count.
    print(f"e+ e+  : {int(ak.sum(ee_plus))}")  # Print the positive dielectron count.
    print(f"e+ mu+ : {int(ak.sum(emu_plus))}")  # Print the positive electron-muon count.
    print(f"mu+ mu+: {int(ak.sum(mumu_plus))}")  # Print the positive dimuon count.
    print()  # Add a blank line in the terminal output.

    print(  # Print a summary of the final combined selection.
        f"From a total of {total_events} events, there are {selected_count} "  # First part of the sentence with counts.
        "satisfying both applied filters: exactly two b-tagged jets and exactly two "  # Explain that both requirements are imposed together.
        "same-charge reconstructed leptons at Delphes level."  # Finish describing the full event selection.
    )  # End the third printed summary sentence.
    print(f"Percentage: {selected_percent:.2f}%")  # Print the percentage for the final selection.

    selected_missing_et = missing_et[selected_events]  # Keep only the missing transverse energy values from selected events.

    selected_jet_pt = jet_pt[selected_events]  # Keep only jet pT values from selected events.
    selected_jet_is_btag = jet_is_btag[selected_events]  # Keep only jet b-tag masks from selected events.

    bjet_pt = ak.sort(selected_jet_pt[selected_jet_is_btag], axis=1, ascending=False)  # Collect selected b-jets and sort them by pT within each event.
    leading_bjet_pt = ak.to_numpy(bjet_pt[:, 0])  # Take the highest-pT b-jet from each selected event.
    subleading_bjet_pt = ak.to_numpy(bjet_pt[:, 1])  # Take the second-highest-pT b-jet from each selected event.

    selected_lepton_pt = ak.sort(all_lepton_pt[selected_events], axis=1, ascending=False)  # Sort the two selected leptons by pT within each event.
    leading_lepton_pt = ak.to_numpy(selected_lepton_pt[:, 0])  # Take the highest-pT lepton from each selected event.
    subleading_lepton_pt = ak.to_numpy(selected_lepton_pt[:, 1])  # Take the second-highest-pT lepton from each selected event.

    non_bjet_pt = ak.sort(selected_jet_pt[~selected_jet_is_btag], axis=1, ascending=False)  # Collect non-b jets in selected events and sort them by pT.
    has_non_bjet_1 = ak.num(non_bjet_pt, axis=1) >= 1  # Check which selected events contain at least one non-b jet.
    has_non_bjet_2 = ak.num(non_bjet_pt, axis=1) >= 2  # Check which selected events contain at least two non-b jets.
    leading_non_bjet_pt = ak.to_numpy(non_bjet_pt[has_non_bjet_1][:, 0])  # Take the leading non-b jet pT where such a jet exists.
    subleading_non_bjet_pt = ak.to_numpy(non_bjet_pt[has_non_bjet_2][:, 1])  # Take the subleading non-b jet pT where at least two such jets exist.

    selection_lines.extend(  # Add extra information about non-b jets to the figure summary.
        [  # Start the list of additional summary lines.
            "",  # Blank line for readability.
            f"Events with >=1 non-b jet: {int(ak.sum(has_non_bjet_1))}",  # Count selected events with at least one non-b jet.
            f"Events with >=2 non-b jets: {int(ak.sum(has_non_bjet_2))}",  # Count selected events with at least two non-b jets.
        ]  # End the added summary lines.
    )  # Finish extending the summary text.
    summary_text = build_summary_text(total_events, run_info, normalization, selection_lines)  # Build the full multi-line text block for the figure.

    fig, axes = plt.subplots(4, 2, figsize=(12, 18))  # Create a 4-by-2 grid of subplots.

    plot_items = [  # Define what to plot on each occupied panel.
        (axes[0, 0], leading_bjet_pt, "Leading b-jet pT [GeV]"),  # Top-left panel: leading b-jet pT.
        (axes[0, 1], subleading_bjet_pt, "Subleading b-jet pT [GeV]"),  # Top-right panel: subleading b-jet pT.
        (axes[1, 0], leading_lepton_pt, "Leading lepton pT [GeV]"),  # Second row left: leading lepton pT.
        (axes[1, 1], subleading_lepton_pt, "Subleading lepton pT [GeV]"),  # Second row right: subleading lepton pT.
        (axes[2, 0], leading_non_bjet_pt, "Leading non-b jet pT [GeV]"),  # Third row left: leading non-b jet pT.
        (axes[2, 1], subleading_non_bjet_pt, "Subleading non-b jet pT [GeV]"),  # Third row right: subleading non-b jet pT.
        (axes[3, 0], selected_missing_et, "Missing transverse momentum [GeV]"),  # Bottom-left panel: missing transverse momentum.
    ]  # End the list of histogram definitions.

    for ax, values, xlabel in plot_items:  # Loop over each histogram specification.
        make_hist(ax, values, xlabel, normalization["ylabel"], normalization["event_weight"])  # Draw the histogram with the chosen normalization.

    fig.suptitle(normalization["figure_title"], fontsize=14)  # Add a global title above all subplots.
    axes[3, 1].axis("off")  # Hide the unused bottom-right panel so it can hold text only.
    axes[3, 1].text(  # Write the run and selection summary into the empty panel.
        0.0,  # Place the text at the left edge of the panel.
        1.0,  # Place the text at the top edge of the panel.
        summary_text,  # Use the previously constructed multi-line summary string.
        ha="left",  # Left-align the text horizontally.
        va="top",  # Top-align the text vertically.
        fontsize=8,  # Use a small font so the full summary fits.
        linespacing=1.1,  # Slightly increase the spacing between lines.
        family="monospace",  # Use a monospace font so columns line up neatly.
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="black"),  # Draw a light box behind the text for readability.
    )  # Finish placing the summary text.

    fig.tight_layout(rect=(0, 0, 1, 0.97))  # Adjust subplot spacing while leaving room for the main title.
    fig.savefig(args.output, dpi=150)  # Save the figure to the requested output file.

    print()  # Add a blank line before the final normalization summary.
    print(f"Histogram normalization: {normalization['mode_label']}")  # Report which normalization mode was used.
    if run_info and run_info["cross_section_pb"] is not None and normalization["event_weight"] is not None:  # Only print the cross section when weighted modes are active.
        print(f"Cross section used for weighting: {run_info['cross_section_pb']:.9g} pb")  # Report the cross section used in the weights.
    if normalization["luminosity_fb"] is not None:  # Only print luminosity in luminosity-scaled mode.
        print(f"Luminosity: {normalization['luminosity_fb']:g} fb^-1")  # Report the luminosity used for scaling.
    if normalization["event_weight"] is not None:  # Only print a per-event weight for weighted histogram modes.
        print(f"Per-event histogram weight: {normalization['event_weight']:.9g}")  # Report the actual constant event weight.
    print()  # Add another blank line before the final status message.
    print(f"Saved histograms to {args.output}")  # Tell the user where the plot image was written.


if __name__ == "__main__":  # Run the script only when this file is executed directly.
    main()  # Start the analysis workflow.
