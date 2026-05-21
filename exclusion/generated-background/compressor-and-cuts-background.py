#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow.parquet as pq
import uproot
import vector


DELHES_TREE_CANDIDATES = ("Delphes", "Events")
BANNER_PATTERN = re.compile(r"^run_(\d+)_tag_(\d+)_banner\.txt$")
ROOT_PATTERN = re.compile(r"^tag_(\d+)_delphes_events\.root$")
DEFAULT_LABEL = "background"
DEFAULT_PROCESS_NUMBER = 1
DEFAULT_STEP_SIZE = "250 MB"
EVENT_BRANCHES = [
    "Event/Event.Number",
    "Electron/Electron.PT",
    "Electron/Electron.Eta",
    "Electron/Electron.Phi",
    "Electron/Electron.Charge",
    "Muon/Muon.PT",
    "Muon/Muon.Eta",
    "Muon/Muon.Phi",
    "Muon/Muon.Charge",
    "Jet/Jet.PT",
    "Jet/Jet.Eta",
    "Jet/Jet.Phi",
    "Jet/Jet.Mass",
    "Jet/Jet.BTag",
    "MissingET/MissingET.MET",
    "MissingET/MissingET.Phi",
]
PDG_LABELS = {
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
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Compress one generated background run into parquet and write the same "
            "analysis outputs used for the signal samples."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=script_dir / "run_01",
        help="Background run directory containing the banner and Delphes ROOT file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
        help="Directory where the parquet folder and CSV/TXT outputs will be written.",
    )
    parser.add_argument(
        "--label",
        default=DEFAULT_LABEL,
        help="Label used in the output filenames. Default: background",
    )
    parser.add_argument(
        "--process-number",
        type=int,
        default=DEFAULT_PROCESS_NUMBER,
        help="Process number stored in the output tables. Default: 1",
    )
    parser.add_argument(
        "--step-size",
        default=DEFAULT_STEP_SIZE,
        help=(
            "Chunk size passed to uproot.iterate. Use an integer entry count or a "
            "memory string such as '250 MB'. Default: 250 MB"
        ),
    )
    args = parser.parse_args()

    if args.process_number <= 0:
        parser.error("--process-number must be a positive integer.")

    if re.fullmatch(r"\d+", str(args.step_size).strip()):
        args.step_size = int(str(args.step_size).strip())

    return args


def format_value_for_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    sanitized = sanitized.strip("-")
    return sanitized or "value"


def build_parquet_dir(output_dir: Path) -> Path:
    return output_dir / "parquets"


def build_parquet_output_path(
    output_dir: Path, process_number: int, label: str
) -> Path:
    return build_parquet_dir(output_dir) / (
        f"proc-{process_number}-{format_value_for_filename(label)}.parquet"
    )


def build_cuts_output_path(output_dir: Path, label: str) -> Path:
    return output_dir / f"cuts-{format_value_for_filename(label)}.csv"


def build_efficiencies_output_path(output_dir: Path, label: str) -> Path:
    return output_dir / f"efficiencies-{format_value_for_filename(label)}.csv"


def build_xsec_output_path(output_dir: Path, label: str) -> Path:
    return output_dir / f"xsec-{format_value_for_filename(label)}.csv"


def build_decay_output_path(output_dir: Path, label: str) -> Path:
    return output_dir / f"decays-{format_value_for_filename(label)}.txt"


def build_compression_summary_path(output_dir: Path) -> Path:
    return build_parquet_dir(output_dir) / "compression-info.txt"


def discover_latest_banner(run_dir: Path) -> Path:
    latest_banner: tuple[int, int, Path] | None = None
    for banner_path in run_dir.glob("run_*_tag_*_banner.txt"):
        match = BANNER_PATTERN.fullmatch(banner_path.name)
        if match is None:
            continue

        candidate = (int(match.group(1)), int(match.group(2)), banner_path.resolve())
        if latest_banner is None or candidate[:2] > latest_banner[:2]:
            latest_banner = candidate

    if latest_banner is None:
        raise FileNotFoundError(f"No banner files found in {run_dir}")

    return latest_banner[2]


def discover_latest_delphes_root(run_dir: Path) -> Path:
    latest_root: tuple[int, Path] | None = None

    for root_path in run_dir.glob("tag_*_delphes_events.root"):
        match = ROOT_PATTERN.fullmatch(root_path.name)
        if match is None:
            continue

        candidate = (int(match.group(1)), root_path.resolve())
        if latest_root is None or candidate[0] > latest_root[0]:
            latest_root = candidate

    if latest_root is None:
        raise FileNotFoundError(f"No Delphes ROOT files found in {run_dir}")

    return latest_root[1]


def read_banner_text(banner_path: Path) -> str:
    return banner_path.read_text(encoding="utf-8", errors="replace")


def extract_generated_events_from_banner_text(banner_text: str) -> int:
    generated_events_match = re.search(
        r"#\s*Number of Events\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        banner_text,
    )
    if generated_events_match is not None:
        return int(float(generated_events_match.group(1)))

    run_card_match = re.search(
        r"^\s*([0-9]+(?:\.[0-9]+)?)\s*=\s*nevents\b",
        banner_text,
        flags=re.MULTILINE,
    )
    if run_card_match is None:
        raise ValueError("Could not find the generated event count in the banner file.")

    return int(float(run_card_match.group(1)))


def extract_init_process_rows(banner_text: str) -> list[tuple[float, float]]:
    init_match = re.search(
        r"<init>\s*\n(.*?)\n</init>",
        banner_text,
        flags=re.DOTALL,
    )
    if init_match is None:
        raise ValueError("Could not find the <init> block in the banner file.")

    lines = [line.strip() for line in init_match.group(1).splitlines() if line.strip()]
    if not lines:
        raise ValueError("The <init> block is empty.")

    header_parts = lines[0].split()
    nprup = None
    if header_parts:
        try:
            nprup = int(header_parts[-1])
        except ValueError:
            nprup = None

    rows: list[tuple[float, float]] = []
    for line in lines[1:]:
        if line.startswith("<"):
            break

        parts = line.split()
        if len(parts) < 2:
            continue

        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue

        if nprup is not None and len(rows) >= nprup:
            break

    if not rows:
        raise ValueError("Could not parse subprocess cross sections from the <init> block.")

    return rows


def extract_total_cross_section_from_banner_text(banner_text: str) -> float:
    integrated_weight_match = re.search(
        r"Integrated weight \(pb\)\s*:\s*([0-9.eE+-]+)",
        banner_text,
    )
    if integrated_weight_match is not None:
        return float(integrated_weight_match.group(1))

    return sum(cross_section for cross_section, _ in extract_init_process_rows(banner_text))


def extract_total_cross_section_error_from_banner_text(banner_text: str) -> float:
    rows = extract_init_process_rows(banner_text)
    return math.sqrt(sum(error * error for _, error in rows))


def extract_mass_from_banner_text(banner_text: str, mass_pdg: int = 36) -> str | None:
    mass_block_match = re.search(
        r"BLOCK MASS\b(.*?)(?:^\s*BLOCK\b|^\s*DECAY\b|^\s*</MGParamCard>)",
        banner_text,
        flags=re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    if mass_block_match is None:
        return None

    mass_match = re.search(
        rf"^\s*{mass_pdg}\s+([0-9.eE+-]+)\b",
        mass_block_match.group(1),
        flags=re.MULTILINE,
    )
    return mass_match.group(1) if mass_match is not None else None


def extract_tanphi_from_banner_text(banner_text: str) -> str | None:
    tanphi_match = re.search(
        r"^\s*1\s+([0-9.eE+-]+)\s+#\s*tanphi\b",
        banner_text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return tanphi_match.group(1) if tanphi_match is not None else None


def extract_pseudoscalar_decay_data(
    banner_text: str, pseudoscalar_pdg: int = 36
) -> tuple[str, dict[str, str]] | None:
    decay_match = re.search(
        rf"^\s*DECAY\s+{pseudoscalar_pdg}\s+([0-9.eE+-]+)\b.*?$"
        r"(.*?)(?=^\s*DECAY\b|^\s*</slha>|\Z)",
        banner_text,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if decay_match is None:
        return None

    width = decay_match.group(1)
    decay_block = decay_match.group(2)
    branching_ratios: dict[str, str] = {}
    for line in decay_block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split("#", 1)[0].split()
        if len(parts) < 4:
            continue

        br = parts[0]
        nda = int(parts[1])
        daughters = parts[2 : 2 + nda]
        if len(daughters) != nda:
            continue

        column_name = f"branching ratio ({' '.join(daughters)})"
        branching_ratios[column_name] = br

    return width, branching_ratios


def format_decay_channel(daughters: str) -> str:
    labels = [PDG_LABELS.get(int(pid), pid) for pid in daughters.split()]
    return " ".join(labels)


def extract_process_descriptions_from_banner_text(banner_text: str) -> list[str]:
    descriptions: list[str] = []
    lines = banner_text.splitlines()
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        prefix = None
        if stripped.startswith("generate "):
            prefix = "generate "
        elif stripped.startswith("add process "):
            prefix = "add process "

        if prefix is None:
            index += 1
            continue

        description = stripped[len(prefix) :].rstrip()
        while description.endswith("\\") and index + 1 < len(lines):
            description = description[:-1] + lines[index + 1].strip()
            index += 1

        descriptions.append(" ".join(description.split()))
        index += 1

    return descriptions


def build_decay_summary_text(banner_text: str, label: str) -> str:
    mass = extract_mass_from_banner_text(banner_text)
    tanphi = extract_tanphi_from_banner_text(banner_text)
    decay_data = extract_pseudoscalar_decay_data(banner_text)

    if decay_data is not None and mass is not None and tanphi is not None:
        width, branching_ratios = decay_data
        lines = [
            f"mass: {mass}",
            f"tanphi: {tanphi}",
            f"total width (GeV): {width}",
            "",
            "BR(a -> X):",
        ]
        for column_name, br in branching_ratios.items():
            daughters = column_name.removeprefix("branching ratio (").removesuffix(")")
            lines.append(f"{format_decay_channel(daughters)}: {float(br) * 100:.3f}%")
        return "\n".join(lines) + "\n"

    lines = [
        f"sample: {label}",
        "mass: n/a",
        "tanphi: n/a",
        "total width (GeV): n/a",
        "",
        "Processes:",
    ]
    process_descriptions = extract_process_descriptions_from_banner_text(banner_text)
    if process_descriptions:
        lines.extend(process_descriptions)
    else:
        lines.append("No process description found in the banner.")

    lines.extend(
        [
            "",
            "BR(a -> X):",
            "not applicable for this background sample",
        ]
    )
    return "\n".join(lines) + "\n"


def build_lepton_array(
    pt: ak.Array,
    eta: ak.Array,
    phi: ak.Array,
    charge: ak.Array,
    mass: float,
) -> ak.Array:
    masses = ak.ones_like(pt, dtype=np.float64) * mass
    return ak.zip(
        {
            "pt": pt,
            "eta": eta,
            "phi": phi,
            "mass": masses,
            "charge": charge,
        },
        with_name="Momentum4D",
    )


def reorder_by_field(array: ak.Array, field: str) -> ak.Array:
    return array[ak.argsort(array[field], axis=1, ascending=False)]


def singleton_event_field_to_numpy(array: ak.Array) -> np.ndarray:
    values = ak.fill_none(ak.firsts(ak.pad_none(array, 1, axis=1, clip=True)), np.nan)
    return ak.to_numpy(values)


def build_event_dataset_from_arrays(arrays: ak.Array) -> ak.Array:
    electron_mass = ak.ones_like(arrays["Electron/Electron.PT"], dtype=np.float64) * 0.000511
    muon_mass = ak.ones_like(arrays["Muon/Muon.PT"], dtype=np.float64) * 0.105658

    electrons = ak.zip(
        {
            "PT": arrays["Electron/Electron.PT"],
            "Eta": arrays["Electron/Electron.Eta"],
            "Phi": arrays["Electron/Electron.Phi"],
            "Mass": electron_mass,
            "Charge": arrays["Electron/Electron.Charge"],
        }
    )
    muons = ak.zip(
        {
            "PT": arrays["Muon/Muon.PT"],
            "Eta": arrays["Muon/Muon.Eta"],
            "Phi": arrays["Muon/Muon.Phi"],
            "Mass": muon_mass,
            "Charge": arrays["Muon/Muon.Charge"],
        }
    )
    jets = ak.zip(
        {
            "PT": arrays["Jet/Jet.PT"],
            "Eta": arrays["Jet/Jet.Eta"],
            "Phi": arrays["Jet/Jet.Phi"],
            "Mass": arrays["Jet/Jet.Mass"],
            "BTag": arrays["Jet/Jet.BTag"],
        }
    )
    jets = reorder_by_field(jets, "PT")

    electron_vectors = build_lepton_array(
        arrays["Electron/Electron.PT"],
        arrays["Electron/Electron.Eta"],
        arrays["Electron/Electron.Phi"],
        arrays["Electron/Electron.Charge"],
        0.000511,
    )
    muon_vectors = build_lepton_array(
        arrays["Muon/Muon.PT"],
        arrays["Muon/Muon.Eta"],
        arrays["Muon/Muon.Phi"],
        arrays["Muon/Muon.Charge"],
        0.105658,
    )
    leptons = ak.concatenate([electron_vectors, muon_vectors], axis=1)
    leptons = reorder_by_field(leptons, "pt")
    bjets = jets[jets.BTag != 0]

    return ak.Array(
        {
            "event_id": singleton_event_field_to_numpy(arrays["Event/Event.Number"]),
            "met": singleton_event_field_to_numpy(arrays["MissingET/MissingET.MET"]),
            "met_phi": singleton_event_field_to_numpy(arrays["MissingET/MissingET.Phi"]),
            "n_electrons": ak.to_numpy(ak.num(electrons, axis=1)),
            "n_muons": ak.to_numpy(ak.num(muons, axis=1)),
            "n_leptons": ak.to_numpy(ak.num(leptons, axis=1)),
            "n_jets": ak.to_numpy(ak.num(jets, axis=1)),
            "n_bjets": ak.to_numpy(ak.num(bjets, axis=1)),
            "Electron": electrons,
            "Muon": muons,
            "Jet": jets,
            "Lepton": ak.zip(
                {
                    "PT": leptons.pt,
                    "Eta": leptons.eta,
                    "Phi": leptons.phi,
                    "Mass": leptons.mass,
                    "Charge": leptons.charge,
                }
            ),
        }
    )


def select_delphes_tree(root_file: uproot.ReadOnlyDirectory, root_path: Path):
    for tree_name in DELHES_TREE_CANDIDATES:
        if tree_name in root_file:
            return root_file[tree_name]
    raise KeyError(f"No Delphes/Events tree found in {root_path}")


def build_lepton_vectors(leptons: ak.Array) -> ak.Array:
    return ak.zip(
        {
            "pt": leptons.PT,
            "eta": leptons.Eta,
            "phi": leptons.Phi,
            "mass": leptons.Mass,
            "charge": leptons.Charge,
        },
        with_name="Momentum4D",
    )


def count_events(mask: ak.Array) -> int:
    return int(ak.sum(mask))


def build_cut_masks(events: ak.Array) -> dict[str, ak.Array]:
    jets = events.Jet
    bjets_passing_cut_i = (jets.BTag > 0) & (jets.PT > 20.0) & (abs(jets.Eta) < 2.5)
    cut_i = ak.any(bjets_passing_cut_i, axis=1)

    cut_ii = cut_i & (events.met > 20.0)

    leptons = build_lepton_vectors(events.Lepton)
    leptons_passing_kinematics = leptons[(leptons.pt > 10.0) & (abs(leptons.eta) < 2.5)]
    exactly_two_selected_leptons = ak.num(leptons_passing_kinematics, axis=1) == 2

    padded_leptons = ak.pad_none(leptons_passing_kinematics, 2, axis=1, clip=True)
    first_lepton = padded_leptons[:, 0]
    second_lepton = padded_leptons[:, 1]
    same_sign_leptons = (
        exactly_two_selected_leptons
        & ak.fill_none(first_lepton.charge * second_lepton.charge > 0, False)
    )
    cut_iii = cut_ii & same_sign_leptons

    dilepton_mass = ak.fill_none((first_lepton + second_lepton).mass, 0.0)
    cut_iv = cut_iii & (dilepton_mass > 100.0)

    return {
        "cut_i": cut_i,
        "cut_ii": cut_ii,
        "cut_iii": cut_iii,
        "cut_iv": cut_iv,
    }


def summarize_cutflow_counts(events: ak.Array) -> dict[str, int]:
    total_events = len(events)
    cut_masks = build_cut_masks(events)
    return {
        "total_events": total_events,
        "pass_cut_i_bjet": count_events(cut_masks["cut_i"]),
        "pass_cut_ii_met": count_events(cut_masks["cut_ii"]),
        "pass_cut_iii_same_sign_dilepton": count_events(cut_masks["cut_iii"]),
        "pass_cut_iv_mll": count_events(cut_masks["cut_iv"]),
    }


def initialize_cutflow_totals() -> dict[str, int]:
    return {
        "total_events": 0,
        "pass_cut_i_bjet": 0,
        "pass_cut_ii_met": 0,
        "pass_cut_iii_same_sign_dilepton": 0,
        "pass_cut_iv_mll": 0,
    }


def build_cutflow_row(process_number: int, counts: dict[str, int]) -> dict[str, int]:
    return {"proc": process_number, **counts}


def stream_events_to_parquet_and_collect_cutflow(
    root_path: Path,
    output_path: Path,
    process_number: int,
    step_size: int | str,
) -> tuple[Path, dict[str, int]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    cutflow_totals = initialize_cutflow_totals()
    chunk_index = 0

    try:
        with uproot.open(root_path) as root_file:
            tree = select_delphes_tree(root_file, root_path)
            for arrays in tree.iterate(EVENT_BRANCHES, library="ak", step_size=step_size):
                chunk_index += 1
                dataset_chunk = build_event_dataset_from_arrays(arrays)
                arrow_table = ak.to_arrow_table(dataset_chunk, extensionarray=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        output_path,
                        arrow_table.schema,
                        compression="zstd",
                    )
                writer.write_table(arrow_table)

                chunk_counts = summarize_cutflow_counts(dataset_chunk)
                for key, value in chunk_counts.items():
                    cutflow_totals[key] += value

                print(
                    f"Processed chunk {chunk_index}: {chunk_counts['total_events']} events "
                    f"({cutflow_totals['total_events']} total)"
                )
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise ValueError(f"No events were read from {root_path}.")

    return output_path.resolve(), build_cutflow_row(process_number, cutflow_totals)


def write_cutflow_csv(output_path: Path, rows: list[dict[str, int]]) -> Path:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "proc",
                "total_events",
                "pass_cut_i_bjet",
                "pass_cut_ii_met",
                "pass_cut_iii_same_sign_dilepton",
                "pass_cut_iv_mll",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path.resolve()


def build_efficiency_rows_from_cutflow(
    cutflow_row: dict[str, int],
    cross_section_pb: float,
    generated_events: int,
) -> list[dict[str, str]]:
    if generated_events <= 0:
        raise ValueError("Generated event count must be positive.")

    event_weight_pb = cross_section_pb / generated_events
    luminosity_450_fb = 450.0
    luminosity_3000_fb = 3000.0

    weighted_total = generated_events * event_weight_pb
    weighted_cut_i = cutflow_row["pass_cut_i_bjet"] * event_weight_pb
    weighted_cut_ii = cutflow_row["pass_cut_ii_met"] * event_weight_pb
    weighted_cut_iii = cutflow_row["pass_cut_iii_same_sign_dilepton"] * event_weight_pb
    weighted_cut_iv = cutflow_row["pass_cut_iv_mll"] * event_weight_pb

    if weighted_total <= 0.0:
        raise ValueError("The total weighted yield is zero; efficiencies are undefined.")

    def build_efficiency_row(cut_label: str, cut_cross_section_pb: float) -> dict[str, str]:
        return {
            "cut": cut_label,
            "efficiency": f"{cut_cross_section_pb / weighted_total:.6f}",
            "cross_section_pb": f"{cut_cross_section_pb:.6f}",
            "yield_450_fb": f"{cut_cross_section_pb * luminosity_450_fb * 1000.0:.6f}",
            "yield_3000_fb": f"{cut_cross_section_pb * luminosity_3000_fb * 1000.0:.6f}",
        }

    return [
        build_efficiency_row("I", weighted_cut_i),
        build_efficiency_row("II", weighted_cut_ii),
        build_efficiency_row("III", weighted_cut_iii),
        build_efficiency_row("IV", weighted_cut_iv),
    ]


def write_efficiencies_csv(output_path: Path, rows: list[dict[str, str]]) -> Path:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cut",
                "efficiency",
                "cross_section_pb",
                "yield_450_fb",
                "yield_3000_fb",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path.resolve()


def build_xsec_row(
    banner_text: str, process_number: int, label: str
) -> dict[str, str]:
    process_descriptions = extract_process_descriptions_from_banner_text(banner_text)
    mass = extract_mass_from_banner_text(banner_text) or ""
    tanphi = extract_tanphi_from_banner_text(banner_text) or ""

    return {
        "process": str(process_number),
        "mass": mass,
        "tanphi": tanphi,
        "cross section pb": f"{extract_total_cross_section_from_banner_text(banner_text):.6e}",
        "cross section error pb": (
            f"{extract_total_cross_section_error_from_banner_text(banner_text):.6e}"
        ),
        "N events (generated)": str(extract_generated_events_from_banner_text(banner_text)),
        "description": " ; ".join(process_descriptions) if process_descriptions else label,
    }


def write_xsec_csv(output_path: Path, rows: list[dict[str, str]]) -> Path:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "process",
                "mass",
                "tanphi",
                "cross section pb",
                "cross section error pb",
                "N events (generated)",
                "description",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path.resolve()


def write_decay_summary(output_path: Path, banner_text: str, label: str) -> Path:
    output_path.write_text(build_decay_summary_text(banner_text, label), encoding="utf-8")
    return output_path.resolve()


def format_file_size(num_bytes: int) -> str:
    return f"{num_bytes / 1024**3:.6f} GB"


def build_compression_summary_lines(
    label: str,
    root_path: Path,
    parquet_path: Path,
) -> list[str]:
    root_size = root_path.stat().st_size
    parquet_size = parquet_path.stat().st_size
    return [
        f"Sample: {label}",
        "Wrote 1 compressed event Parquet file.",
        "ROOT vs Parquet sizes:",
        f"{parquet_path.stem}: {format_file_size(root_size)} -> {format_file_size(parquet_size)}",
        f"total: {format_file_size(root_size)} -> {format_file_size(parquet_size)}",
    ]


def write_compression_summary(
    output_path: Path,
    label: str,
    root_path: Path,
    parquet_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(build_compression_summary_lines(label, root_path, parquet_path)) + "\n",
        encoding="utf-8",
    )
    return output_path.resolve()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    label = args.label

    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    vector.register_awkward()
    banner_path = discover_latest_banner(run_dir)
    root_path = discover_latest_delphes_root(run_dir)
    print(f"Using banner: {banner_path}")
    print(f"Using Delphes ROOT: {root_path}")
    banner_text = read_banner_text(banner_path)

    print(f"Streaming ROOT into parquet with step size {args.step_size!r}...")
    parquet_output_path, cutflow_row = stream_events_to_parquet_and_collect_cutflow(
        root_path,
        build_parquet_output_path(output_dir, args.process_number, label),
        args.process_number,
        args.step_size,
    )
    compression_summary_path = write_compression_summary(
        build_compression_summary_path(output_dir),
        label,
        root_path,
        parquet_output_path,
    )

    cuts_output_path = write_cutflow_csv(
        build_cuts_output_path(output_dir, label),
        [cutflow_row],
    )

    xsec_row = build_xsec_row(banner_text, args.process_number, label)
    xsec_output_path = write_xsec_csv(
        build_xsec_output_path(output_dir, label),
        [xsec_row],
    )

    efficiencies_output_path = write_efficiencies_csv(
        build_efficiencies_output_path(output_dir, label),
        build_efficiency_rows_from_cutflow(
            cutflow_row,
            float(xsec_row["cross section pb"]),
            int(xsec_row["N events (generated)"]),
        ),
    )

    decay_output_path = write_decay_summary(
        build_decay_output_path(output_dir, label),
        banner_text,
        label,
    )

    print(f"Wrote {parquet_output_path}")
    print(f"Wrote {compression_summary_path}")
    print(f"Wrote {cuts_output_path}")
    print(f"Wrote {efficiencies_output_path}")
    print(f"Wrote {xsec_output_path}")
    print(f"Wrote {decay_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
