#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math     
import os
import re
import subprocess
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
import vector

from generated_signal_paths import build_generated_dir, format_value_for_filename


DEFAULT_MG5_BIN = Path(
    os.environ.get("MG5_BIN_DIR", "/home/patricio/Documents/mg5amcnlo-3.x/bin")
)
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("tmp_run.mg5")
DELHES_TREE_CANDIDATES = ("Delphes", "Events")
PROCESS_DIR_PATTERN = re.compile(r"^proc-(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write a MadGraph .mg5 command file that launches every generated "
            "proc-* directory in the local MG5 bin folder with the same number "
            "of events."
        )
    )
    parser.add_argument(
        "nevents",
        type=int,
        help="Number of events to generate in each discovered process directory.",
    )
    parser.add_argument(
        "--mass-pdg",
        type=int,
        default=36,
        help="PDG code whose mass entry will be updated. Default: 36",
    )
    parser.add_argument(
        "--mass",
        default="200",
        help="Mass value written through 'set mass <pdg> <value>'. Default: 200",
    )
    parser.add_argument(
        "--mass-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Optional linear mass scan as: --mass-range MIN MAX",
    )
    parser.add_argument(
        "--mass-points",
        type=int,
        default=3,
        help="Number of mass points when --mass-range is used. Default: 3",
    )
    parser.add_argument(
        "--tanphi-index",
        type=int,
        default=1,
        help="Index used in the TANPHI block. Default: 1",
    )
    parser.add_argument(
        "--tanphi",
        default="60",
        help="TANPHI value written through 'set tanphi <index> <value>'. Default: 60",
    )
    parser.add_argument(
        "--tanphi-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Optional logarithmic tanphi scan as: --tanphi-range MIN MAX",
    )
    parser.add_argument(
        "--tanphi-points",
        type=int,
        default=3,
        help="Number of tanphi points when --tanphi-range is used. Default: 3",
    )
    parser.add_argument(
        "--ebeam1",
        default="7000",
        help="Beam-1 energy in GeV. Default: 7000",
    )
    parser.add_argument(
        "--ebeam2",
        default="7000",
        help="Beam-2 energy in GeV. Default: 7000",
    )
    parser.add_argument(
        "--mg5-bin",
        type=Path,
        default=DEFAULT_MG5_BIN,
        help="Path to the MadGraph bin directory that contains proc-* outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination .mg5 file. Default: exclusion/tmp_run.mg5",
    )
    parser.add_argument(
        "--write-only",
        action="store_true",
        help="Only write the .mg5 card without launching MadGraph.",
    )
    parser.add_argument(
        "--xsec-output",
        type=Path,
        default=None,
        help=(
            "Destination CSV file for cross section summaries. "
            "Default: exclusion/generated-signal/generated-m-<mass>-tanphi-<tanphi>/"
            "xsec-m-<mass>-tanphi-<tanphi>.csv"
        ),
    )
    args = parser.parse_args()

    if args.nevents <= 0:
        parser.error("nevents must be a positive integer.")
    if args.mass_points <= 0:
        parser.error("--mass-points must be a positive integer.")
    if args.tanphi_points <= 0:
        parser.error("--tanphi-points must be a positive integer.")
    if args.mass_range is not None:
        args.mass_min, args.mass_max = args.mass_range
        if args.mass_min <= 0 or args.mass_max <= 0:
            parser.error("--mass-range values must be positive.")
        if args.mass_min > args.mass_max:
            parser.error("--mass-range requires MIN <= MAX.")
    if args.tanphi_range is not None:
        args.tanphi_min, args.tanphi_max = args.tanphi_range
        if args.tanphi_min <= 0 or args.tanphi_max <= 0:
            parser.error("--tanphi-range values must be positive.")
        if args.tanphi_min > args.tanphi_max:
            parser.error("--tanphi-range requires MIN <= MAX.")
    if args.output.suffix != ".mg5":
        parser.error("--output must end with .mg5.")
    if args.xsec_output is not None and (
        args.mass_range is not None or args.tanphi_range is not None
    ):
        parser.error("--xsec-output cannot be used together with scan ranges.")
    if args.xsec_output is None and args.mass_range is None and args.tanphi_range is None:
        args.xsec_output = build_default_xsec_output(args.mass, args.tanphi)
    if args.xsec_output is not None and args.xsec_output.suffix != ".csv":
        parser.error("--xsec-output must end with .csv.")

    return args

def build_default_xsec_output(mass: str, tanphi: str) -> Path:
    filename = (
        f"xsec-m-{format_value_for_filename(mass)}"
        f"-tanphi-{format_value_for_filename(tanphi)}.csv"
    )
    return build_default_generated_dir(mass, tanphi) / filename


def format_scan_value(value: float) -> str:
    return f"{value:.12g}"


def linspace(start: float, stop: float, count: int) -> list[float]:
    if count == 1:
        return [start]
    step = (stop - start) / (count - 1)
    return [start + index * step for index in range(count)]


def logspace(start: float, stop: float, count: int) -> list[float]:
    if count == 1:
        return [start]
    log_start = math.log10(start)
    log_stop = math.log10(stop)
    return [10 ** value for value in linspace(log_start, log_stop, count)]


def build_default_decays_output(mass: str, tanphi: str) -> Path:
    filename = (
        f"decays-m-{format_value_for_filename(mass)}"
        f"-tanphi-{format_value_for_filename(tanphi)}.txt"
    )
    return build_default_generated_dir(mass, tanphi) / filename


def build_default_generated_dir(mass: str, tanphi: str) -> Path:
    return build_generated_dir(Path(__file__).resolve().parent, mass, tanphi)


def build_default_compressed_events_dir(mass: str, tanphi: str) -> Path:
    return build_default_generated_dir(mass, tanphi) / "parquets"


def build_scan_values(
    point_value: str,
    range_values: tuple[float, float] | None,
    point_count: int,
    spacing: str,
) -> list[str]:
    if range_values is None:
        return [point_value]

    start, stop = range_values
    values = (
        linspace(start, stop, point_count)
        if spacing == "linear"
        else logspace(start, stop, point_count)
    )
    return [format_scan_value(value) for value in values]


def build_compressed_event_output(
    process_dir: Path,
    mass: str,
    tanphi: str,
    output_dir: Path,
) -> Path:
    return output_dir / (
        f"{process_dir.name}-m-{format_value_for_filename(mass)}"
        f"-tanphi-{format_value_for_filename(tanphi)}.parquet"
    )


def discover_process_directories(mg5_bin: Path) -> list[Path]:
    if not mg5_bin.exists():
        raise FileNotFoundError(f"MadGraph bin directory not found: {mg5_bin}")
    if not mg5_bin.is_dir():
        raise NotADirectoryError(f"MadGraph bin path is not a directory: {mg5_bin}")

    process_entries: list[tuple[int, str, Path]] = []
    for path in mg5_bin.iterdir():
        match = PROCESS_DIR_PATTERN.fullmatch(path.name)
        if match is None or not path.is_dir():
            continue
        if not (path / "Cards" / "proc_card_mg5.dat").is_file():
            continue

        process_entries.append((int(match.group(1)), path.name, path.resolve()))

    process_dirs = [
        path for _, _, path in sorted(process_entries, key=lambda item: (item[0], item[1]))
    ]
    if not process_dirs:
        raise FileNotFoundError(
            f"No generated proc-* directories were found inside {mg5_bin}."
        )

    return process_dirs


def build_mg5_card(
    process_dirs: list[Path],
    nevents: int,
    mass_pdg: int,
    mass: str,
    tanphi_index: int,
    tanphi: str,
    ebeam1: str,
    ebeam2: str,
) -> str:
    lines = [
        "set automatic_html_opening False --no_save",
        "set notification_center False --no_save",
        "",
    ]

    for process_dir in process_dirs:
        lines.extend(
            [
                f"launch {process_dir.name}",
                "shower=Pythia8",
                "detector=Delphes",
                f"set nevents {nevents}",
                f"set mass {mass_pdg} {mass}",
                "set width 36 Auto",
                f"set tanphi {tanphi_index} {tanphi}",
                f"set ebeam1 {ebeam1}",
                f"set ebeam2 {ebeam2}",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def write_card(output_path: Path, card_text: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(card_text, encoding="utf-8")
    return output_path.resolve()


def run_madgraph(mg5_bin: Path, card_path: Path) -> int:
    mg5_executable = mg5_bin / "mg5_aMC"
    if not mg5_executable.is_file():
        raise FileNotFoundError(f"MadGraph executable not found: {mg5_executable}")
    if not os.access(mg5_executable, os.X_OK):
        raise PermissionError(f"MadGraph executable is not executable: {mg5_executable}")

    completed = subprocess.run(
        [str(mg5_executable.resolve()), str(card_path.resolve())],
        cwd=str(mg5_bin),
        check=False,
    )
    return completed.returncode


def extract_process_description(process_dir: Path) -> str:
    proc_card_path = process_dir / "Cards" / "proc_card_mg5.dat"
    lines = proc_card_path.read_text(encoding="utf-8", errors="replace").splitlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("generate "):
            i += 1
            continue

        description = line[len("generate ") :].rstrip()
        while description.endswith("\\") and i + 1 < len(lines):
            description = description[:-1] + lines[i + 1].strip()
            i += 1
        return " ".join(description.split())

    raise ValueError(f"No 'generate' command found in {proc_card_path}")


def extract_requested_events_from_banner(banner_path: Path) -> str:
    content = banner_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"^\s*([0-9]+(?:\.[0-9]+)?)\s*=\s*nevents\b",
        content,
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"Could not find requested event count in {banner_path}")

    nevents = match.group(1)
    return nevents[:-2] if nevents.endswith(".0") else nevents


def extract_cross_section_from_banner(banner_path: Path) -> tuple[str, str, str]:
    content = banner_path.read_text(encoding="utf-8", errors="replace")

    generated_events_match = re.search(
        r"#\s*Number of Events\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        content,
    )
    if generated_events_match is not None:
        generated_events = generated_events_match.group(1)
        if generated_events.endswith(".0"):
            generated_events = generated_events[:-2]
    else:
        generated_events = extract_requested_events_from_banner(banner_path)

    init_match = re.search(
        r"<init>\s*\n.*\n\s*([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+[0-9.eE+-]+\s+\d+",
        content,
        flags=re.DOTALL,
    )
    if init_match is None:
        raise ValueError(f"Could not find cross section data in {banner_path}")

    return init_match.group(1), init_match.group(2), generated_events


def extract_mass_from_banner(banner_path: Path, mass_pdg: int = 36) -> str:
    content = banner_path.read_text(encoding="utf-8", errors="replace")
    mass_block_match = re.search(
        r"BLOCK MASS\b(.*?)(?:^\s*BLOCK\b|^\s*DECAY\b|^\s*</MGParamCard>)",
        content,
        flags=re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    if mass_block_match is None:
        raise ValueError(f"Could not find MASS block in {banner_path}")

    mass_match = re.search(
        rf"^\s*{mass_pdg}\s+([0-9.eE+-]+)\b",
        mass_block_match.group(1),
        flags=re.MULTILINE,
    )
    if mass_match is None:
        raise ValueError(f"Could not find mass entry for PDG {mass_pdg} in {banner_path}")

    return mass_match.group(1)


def extract_tanphi_from_banner(banner_path: Path) -> str:
    content = banner_path.read_text(encoding="utf-8", errors="replace")
    tanphi_match = re.search(
        r"^\s*1\s+([0-9.eE+-]+)\s+#\s*tanphi\b",
        content,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if tanphi_match is None:
        raise ValueError(f"Could not find tanphi entry in {banner_path}")

    return tanphi_match.group(1)


def extract_pseudoscalar_decay_data(
    banner_path: Path, pseudoscalar_pdg: int = 36
) -> tuple[str, dict[str, str]]:
    content = banner_path.read_text(encoding="utf-8", errors="replace")
    decay_match = re.search(
        rf"^\s*DECAY\s+{pseudoscalar_pdg}\s+([0-9.eE+-]+)\b.*?$"
        r"(.*?)(?=^\s*DECAY\b|^\s*</slha>|\Z)",
        content,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if decay_match is None:
        raise ValueError(
            f"Could not find DECAY block for pseudoscalar PDG {pseudoscalar_pdg} in {banner_path}"
        )

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


PDG_LABELS = {
    -24: "W-",
    -16: "vt~",
    -15: "tau+",
    -14: "vm~",
    -13: "mu+",
    -12: "ve~",
    -11: "e+",
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
    11: "e-",
    12: "ve",
    13: "mu-",
    14: "vm",
    15: "tau-",
    16: "vt",
    21: "g",
    22: "gamma",
    23: "Z",
    24: "W+",
    25: "H",
    36: "AP",
}


def format_decay_channel(daughters: str) -> str:
    labels = [PDG_LABELS.get(int(pid), pid) for pid in daughters.split()]
    return " ".join(labels)


def build_decay_summary_text(banner_path: Path) -> str:
    mass = extract_mass_from_banner(banner_path)
    tanphi = extract_tanphi_from_banner(banner_path)
    width, branching_ratios = extract_pseudoscalar_decay_data(banner_path)

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


def write_decay_summary(output_path: Path, banner_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_decay_summary_text(banner_path), encoding="utf-8")
    return output_path.resolve()


def extract_latest_existing_banner(process_dir: Path) -> tuple[int, int, Path]:
    latest_banner: tuple[int, int, Path] | None = None
    for banner_path in process_dir.glob("Events/run_*/run_*_tag_*_banner.txt"):
        match = re.fullmatch(
            r"run_(\d+)_tag_(\d+)_banner\.txt",
            banner_path.name,
        )
        if match is None:
            continue

        candidate = (int(match.group(1)), int(match.group(2)), banner_path)
        if latest_banner is None or candidate[:2] > latest_banner[:2]:
            latest_banner = candidate

    if latest_banner is None:
        raise FileNotFoundError(f"No banner files found in {process_dir / 'Events'}")

    return latest_banner


def extract_latest_delphes_root(process_dir: Path) -> tuple[int, int, Path]:
    latest_root: tuple[int, int, Path] | None = None
    for root_path in process_dir.glob("Events/run_*/tag_*_delphes_events.root"):
        run_match = re.fullmatch(r"run_(\d+)", root_path.parent.name)
        tag_match = re.fullmatch(r"tag_(\d+)_delphes_events\.root", root_path.name)
        if run_match is None or tag_match is None:
            continue

        candidate = (int(run_match.group(1)), int(tag_match.group(1)), root_path)
        if latest_root is None or candidate[:2] > latest_root[:2]:
            latest_root = candidate

    if latest_root is None:
        raise FileNotFoundError(f"No Delphes ROOT files found in {process_dir / 'Events'}")

    return latest_root


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


def build_jet_array(
    pt: ak.Array,
    eta: ak.Array,
    phi: ak.Array,
    mass: ak.Array,
    btag: ak.Array,
) -> ak.Array:
    return ak.zip(
        {
            "pt": pt,
            "eta": eta,
            "phi": phi,
            "mass": mass,
            "btag": btag,
        },
        with_name="Momentum4D",
    )


def reorder_by_field(array: ak.Array, field: str) -> ak.Array:
    return array[ak.argsort(array[field], axis=1, ascending=False)]


def singleton_event_field_to_numpy(array: ak.Array) -> np.ndarray:
    values = ak.fill_none(ak.firsts(ak.pad_none(array, 1, axis=1, clip=True)), np.nan)
    return ak.to_numpy(values)


def build_event_dataset(root_path: Path) -> ak.Array:
    vector.register_awkward()
    branches = [
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
    with uproot.open(root_path) as root_file:
        for tree_name in DELHES_TREE_CANDIDATES:
            if tree_name in root_file:
                arrays = root_file[tree_name].arrays(branches, library="ak")
                break
        else:
            raise KeyError(f"No Delphes/Events tree found in {root_path}")

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


def write_compressed_event_parquet(output_path: Path, dataset: ak.Array) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ak.to_parquet(dataset, output_path, compression="zstd")
    return output_path.resolve()


def format_file_size(num_bytes: int) -> str:
    return f"{num_bytes / 1024**3:.6f} GB"


def build_compression_summary_lines(
    mass: str,
    tanphi: str,
    compressed_event_paths: list[tuple[Path, Path]],
) -> list[str]:
    lines = [
        f"Point: mass={mass}, tanphi={tanphi}",
        f"Wrote {len(compressed_event_paths)} compressed event Parquet files.",
        "ROOT vs Parquet sizes:",
    ]
    total_root_size = 0
    total_parquet_size = 0
    for root_path, parquet_path in compressed_event_paths:
        root_size = root_path.stat().st_size
        parquet_size = parquet_path.stat().st_size
        total_root_size += root_size
        total_parquet_size += parquet_size
        lines.append(
            f"{parquet_path.stem}: "
            f"{format_file_size(root_size)} -> {format_file_size(parquet_size)}"
        )
    lines.append(
        f"total: {format_file_size(total_root_size)} -> "
        f"{format_file_size(total_parquet_size)}"
    )
    return lines


def write_compression_summary(
    output_dir: Path,
    mass: str,
    tanphi: str,
    compressed_event_paths: list[tuple[Path, Path]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "compression-info.txt"
    summary_text = "\n".join(
        build_compression_summary_lines(mass, tanphi, compressed_event_paths)
    )
    summary_path.write_text(f"{summary_text}\n", encoding="utf-8")
    return summary_path.resolve()


def compress_delphes_events(
    process_dirs: list[Path],
    mass: str,
    tanphi: str,
) -> list[tuple[Path, Path]]:
    output_dir = build_default_compressed_events_dir(mass, tanphi)
    written_paths: list[tuple[Path, Path]] = []
    for process_dir in process_dirs:
        _, _, root_path = extract_latest_delphes_root(process_dir)
        dataset = build_event_dataset(root_path)
        output_path = build_compressed_event_output(process_dir, mass, tanphi, output_dir)
        parquet_path = write_compressed_event_parquet(output_path, dataset)
        written_paths.append((root_path.resolve(), parquet_path))
    return written_paths


def build_xsec_rows(process_dirs: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for process_dir in process_dirs:
        process_match = PROCESS_DIR_PATTERN.fullmatch(process_dir.name)
        if process_match is None:
            raise ValueError(f"Unexpected process directory name: {process_dir.name}")

        _, _, banner_path = extract_latest_existing_banner(process_dir)
        cross_section_pb, cross_section_error_pb, generated_events = (
            extract_cross_section_from_banner(banner_path)
        )
        rows.append(
            {
                "process": process_match.group(1),
                "mass": extract_mass_from_banner(banner_path),
                "tanphi": extract_tanphi_from_banner(banner_path),
                "cross section pb": cross_section_pb,
                "cross section error pb": cross_section_error_pb,
                "N events (generated)": generated_events,
                "description": extract_process_description(process_dir),
            }
        )

    return rows


def write_xsec_csv(output_path: Path, rows: list[dict[str, str]]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_fieldnames = [
        "process",
        "mass",
        "tanphi",
        "cross section pb",
        "cross section error pb",
        "N events (generated)",
        "description",
    ]
    extra_fieldnames = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in base_fieldnames
        }
    )
    fieldnames = base_fieldnames + extra_fieldnames
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path.resolve()


def run_single_point(
    process_dirs: list[Path],
    args: argparse.Namespace,
    mass: str,
    tanphi: str,
    xsec_output: Path | None,
) -> int:
    card_text = build_mg5_card(
        process_dirs,
        args.nevents,
        args.mass_pdg,
        mass,
        args.tanphi_index,
        tanphi,
        args.ebeam1,
        args.ebeam2,
    )
    output_path = write_card(args.output, card_text)

    print(
        f"Wrote {output_path} for {len(process_dirs)} processes "
        f"with {args.nevents} events each."
    )
    print(f"Point: mass={mass}, tanphi={tanphi}")
    for process_dir in process_dirs:
        print(process_dir.name)

    if args.write_only:
        print("Skipping MadGraph launch because --write-only was requested.")
        return 0

    print(f"Launching MadGraph with {output_path}...")
    returncode = run_madgraph(args.mg5_bin.resolve(), output_path)
    if returncode != 0:
        print(f"MadGraph finished with a non-zero exit code: {returncode}")
        return returncode

    xsec_rows = build_xsec_rows(process_dirs)
    xsec_output_path = write_xsec_csv(
        xsec_output if xsec_output is not None else build_default_xsec_output(mass, tanphi),
        xsec_rows,
    )
    _, _, decay_banner_path = extract_latest_existing_banner(process_dirs[0])
    decay_output_path = write_decay_summary(
        build_default_decays_output(mass, tanphi),
        decay_banner_path,
    )
    compressed_event_paths = compress_delphes_events(process_dirs, mass, tanphi)
    compression_summary_path = write_compression_summary(
        build_default_compressed_events_dir(mass, tanphi),
        mass,
        tanphi,
        compressed_event_paths,
    )
    compression_summary_lines = build_compression_summary_lines(
        mass,
        tanphi,
        compressed_event_paths,
    )

    print("MadGraph finished successfully.")
    print(f"Wrote cross section summary to {xsec_output_path}.")
    print(f"Wrote decay summary to {decay_output_path}.")
    print(f"Wrote compression summary to {compression_summary_path}.")
    for line in compression_summary_lines[1:]:
        print(line)
    return 0


def main() -> int:
    args = parse_args()
    process_dirs = discover_process_directories(args.mg5_bin.resolve())
    mass_values = build_scan_values(
        args.mass,
        args.mass_range,
        args.mass_points,
        spacing="linear",
    )
    tanphi_values = build_scan_values(
        args.tanphi,
        args.tanphi_range,
        args.tanphi_points,
        spacing="logarithmic",
    )

    if len(mass_values) == 1 and len(tanphi_values) == 1:
        return run_single_point(
            process_dirs,
            args,
            mass_values[0],
            tanphi_values[0],
            args.xsec_output,
        )

    total_points = len(mass_values) * len(tanphi_values)
    print(
        f"Running scan over {len(mass_values)} mass points and "
        f"{len(tanphi_values)} tanphi points ({total_points} total points)."
    )
    point_index = 0
    for mass in mass_values:
        for tanphi in tanphi_values:
            point_index += 1
            print(f"[{point_index}/{total_points}] Starting mass={mass}, tanphi={tanphi}")
            returncode = run_single_point(process_dirs, args, mass, tanphi, None)
            if returncode != 0:
                return returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
