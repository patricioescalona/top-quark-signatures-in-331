#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


GENERATED_SIGNAL_CONTAINER_NAME = "generated-signal"
GENERATED_DIR_PATTERN = re.compile(r"^generated-m-(.+)-tanphi-(.+)$")


def format_value_for_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    sanitized = sanitized.strip("-")
    return sanitized or "value"


def build_generated_signal_root(base_dir: Path) -> Path:
    return base_dir / GENERATED_SIGNAL_CONTAINER_NAME


def build_generated_dir(base_dir: Path, mass: str, tanphi: str) -> Path:
    dirname = (
        f"generated-m-{format_value_for_filename(mass)}"
        f"-tanphi-{format_value_for_filename(tanphi)}"
    )
    return build_generated_signal_root(base_dir) / dirname


def discover_generated_dirs(base_dir: Path) -> list[tuple[str, str, Path]]:
    generated_root = build_generated_signal_root(base_dir)
    generated_dirs: list[tuple[str, str, Path]] = []

    for path in sorted(generated_root.glob("generated-m-*-tanphi-*")):
        if not path.is_dir():
            continue
        match = GENERATED_DIR_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        generated_dirs.append((match.group(1), match.group(2), path.resolve()))

    if not generated_dirs:
        raise FileNotFoundError(f"No generated signal folders found in {generated_root}")

    return generated_dirs
