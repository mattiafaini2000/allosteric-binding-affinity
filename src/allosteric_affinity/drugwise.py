"""Adapters for the external DrugWise feature implementation.

The external ``get_scores`` module expects its repository as the working
directory. Importing this module does not import DrugWise or load a model.
"""

from __future__ import annotations

import csv
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


README_SOURCE = "lorentz_tau1p5_power2p5"
DEFAULT_SOURCE = "lorentz_tau0p5_power5"
LORENTZ_PARAMETER_SETS = {
    README_SOURCE: {"kernel_type": "lorentz_kernel", "tau": 1.5, "power": 2.5, "cutoff": 6.0},
    DEFAULT_SOURCE: {"kernel_type": "lorentz_kernel", "tau": 0.5, "power": 5.0, "cutoff": 6.0},
}


@contextmanager
def drugwise_import_context(drugwise_dir: Path) -> Iterator[None]:
    """Temporarily expose external ``get_scores`` and its relative data paths."""
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    os.chdir(drugwise_dir)
    sys.path.insert(0, str(drugwise_dir / "src"))
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


def load_canonical_ligand_atom_types(drugwise_dir: Path) -> dict[str, str]:
    """Read the DrugWise SYBYL atom-type spelling table."""
    path = drugwise_dir / "utils/ligand_SYBYL_atom_types.csv"
    canonical: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            atom_type = row["AtomType"].strip()
            canonical[atom_type.lower()] = atom_type
    return canonical


def canonicalize_mol2_atom_types(mol2_path: Path, drugwise_dir: Path) -> None:
    """Match MOL2 atom-type case to the external DrugWise vocabulary in place."""
    canonical = load_canonical_ligand_atom_types(drugwise_dir)
    lines = mol2_path.read_text(encoding="utf-8", errors="replace").splitlines()
    output: list[str] = []
    in_atom_section = False

    changed = False
    for line in lines:
        if line.startswith("@<TRIPOS>"):
            in_atom_section = line.strip().upper() == "@<TRIPOS>ATOM"
            output.append(line)
            continue
        if in_atom_section and line.strip():
            parts = line.split()
            if len(parts) >= 6:
                atom_type = parts[5]
                canonical_type = canonical.get(atom_type.lower(), atom_type)
                if canonical_type != atom_type:
                    parts[5] = canonical_type
                    changed = True
                    line = " ".join(parts)
        output.append(line)
    if changed:
        mol2_path.write_text("\n".join([*output, ""]), encoding="utf-8", newline="\n")


def flatten_score_frame(frame: pd.DataFrame, type_column: str) -> dict[str, float]:
    """Flatten DrugWise atom-type rows into ``type_descriptor`` feature names."""
    value_columns = [column for column in frame.columns if column != type_column]
    values: dict[str, float] = {}
    for _, row in frame.iterrows():
        atom_type = row[type_column]
        for column in value_columns:
            values[f"{atom_type}_{column}"] = float(row[column])
    return values


def compute_base_descriptors(
    scorers: dict[str, Any], protein_file: Path, ligand_file: Path
) -> dict[str, float]:
    """Combine GGL, pairwise, and triplet descriptors in the original order."""
    ggl_values = flatten_score_frame(
        scorers["ggl"].get_ggl_score(str(protein_file), str(ligand_file)), "ATOM_PAIR"
    )
    pair_values = flatten_score_frame(
        scorers["pairwise"].get_pairwise_score(str(protein_file), str(ligand_file)),
        "ATOM_PAIR",
    )
    triplet_values = flatten_score_frame(
        scorers["triplet"].get_triplet(str(protein_file), str(ligand_file)), "type"
    )
    base_values: dict[str, float] = {}
    base_values.update(ggl_values)
    base_values.update(pair_values)
    base_values.update(triplet_values)
    return base_values


def strip_pandas_suffix(feature_name: str) -> str:
    """Map selected ``_x``/``_y`` columns to released base descriptors."""
    if feature_name.endswith("_x") or feature_name.endswith("_y"):
        return feature_name[:-2]
    return feature_name


def source_for_feature(feature_name: str, variant: dict[str, str]) -> str:
    """Select the configured Lorentz source for a model feature column."""
    if feature_name.endswith("_x"):
        return variant["x_source"]
    if feature_name.endswith("_y"):
        return variant["y_source"]
    return variant["unsuffixed_source"]
