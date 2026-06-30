#!/usr/bin/env python3
"""Summarize total ASD affinity-label availability across known sources.

The summary combines these candidate label sources:

- ASD Hit-to-Lead activity table, joined by ASD target/modulator IDs.
- PDBbind v2020 general labels, joined by PDB ID.
- DrugWise BDB2020+ labels, joined by PDB ID.
- ChEMBL exact ligand-target activities from local caches produced by
  `scripts/asd_check_chembl_recovery.py`.

Aggregate counts are written to local outputs. Row-level source data
and API caches stay local because source database redistribution terms differ.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tarfile
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .discovery_chembl import (
    ACTIVITY_TYPES,
    DEFAULT_CACHE_DIR,
    DEFAULT_RCSB_CACHE,
    REPO_ROOT,
    STRICT_BINDING_TYPES,
    build_row_records,
    display_path,
    read_asd_rows,
    read_json,
    read_tsv,
    require_chembl_caches,
)


DEFAULT_ASD_AS_ARCHIVE = REPO_ROOT / "data/raw/asd/archives/ASD_Release_202309_AS.tar.gz"
DEFAULT_ASD_HL_ARCHIVE = REPO_ROOT / "data/raw/asd/archives/ASD_Release_202306_HL.tar.gz"
DEFAULT_DRUGWISE_DATA = REPO_ROOT / "DrugWise-Implementation-main/data"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/discovery/asd_total_affinity_availability.json"

ASD_HL_URL = (
    "https://mdl.shsmu.edu.cn/ASD2023Common/static_file/archive_2023/"
    "ASD_Release_202306_HL.tar.gz"
)


def download_if_missing(path: Path, url: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_hl_rows(path: Path) -> list[dict[str, str]]:
    with tarfile.open(path, "r:gz") as tar:
        member = next(
            m for m in tar.getmembers() if Path(m.name).name == "ASD_Release_202306_HL.txt"
        )
        handle = tar.extractfile(member)
        if handle is None:
            raise RuntimeError(f"Could not extract {member.name} from {path}")
        text = handle.read().decode("utf-8-sig")
    return list(csv.DictReader(text.splitlines(), delimiter="\t"))


def numeric(value: str | None) -> bool:
    if value is None:
        return False
    value = value.strip()
    if not value or value.lower() == "null":
        return False
    try:
        result = float(value)
    except ValueError:
        return False
    return math.isfinite(result)


def activity_value_field(row: dict[str, str]) -> str:
    for key in row:
        if key.startswith("Activity_Value"):
            return key
    raise KeyError("Could not find Activity_Value column in ASD Hit-to-Lead table")


def source_summary(indices: set[int], asd_rows: list[dict[str, str]], method_by_pdb: dict[str, str]) -> dict[str, Any]:
    rows = [asd_rows[index] for index in sorted(indices)]
    target_mod_keys = {
        target_modulator_key(row)
        for row in rows
        if target_modulator_key(row)
    }
    pdb_ids = {
        row.get("allosteric_pdb", "").strip().upper()
        for row in rows
        if row.get("allosteric_pdb", "").strip()
    }
    target_ids = {
        row.get("target_id", "").strip()
        for row in rows
        if row.get("target_id", "").strip()
    }
    method_counts = Counter(method_by_pdb.get(pdb_id, "UNKNOWN") for pdb_id in pdb_ids)
    row_method_counts = Counter(
        method_by_pdb.get(row.get("allosteric_pdb", "").strip().upper(), "UNKNOWN")
        for row in rows
    )
    return {
        "asd_rows": len(rows),
        "unique_pdb_ids": len(pdb_ids),
        "unique_target_ids": len(target_ids),
        "unique_target_modulator_keys": len(target_mod_keys),
        "unique_pdb_methods": dict(method_counts.most_common()),
        "row_methods": dict(row_method_counts.most_common()),
    }


def target_modulator_key(row: dict[str, str]) -> str:
    target_id = row.get("target_id", "").strip()
    modulator_serial = row.get("modulator_serial", "").strip()
    if target_id and modulator_serial:
        return f"{target_id}|{modulator_serial}"
    pdb_id = row.get("allosteric_pdb", "").strip().upper()
    alias = row.get("modulator_alias", "").strip()
    residue = row.get("modulator_resi", "").strip()
    if target_id and pdb_id and alias:
        return f"{target_id}|{pdb_id}|{alias}|{residue}"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asd-as-archive", type=Path, default=DEFAULT_ASD_AS_ARCHIVE)
    parser.add_argument("--asd-hl-archive", type=Path, default=DEFAULT_ASD_HL_ARCHIVE)
    parser.add_argument("--rcsb-cache", type=Path, default=DEFAULT_RCSB_CACHE)
    parser.add_argument("--chembl-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--drugwise-data", type=Path, default=DEFAULT_DRUGWISE_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--download-asd-hl",
        action="store_true",
        help="Download the ASD Hit-to-Lead archive if it is missing.",
    )
    args = parser.parse_args()

    if args.download_asd_hl:
        download_if_missing(args.asd_hl_archive, ASD_HL_URL)

    require_chembl_caches(args.chembl_cache_dir)
    asd_rows = read_asd_rows(args.asd_as_archive)
    method_by_pdb = {
        row.get("pdb_id", "").strip().upper(): row.get("methods", "").strip()
        for row in read_tsv(args.rcsb_cache)
        if row.get("pdb_id", "").strip()
    }

    row_indices = set(range(len(asd_rows)))

    pdbbind_path = args.drugwise_data / "PDBbindv2020_General.csv"
    pdbbind_pdbs = {
        row.get("PDBID", "").strip().upper()
        for row in read_csv(pdbbind_path)
        if row.get("PDBID", "").strip() and row.get("pK", "").strip()
    }
    pdbbind_indices = {
        index
        for index, row in enumerate(asd_rows)
        if row.get("allosteric_pdb", "").strip().upper() in pdbbind_pdbs
    }

    bdb_path = args.drugwise_data / "BDB2020+.csv"
    bdb_pdbs = {
        row.get("PDBID", "").strip().upper()
        for row in read_csv(bdb_path)
        if row.get("PDBID", "").strip() and row.get("pK", "").strip()
    }
    bdb_indices = {
        index
        for index, row in enumerate(asd_rows)
        if row.get("allosteric_pdb", "").strip().upper() in bdb_pdbs
    }

    hl_any_indices: set[int] = set()
    hl_strict_indices: set[int] = set()
    hl_endpoint_by_key: dict[str, set[str]] = {}
    if args.asd_hl_archive.exists():
        hl_rows = read_hl_rows(args.asd_hl_archive)
        value_key = activity_value_field(hl_rows[0])
        for hl_row in hl_rows:
            activity_type = hl_row.get("Activity_Type", "").strip()
            if activity_type not in ACTIVITY_TYPES:
                continue
            if not numeric(hl_row.get(value_key)):
                continue
            key = f"{hl_row.get('Target_ID', '').strip()}|{hl_row.get('Modulator_ID', '').strip()}"
            hl_endpoint_by_key.setdefault(key, set()).add(activity_type)

        for index, row in enumerate(asd_rows):
            key = f"{row.get('target_id', '').strip()}|{row.get('modulator_serial', '').strip()}"
            endpoints = hl_endpoint_by_key.get(key, set())
            if endpoints:
                hl_any_indices.add(index)
            if endpoints & STRICT_BINDING_TYPES:
                hl_strict_indices.add(index)
    else:
        hl_rows = []

    chemcomp_cache = read_json(args.chembl_cache_dir / "rcsb_chemcomp_by_alias.json", {})
    molecule_cache = read_json(args.chembl_cache_dir / "chembl_molecule_by_inchikey.json", {})
    target_cache = read_json(args.chembl_cache_dir / "chembl_target_by_uniprot.json", {})
    activity_cache = read_json(args.chembl_cache_dir / "chembl_activity_by_molecule_target.json", {})
    row_records, _, _, _ = build_row_records(asd_rows, chemcomp_cache, molecule_cache, target_cache)

    chembl_any_indices: set[int] = set()
    chembl_strict_indices: set[int] = set()
    for record in row_records:
        activities = []
        for molecule_id in record["molecule_ids"]:
            for target_id in record["target_chembl_ids"]:
                key = f"{molecule_id}|{target_id}"
                activities.extend((activity_cache.get(key) or {}).get("activities", []))
        endpoints = {activity.get("standard_type") for activity in activities}
        if endpoints & set(ACTIVITY_TYPES):
            chembl_any_indices.add(record["row_index"])
        if endpoints & STRICT_BINDING_TYPES:
            chembl_strict_indices.add(record["row_index"])

    source_indices = {
        "asd_hitlead_any_activity": hl_any_indices,
        "asd_hitlead_strict_kd_ki": hl_strict_indices,
        "pdbbind_v2020_pdbid_candidate": pdbbind_indices,
        "drugwise_bdb2020plus_pdbid_candidate": bdb_indices,
        "chembl_exact_ligand_target_any_activity": chembl_any_indices,
        "chembl_exact_ligand_target_strict_kd_ki": chembl_strict_indices,
    }

    broad_union = hl_any_indices | pdbbind_indices | bdb_indices | chembl_any_indices
    strict_observed_union = hl_strict_indices | chembl_strict_indices
    strict_plus_pdbbind_candidate_union = (
        hl_strict_indices | chembl_strict_indices | pdbbind_indices | bdb_indices
    )
    all_candidate_union = broad_union

    overlap_matrix: dict[str, dict[str, int]] = {}
    names = sorted(source_indices)
    for left in names:
        overlap_matrix[left] = {}
        for right in names:
            overlap_matrix[left][right] = len(source_indices[left] & source_indices[right])

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "asd_as_archive": display_path(args.asd_as_archive),
            "asd_hl_archive": display_path(args.asd_hl_archive),
            "rcsb_cache": display_path(args.rcsb_cache),
            "chembl_cache_dir": display_path(args.chembl_cache_dir),
            "drugwise_data": display_path(args.drugwise_data),
        },
        "definitions": {
            "broad_activity_or_pk_like_union": "ASD Hit-to-Lead non-null selected endpoints, PDBbind v2020 PDB-ID candidates, DrugWise BDB2020+ PDB-ID candidates, and ChEMBL exact ligand-target selected endpoints.",
            "strict_observed_kd_ki_union": "Only sources where Kd/Ki endpoint type is explicit in this pass: ASD Hit-to-Lead and ChEMBL.",
            "strict_plus_pdbbind_candidate_union": "Strict observed Kd/Ki rows plus PDBbind/BDB PDB-ID candidate pK rows whose original endpoint type has not yet been split here.",
        },
        "source_counts": {
            name: source_summary(indices, asd_rows, method_by_pdb)
            for name, indices in source_indices.items()
        },
        "unions": {
            "broad_activity_or_pk_like_union": source_summary(
                broad_union, asd_rows, method_by_pdb
            ),
            "strict_observed_kd_ki_union": source_summary(
                strict_observed_union, asd_rows, method_by_pdb
            ),
            "strict_plus_pdbbind_candidate_union": source_summary(
                strict_plus_pdbbind_candidate_union, asd_rows, method_by_pdb
            ),
            "all_candidate_union": source_summary(all_candidate_union, asd_rows, method_by_pdb),
        },
        "overlap_matrix_asd_rows": overlap_matrix,
        "notes": [
            "PDBbind and BDB2020+ are PDB-ID candidate labels and require ligand identity validation against ASD modulator annotations.",
            "ChEMBL labels are exact molecule-target labels after RCSB InChIKey and UniProt mapping, but not necessarily exact PDB-complex labels.",
            "ASD Hit-to-Lead labels are joined by ASD target/modulator IDs and preserve endpoint type.",
            "Row-level label tables remain local/ignored; this report stores aggregate counts only.",
            f"ASD Hit-to-Lead archive rows read: {len(hl_rows)}.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if row_indices - all_candidate_union:
        print(
            f"Unlabeled by these candidate routes: {len(row_indices - all_candidate_union)} ASD rows",
        )


if __name__ == "__main__":
    main()
