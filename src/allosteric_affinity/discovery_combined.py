#!/usr/bin/env python3
"""Summarize combined PDBbind and ChEMBL affinity-label recovery for ASD."""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .discovery_chembl import build_row_records, read_json, require_chembl_caches
from .io import display_path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASD_ARCHIVE = REPO_ROOT / "data/raw/asd/archives/ASD_Release_202309_AS.tar.gz"
DEFAULT_PDBBIND = REPO_ROOT / "DrugWise-Implementation-main/data/PDBbindv2020_General.csv"
DEFAULT_CHEMBL_CACHE = REPO_ROOT / "data/interim/chembl"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/discovery/asd_combined_affinity_recovery_summary.json"


def read_asd_rows(archive_path: Path) -> list[dict[str, str]]:
    with tarfile.open(archive_path, "r:gz") as tar:
        member = next(
            m for m in tar.getmembers() if Path(m.name).name == "ASD_Release_202309_AS.txt"
        )
        handle = tar.extractfile(member)
        if handle is None:
            raise RuntimeError(f"Could not extract {member.name} from {archive_path}")
        text = handle.read().decode("utf-8-sig")
    return list(csv.DictReader(text.splitlines(), delimiter="\t"))


def read_pdbbind_pdb_ids(path: Path) -> set[str]:
    pdb_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            pdb_id = row.get("PDBID", "").strip().upper()
            pk = row.get("pK", "").strip()
            if pdb_id and pk:
                pdb_ids.add(pdb_id)
    return pdb_ids


def summarize_indices(asd_rows: list[dict[str, str]], indices: set[int]) -> dict[str, int]:
    rows = [asd_rows[index] for index in indices]
    return {
        "asd_rows": len(indices),
        "unique_pdb_ids": len(
            {
                row.get("allosteric_pdb", "").strip().upper()
                for row in rows
                if row.get("allosteric_pdb", "").strip()
            }
        ),
        "unique_target_ids": len(
            {row.get("target_id", "").strip() for row in rows if row.get("target_id", "").strip()}
        ),
        "unique_target_modulator_keys": len(
            {
                f"{row.get('target_id', '').strip()}|{row.get('modulator_serial', '').strip()}"
                for row in rows
                if row.get("target_id", "").strip() or row.get("modulator_serial", "").strip()
            }
        ),
    }


def get_chembl_recoverable_rows(
    asd_rows: list[dict[str, str]], cache_dir: Path
) -> tuple[set[int], set[int], Counter[str]]:
    chemcomp_cache = read_json(cache_dir / "rcsb_chemcomp_by_alias.json", {})
    molecule_cache = read_json(cache_dir / "chembl_molecule_by_inchikey.json", {})
    target_cache = read_json(cache_dir / "chembl_target_by_uniprot.json", {})
    activity_cache = read_json(cache_dir / "chembl_activity_by_molecule_target.json", {})
    row_records, *_ = build_row_records(asd_rows, chemcomp_cache, molecule_cache, target_cache)

    chembl_any_rows: set[int] = set()
    chembl_strict_rows: set[int] = set()
    row_type_counter: Counter[str] = Counter()

    for record in row_records:
        activities: list[dict[str, Any]] = []
        for molecule_id in record["molecule_ids"]:
            for target_id in record["target_chembl_ids"]:
                activities.extend(
                    (activity_cache.get(f"{molecule_id}|{target_id}") or {}).get(
                        "activities", []
                    )
                )
        types = {activity.get("standard_type") for activity in activities if activity.get("standard_type")}
        if types:
            chembl_any_rows.add(record["row_index"])
            row_type_counter.update(types)
        if types & {"Kd", "Ki"}:
            chembl_strict_rows.add(record["row_index"])

    return chembl_any_rows, chembl_strict_rows, row_type_counter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asd-archive", type=Path, default=DEFAULT_ASD_ARCHIVE)
    parser.add_argument("--pdbbind", type=Path, default=DEFAULT_PDBBIND)
    parser.add_argument("--chembl-cache", type=Path, default=DEFAULT_CHEMBL_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    require_chembl_caches(args.chembl_cache)
    asd_rows = read_asd_rows(args.asd_archive)
    pdbbind_pdb_ids = read_pdbbind_pdb_ids(args.pdbbind)
    pdbbind_rows = {
        index
        for index, row in enumerate(asd_rows)
        if row.get("allosteric_pdb", "").strip().upper() in pdbbind_pdb_ids
    }
    chembl_any_rows, chembl_strict_rows, chembl_type_counter = get_chembl_recoverable_rows(
        asd_rows, args.chembl_cache
    )

    union_any = pdbbind_rows | chembl_any_rows
    union_strict = pdbbind_rows | chembl_strict_rows

    row_bins = Counter()
    for index in range(len(asd_rows)):
        in_pdbbind = index in pdbbind_rows
        in_chembl_any = index in chembl_any_rows
        in_chembl_strict = index in chembl_strict_rows
        if in_pdbbind and in_chembl_any:
            row_bins["both_pdbbind_and_chembl_any"] += 1
        elif in_pdbbind:
            row_bins["pdbbind_only"] += 1
        elif in_chembl_any:
            row_bins["chembl_any_only"] += 1
        if in_pdbbind and in_chembl_strict:
            row_bins["both_pdbbind_and_chembl_strict"] += 1
        elif in_chembl_strict:
            row_bins["chembl_strict_only"] += 1

    pdbbind_pdb_matched = {
        asd_rows[index].get("allosteric_pdb", "").strip().upper() for index in pdbbind_rows
    }
    chembl_any_pdb_matched = {
        asd_rows[index].get("allosteric_pdb", "").strip().upper() for index in chembl_any_rows
    }
    chembl_strict_pdb_matched = {
        asd_rows[index].get("allosteric_pdb", "").strip().upper() for index in chembl_strict_rows
    }

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "asd_archive": display_path(args.asd_archive),
            "pdbbind": display_path(args.pdbbind),
            "chembl_cache": display_path(args.chembl_cache),
        },
        "definitions": {
            "pdbbind_v2020": "PDBbind v2020 general pK joined by exact PDB ID; candidate exact-complex labels requiring ASD ligand identity validation.",
            "chembl_any": "Exact RCSB-ligand-InChIKey to ChEMBL-molecule and UniProt to ChEMBL-target activity with pChEMBL for Kd, Ki, IC50, EC50, or AC50.",
            "chembl_strict_kd_ki": "Subset of ChEMBL exact ligand-target recovery with Kd or Ki endpoint.",
        },
        "source_counts": {
            "pdbbind_v2020": summarize_indices(asd_rows, pdbbind_rows),
            "chembl_any": summarize_indices(asd_rows, chembl_any_rows),
            "chembl_strict_kd_ki": summarize_indices(asd_rows, chembl_strict_rows),
        },
        "combined_counts": {
            "pdbbind_v2020_or_chembl_any": summarize_indices(asd_rows, union_any),
            "pdbbind_v2020_or_chembl_strict_kd_ki": summarize_indices(asd_rows, union_strict),
        },
        "overlap_counts_by_asd_row": dict(row_bins.most_common()),
        "overlap_counts_by_unique_pdb": {
            "pdbbind_and_chembl_any": len(pdbbind_pdb_matched & chembl_any_pdb_matched),
            "pdbbind_and_chembl_strict_kd_ki": len(
                pdbbind_pdb_matched & chembl_strict_pdb_matched
            ),
            "pdbbind_or_chembl_any": len(pdbbind_pdb_matched | chembl_any_pdb_matched),
            "pdbbind_or_chembl_strict_kd_ki": len(
                pdbbind_pdb_matched | chembl_strict_pdb_matched
            ),
        },
        "chembl_row_counts_by_endpoint_type": dict(chembl_type_counter.most_common()),
        "notes": [
            "The broad union counts rows with either a PDBbind pK candidate or any selected ChEMBL pChEMBL endpoint.",
            "The strict union keeps all PDBbind pK candidates and adds only ChEMBL Kd/Ki rows; PDBbind endpoint provenance was not split in this pass.",
            "Neither union should be used as a final supervised set before resolving duplicate labels and validating ligand identity.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
