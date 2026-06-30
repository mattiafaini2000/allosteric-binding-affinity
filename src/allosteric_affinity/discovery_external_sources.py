#!/usr/bin/env python3
"""Summarize external affinity-label overlap for ASD allosteric complexes.

This script intentionally writes aggregate summaries only. Row-level ASD data
and third-party labels remain local because ASD/PDBbind/BindingDB redistribution
terms need to be handled carefully.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import display_path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASD_ARCHIVE = REPO_ROOT / "data/raw/asd/archives/ASD_Release_202309_AS.tar.gz"
DEFAULT_RCSB_CACHE = REPO_ROOT / "data/interim/asd/rcsb_pdb_methods.tsv"
DEFAULT_DRUGWISE_DATA = REPO_ROOT / "DrugWise-Implementation-main/data"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/discovery/asd_external_affinity_source_recovery.json"


LABEL_SOURCES = [
    {
        "name": "PDBbindv2020_General",
        "filename": "PDBbindv2020_General.csv",
        "pdb_col": "PDBID",
        "pk_col": "pK",
        "kind": "pdbbind",
    },
    {
        "name": "PDBbindv2016_GeneralSet",
        "filename": "PDBbindv2016_GeneralSet.csv",
        "pdb_col": "PDBID",
        "pk_col": "pK",
        "kind": "pdbbind",
    },
    {
        "name": "PDBbindv2016_RefinedSet",
        "filename": "PDBbindv2016_RefinedSet.csv",
        "pdb_col": "PDBID",
        "pk_col": "pK",
        "kind": "pdbbind",
    },
    {
        "name": "CASF_2016_CoreSet",
        "filename": "CASF_2016_CoreSet.csv",
        "pdb_col": "PDBID",
        "pk_col": "pK",
        "kind": "casf",
    },
    {
        "name": "BDB2020plus",
        "filename": "BDB2020+.csv",
        "pdb_col": "PDBID",
        "pk_col": "pK",
        "kind": "bindingdb_derived",
    },
]


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


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def unique_count(rows: list[dict[str, str]], field: str) -> int:
    return len({row.get(field, "").upper() for row in rows if row.get(field, "")})


def summarize_source(
    source: dict[str, str],
    asd_rows: list[dict[str, str]],
    method_by_pdb: dict[str, str],
    drugwise_data: Path,
) -> dict[str, Any]:
    path = drugwise_data / source["filename"]
    if not path.exists():
        return {
            "source": source["name"],
            "kind": source["kind"],
            "status": "missing",
            "path": display_path(path),
        }

    label_rows = read_csv(path)
    pdb_to_pk: dict[str, str] = {}
    for row in label_rows:
        pdb_id = row.get(source["pdb_col"], "").strip().upper()
        pk = row.get(source["pk_col"], "").strip()
        if pdb_id and pk:
            pdb_to_pk[pdb_id] = pk

    matched_rows = [
        row
        for row in asd_rows
        if row.get("allosteric_pdb", "").strip().upper() in pdb_to_pk
    ]
    matched_pdbs = {
        row.get("allosteric_pdb", "").strip().upper()
        for row in matched_rows
        if row.get("allosteric_pdb", "").strip()
    }
    matched_methods = Counter(method_by_pdb.get(pdb_id, "UNKNOWN") for pdb_id in matched_pdbs)
    matched_row_methods = Counter(
        method_by_pdb.get(row.get("allosteric_pdb", "").strip().upper(), "UNKNOWN")
        for row in matched_rows
    )
    matched_classes = Counter(row.get("modulator_class", "") for row in matched_rows)
    matched_targets = {
        row.get("target_id", "").strip()
        for row in matched_rows
        if row.get("target_id", "").strip()
    }

    pk_values = []
    for row in matched_rows:
        pk = pdb_to_pk.get(row.get("allosteric_pdb", "").strip().upper())
        try:
            if pk is not None:
                pk_values.append(float(pk))
        except ValueError:
            pass

    pk_summary = None
    if pk_values:
        pk_summary = {
            "count": len(pk_values),
            "min": round(min(pk_values), 3),
            "max": round(max(pk_values), 3),
            "mean": round(statistics.fmean(pk_values), 3),
        }

    return {
        "source": source["name"],
        "kind": source["kind"],
        "status": "ok",
        "path": display_path(path),
        "label_rows": len(label_rows),
        "unique_label_pdb_ids": len(pdb_to_pk),
        "asd_rows_matched_by_pdb_id": len(matched_rows),
        "asd_unique_pdb_ids_matched": len(matched_pdbs),
        "asd_unique_target_ids_matched": len(matched_targets),
        "matched_unique_pdb_methods": dict(matched_methods.most_common()),
        "matched_as_row_methods": dict(matched_row_methods.most_common()),
        "matched_as_row_modulator_classes": dict(matched_classes.most_common()),
        "matched_as_row_pk_summary": pk_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asd-archive", type=Path, default=DEFAULT_ASD_ARCHIVE)
    parser.add_argument("--rcsb-cache", type=Path, default=DEFAULT_RCSB_CACHE)
    parser.add_argument("--drugwise-data", type=Path, default=DEFAULT_DRUGWISE_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    asd_rows = read_asd_rows(args.asd_archive)
    methods = read_tsv(args.rcsb_cache)
    method_by_pdb = {
        row.get("pdb_id", "").strip().upper(): row.get("methods", "").strip()
        for row in methods
        if row.get("pdb_id", "").strip()
    }

    sources = [
        summarize_source(source, asd_rows, method_by_pdb, args.drugwise_data)
        for source in LABEL_SOURCES
    ]

    union_pdbbind_2020_or_bdb: set[str] = set()
    for source in sources:
        if source["source"] not in {"PDBbindv2020_General", "BDB2020plus"}:
            continue
        config = next(item for item in LABEL_SOURCES if item["name"] == source["source"])
        path = args.drugwise_data / config["filename"]
        if source["status"] != "ok" or not path.exists():
            continue
        for row in read_csv(path):
            pdb_id = row.get(config["pdb_col"], "").strip().upper()
            pk = row.get(config["pk_col"], "").strip()
            if pdb_id and pk:
                union_pdbbind_2020_or_bdb.add(pdb_id)

    union_matches = [
        row
        for row in asd_rows
        if row.get("allosteric_pdb", "").strip().upper() in union_pdbbind_2020_or_bdb
    ]
    union_unique_pdbs = {
        row.get("allosteric_pdb", "").strip().upper()
        for row in union_matches
        if row.get("allosteric_pdb", "").strip()
    }

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "asd_archive": display_path(args.asd_archive),
            "rcsb_cache": display_path(args.rcsb_cache),
            "drugwise_data": display_path(args.drugwise_data),
        },
        "asd": {
            "site_rows": len(asd_rows),
            "unique_pdb_ids": unique_count(asd_rows, "allosteric_pdb"),
            "unique_target_ids": len(
                {row.get("target_id", "") for row in asd_rows if row.get("target_id", "")}
            ),
        },
        "sources": sources,
        "union_pdbbind2020_or_bdb2020plus": {
            "asd_rows_matched_by_pdb_id": len(union_matches),
            "asd_unique_pdb_ids_matched": len(union_unique_pdbs),
            "matched_unique_pdb_methods": dict(
                Counter(method_by_pdb.get(pdb_id, "UNKNOWN") for pdb_id in union_unique_pdbs)
            ),
        },
        "notes": [
            "PDB-ID overlap is not sufficient to prove that the external pK label is for the exact ASD allosteric ligand when a PDB entry contains multiple ligands/cofactors/ions.",
            "Exact ChEMBL or BindingDB recovery needs ligand identity mapping plus target mapping and assay filtering.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
