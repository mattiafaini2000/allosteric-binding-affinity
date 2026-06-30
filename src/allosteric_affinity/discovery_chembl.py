#!/usr/bin/env python3
"""Estimate ChEMBL activity-label recovery for ASD allosteric complexes.

The recovery route is deliberately conservative:

1. Split ASD `modulator_alias` values into candidate RCSB chemical-component IDs.
2. Query RCSB CCD metadata for each component and extract InChIKey.
3. Query ChEMBL molecules by exact standard InChIKey.
4. Query ChEMBL targets by ASD UniProt accession.
5. Query ChEMBL activities for exact molecule-target pairs.

Aggregate summaries are written to local outputs. Local caches in
`data/interim/chembl/` may contain API response snippets and are git-ignored.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import tarfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .io import display_path, read_json, write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASD_ARCHIVE = REPO_ROOT / "data/raw/asd/archives/ASD_Release_202309_AS.tar.gz"
DEFAULT_RCSB_CACHE = REPO_ROOT / "data/interim/asd/rcsb_pdb_methods.tsv"
DEFAULT_CACHE_DIR = REPO_ROOT / "data/interim/chembl"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/discovery/asd_chembl_recovery_summary.json"

ACTIVITY_TYPES = ("Kd", "Ki", "IC50", "EC50", "AC50")
STRICT_BINDING_TYPES = {"Kd", "Ki"}


def require_chembl_caches(cache_dir: Path) -> None:
    """Require the four local identity and activity caches used for ASD rows."""
    names = (
        "rcsb_chemcomp_by_alias.json",
        "chembl_molecule_by_inchikey.json",
        "chembl_target_by_uniprot.json",
        "chembl_activity_by_molecule_target.json",
    )
    missing = [cache_dir / name for name in names if not (cache_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "ChEMBL recovery needs local caches: "
            + ", ".join(str(path) for path in missing)
            + ". Use --online with the acquisition workflow to populate them."
        )


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


def normalize_alias_token(token: str) -> str | None:
    token = token.strip().upper()
    if not token:
        return None
    if token.startswith("CHAIN"):
        return None
    if "PEPTIDE" in token:
        return None
    if not re.fullmatch(r"[A-Z0-9]{1,6}", token):
        return None
    return token


def split_aliases(alias: str) -> list[str]:
    tokens = re.split(r";|@@|,|\s+", alias or "")
    normalized = [normalize_alias_token(token) for token in tokens]
    return sorted({token for token in normalized if token})


def split_uniprots(value: str) -> list[str]:
    tokens = re.split(r";|,|\s+", value or "")
    return sorted({token.strip().upper() for token in tokens if token.strip()})


def request_json(url: str, timeout: int = 30, attempts: int = 3) -> dict[str, Any] | None:
    import requests

    for attempt in range(attempts):
        try:
            response = requests.get(
                url,
                timeout=timeout,
                headers={"Accept": "application/json", "User-Agent": "asd-chembl-recovery/0.1"},
            )
            if response.status_code == 404:
                return None
            if response.status_code in {429, 500, 502, 503, 504}:
                time.sleep(1.5 * (attempt + 1))
                continue
            response.raise_for_status()
            return response.json()
        except Exception:
            if attempt == attempts - 1:
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def get_rcsb_chemcomp(alias: str) -> dict[str, Any]:
    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{alias}"
    data = request_json(url)
    result: dict[str, Any] = {"status": "missing", "inchikey": None, "name": None}
    if not data:
        return result

    descriptors = data.get("pdbx_chem_comp_descriptor") or []
    inchikey = None
    for descriptor in descriptors:
        if descriptor.get("type") == "InChIKey":
            inchikey = descriptor.get("descriptor")
            break
    chem_comp = data.get("chem_comp") or {}
    result.update(
        {
            "status": "ok",
            "inchikey": inchikey,
            "name": chem_comp.get("name"),
            "type": chem_comp.get("type"),
            "formula": chem_comp.get("formula"),
        }
    )
    return result


def get_chembl_molecules(inchikey: str) -> dict[str, Any]:
    url = (
        "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
        f"?molecule_structures__standard_inchi_key={inchikey}&limit=100"
    )
    data = request_json(url)
    molecules = []
    for molecule in (data or {}).get("molecules", []):
        hierarchy = molecule.get("molecule_hierarchy") or {}
        molecules.append(
            {
                "molecule_chembl_id": molecule.get("molecule_chembl_id"),
                "parent_molecule_chembl_id": hierarchy.get("parent_chembl_id"),
                "pref_name": molecule.get("pref_name"),
                "molecule_type": molecule.get("molecule_type"),
            }
        )
    return {
        "status": "ok" if molecules else "missing",
        "molecules": molecules,
        "total_count": (data or {}).get("page_meta", {}).get("total_count", 0),
    }


def get_chembl_targets(uniprot: str) -> dict[str, Any]:
    url = (
        "https://www.ebi.ac.uk/chembl/api/data/target.json"
        f"?target_components__accession={uniprot}&limit=100"
    )
    data = request_json(url)
    targets = []
    for target in (data or {}).get("targets", []):
        targets.append(
            {
                "target_chembl_id": target.get("target_chembl_id"),
                "pref_name": target.get("pref_name"),
                "organism": target.get("organism"),
                "target_type": target.get("target_type"),
            }
        )
    return {
        "status": "ok" if targets else "missing",
        "targets": targets,
        "total_count": (data or {}).get("page_meta", {}).get("total_count", 0),
    }


def get_chembl_activities(molecule_id: str, target_id: str) -> dict[str, Any]:
    standard_types = ",".join(ACTIVITY_TYPES)
    url = (
        "https://www.ebi.ac.uk/chembl/api/data/activity.json"
        f"?molecule_chembl_id={molecule_id}"
        f"&target_chembl_id={target_id}"
        f"&standard_type__in={standard_types}"
        "&pchembl_value__isnull=false"
        "&limit=1000"
    )
    data = request_json(url, timeout=45)
    activities = []
    for activity in (data or {}).get("activities", []):
        pchembl = activity.get("pchembl_value")
        standard_type = activity.get("standard_type")
        if not pchembl or standard_type not in ACTIVITY_TYPES:
            continue
        activities.append(
            {
                "activity_id": activity.get("activity_id"),
                "standard_type": standard_type,
                "standard_value": activity.get("standard_value"),
                "standard_units": activity.get("standard_units"),
                "pchembl_value": pchembl,
                "standard_relation": activity.get("standard_relation"),
                "assay_chembl_id": activity.get("assay_chembl_id"),
                "assay_type": activity.get("assay_type"),
                "document_chembl_id": activity.get("document_chembl_id"),
                "document_year": activity.get("document_year"),
                "target_chembl_id": activity.get("target_chembl_id"),
                "molecule_chembl_id": activity.get("molecule_chembl_id"),
            }
        )
    return {
        "status": "ok" if activities else "missing",
        "activities": activities,
        "total_count": (data or {}).get("page_meta", {}).get("total_count", 0),
    }


def update_cache_parallel(
    cache: dict[str, Any],
    keys: Iterable[str],
    fetch,
    cache_path: Path,
    workers: int,
    label: str,
) -> dict[str, Any]:
    missing = [key for key in sorted(set(keys)) if key not in cache]
    if not missing:
        print(f"{label}: cache complete ({len(cache)} entries)")
        return cache

    print(f"{label}: fetching {len(missing)} missing entries ({len(cache)} cached)")
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_key = {executor.submit(fetch, key): key for key in missing}
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                cache[key] = future.result()
            except Exception as exc:  # defensive, fetchers already catch network issues
                cache[key] = {"status": "error", "error": str(exc)}
            completed += 1
            if completed % 50 == 0:
                write_json(cache_path, cache)
                print(f"{label}: {completed}/{len(missing)} fetched")
    write_json(cache_path, cache)
    return cache


def build_row_records(
    asd_rows: list[dict[str, str]],
    chemcomp_cache: dict[str, Any],
    molecule_cache: dict[str, Any],
    target_cache: dict[str, Any],
) -> tuple[list[dict[str, Any]], set[str], set[str], set[tuple[str, str]]]:
    row_records = []
    needed_inchikeys: set[str] = set()
    needed_uniprots: set[str] = set()
    needed_pairs: set[tuple[str, str]] = set()

    for idx, row in enumerate(asd_rows):
        aliases = split_aliases(row.get("modulator_alias", ""))
        uniprots = split_uniprots(row.get("pdb_uniprot", ""))
        inchikeys = sorted(
            {
                (chemcomp_cache.get(alias) or {}).get("inchikey")
                for alias in aliases
                if (chemcomp_cache.get(alias) or {}).get("inchikey")
            }
        )
        needed_inchikeys.update(inchikeys)
        needed_uniprots.update(uniprots)

        molecule_ids: set[str] = set()
        for inchikey in inchikeys:
            for molecule in (molecule_cache.get(inchikey) or {}).get("molecules", []):
                for field in ("molecule_chembl_id", "parent_molecule_chembl_id"):
                    value = molecule.get(field)
                    if value:
                        molecule_ids.add(value)

        target_ids: set[str] = set()
        for uniprot in uniprots:
            for target in (target_cache.get(uniprot) or {}).get("targets", []):
                target_id = target.get("target_chembl_id")
                if target_id:
                    target_ids.add(target_id)

        for molecule_id in molecule_ids:
            for target_id in target_ids:
                needed_pairs.add((molecule_id, target_id))

        row_records.append(
            {
                "row_index": idx,
                "allosteric_pdb": row.get("allosteric_pdb", "").upper(),
                "target_id": row.get("target_id", ""),
                "modulator_serial": row.get("modulator_serial", ""),
                "modulator_alias": row.get("modulator_alias", ""),
                "modulator_class": row.get("modulator_class", ""),
                "pdb_uniprot": row.get("pdb_uniprot", ""),
                "aliases": aliases,
                "uniprots": uniprots,
                "inchikeys": inchikeys,
                "molecule_ids": sorted(molecule_ids),
                "target_chembl_ids": sorted(target_ids),
            }
        )

    return row_records, needed_inchikeys, needed_uniprots, needed_pairs


def summarize(
    asd_rows: list[dict[str, str]],
    row_records: list[dict[str, Any]],
    activity_cache: dict[str, Any],
    method_by_pdb: dict[str, str],
    args: argparse.Namespace,
    aliases: set[str],
    inchikeys: set[str],
    uniprots: set[str],
    pairs: set[tuple[str, str]],
    chemcomp_cache: dict[str, Any],
    molecule_cache: dict[str, Any],
    target_cache: dict[str, Any],
) -> dict[str, Any]:
    recoverable_rows = []
    strict_rows = []
    row_type_counter: Counter[str] = Counter()
    activity_records_by_type: Counter[str] = Counter()
    unique_activity_ids: set[str] = set()

    for record in row_records:
        activities = []
        for molecule_id in record["molecule_ids"]:
            for target_id in record["target_chembl_ids"]:
                key = f"{molecule_id}|{target_id}"
                activities.extend((activity_cache.get(key) or {}).get("activities", []))
        unique_types = sorted({activity.get("standard_type") for activity in activities if activity})
        if activities:
            recoverable_rows.append(record)
            row_type_counter.update(unique_types)
            for activity in activities:
                if activity.get("activity_id") is not None:
                    unique_activity_ids.add(str(activity["activity_id"]))
                if activity.get("standard_type"):
                    activity_records_by_type[activity["standard_type"]] += 1
        if any(activity.get("standard_type") in STRICT_BINDING_TYPES for activity in activities):
            strict_rows.append(record)

    def unique_field(records: list[dict[str, Any]], field: str) -> int:
        return len({record[field] for record in records if record.get(field)})

    method_counter = Counter(
        method_by_pdb.get(record["allosteric_pdb"], "UNKNOWN") for record in recoverable_rows
    )
    strict_method_counter = Counter(
        method_by_pdb.get(record["allosteric_pdb"], "UNKNOWN") for record in strict_rows
    )
    class_counter = Counter(record["modulator_class"] for record in recoverable_rows)
    strict_class_counter = Counter(record["modulator_class"] for record in strict_rows)

    chemcomp_ok = [value for value in chemcomp_cache.values() if value.get("status") == "ok"]
    chemcomp_with_inchikey = [value for value in chemcomp_ok if value.get("inchikey")]
    molecules_ok = [value for value in molecule_cache.values() if value.get("status") == "ok"]
    targets_ok = [value for value in target_cache.values() if value.get("status") == "ok"]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "asd_archive": display_path(args.asd_archive),
            "rcsb_cache": display_path(args.rcsb_cache),
            "cache_dir": display_path(args.cache_dir),
        },
        "query_scope": {
            "asd_rows": len(asd_rows),
            "unique_candidate_alias_tokens": len(aliases),
            "unique_alias_tokens_with_rcsb_chemcomp": len(chemcomp_ok),
            "unique_alias_tokens_with_inchikey": len(chemcomp_with_inchikey),
            "unique_inchikeys_from_asd_aliases": len(inchikeys),
            "unique_inchikeys_with_chembl_molecule": len(molecules_ok),
            "unique_uniprots_from_asd": len(uniprots),
            "unique_uniprots_with_chembl_target": len(targets_ok),
            "unique_chembl_molecule_target_pairs_queried": len(pairs),
        },
        "recoverable_any_activity": {
            "asd_rows": len(recoverable_rows),
            "unique_pdb_ids": unique_field(recoverable_rows, "allosteric_pdb"),
            "unique_target_ids": unique_field(recoverable_rows, "target_id"),
            "unique_target_modulator_keys": len(
                {
                    f"{record['target_id']}|{record['modulator_serial']}"
                    for record in recoverable_rows
                }
            ),
            "unique_chembl_activity_ids": len(unique_activity_ids),
            "row_counts_by_method": dict(method_counter.most_common()),
            "row_counts_by_modulator_class": dict(class_counter.most_common()),
            "row_counts_by_activity_type": dict(row_type_counter.most_common()),
            "activity_record_counts_by_type": dict(activity_records_by_type.most_common()),
        },
        "recoverable_strict_binding_kd_ki": {
            "asd_rows": len(strict_rows),
            "unique_pdb_ids": unique_field(strict_rows, "allosteric_pdb"),
            "unique_target_ids": unique_field(strict_rows, "target_id"),
            "unique_target_modulator_keys": len(
                {
                    f"{record['target_id']}|{record['modulator_serial']}"
                    for record in strict_rows
                }
            ),
            "row_counts_by_method": dict(strict_method_counter.most_common()),
            "row_counts_by_modulator_class": dict(strict_class_counter.most_common()),
        },
        "notes": [
            "Recoverable means at least one ChEMBL pChEMBL activity exists for an exact RCSB-ligand-InChIKey to ChEMBL-molecule and UniProt to ChEMBL-target pair.",
            "This is an exact ligand-target label, not necessarily an exact PDB-complex label or exact assay-condition match.",
            "Rows with peptide, chain-only, multi-component, or ion annotations often cannot be mapped through RCSB CCD InChIKey to ChEMBL molecules.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asd-archive", type=Path, default=DEFAULT_ASD_ARCHIVE)
    parser.add_argument("--rcsb-cache", type=Path, default=DEFAULT_RCSB_CACHE)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--online",
        action="store_true",
        help="Fetch missing RCSB and ChEMBL cache records.",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Use existing caches only; do not query RCSB or ChEMBL.",
    )
    args = parser.parse_args()
    if args.online and args.skip_fetch:
        raise ValueError("--online and --skip-fetch cannot be used together")
    fetch_missing = args.online and not args.skip_fetch

    asd_rows = read_asd_rows(args.asd_archive)
    method_by_pdb = {
        row.get("pdb_id", "").strip().upper(): row.get("methods", "").strip()
        for row in read_tsv(args.rcsb_cache)
        if row.get("pdb_id", "").strip()
    }

    aliases = {
        alias for row in asd_rows for alias in split_aliases(row.get("modulator_alias", ""))
    }
    uniprots = {
        uniprot for row in asd_rows for uniprot in split_uniprots(row.get("pdb_uniprot", ""))
    }

    chemcomp_path = args.cache_dir / "rcsb_chemcomp_by_alias.json"
    molecule_path = args.cache_dir / "chembl_molecule_by_inchikey.json"
    target_path = args.cache_dir / "chembl_target_by_uniprot.json"
    activity_path = args.cache_dir / "chembl_activity_by_molecule_target.json"

    if not fetch_missing:
        require_chembl_caches(args.cache_dir)

    chemcomp_cache = read_json(chemcomp_path, {})
    molecule_cache = read_json(molecule_path, {})
    target_cache = read_json(target_path, {})
    activity_cache = read_json(activity_path, {})

    if fetch_missing:
        chemcomp_cache = update_cache_parallel(
            chemcomp_cache, aliases, get_rcsb_chemcomp, chemcomp_path, args.workers, "RCSB CCD"
        )

    row_records, inchikeys, uniprots, pairs = build_row_records(
        asd_rows, chemcomp_cache, molecule_cache, target_cache
    )

    if fetch_missing:
        molecule_cache = update_cache_parallel(
            molecule_cache,
            inchikeys,
            get_chembl_molecules,
            molecule_path,
            args.workers,
            "ChEMBL molecules",
        )
        target_cache = update_cache_parallel(
            target_cache,
            uniprots,
            get_chembl_targets,
            target_path,
            args.workers,
            "ChEMBL targets",
        )

    row_records, inchikeys, uniprots, pairs = build_row_records(
        asd_rows, chemcomp_cache, molecule_cache, target_cache
    )

    if fetch_missing:
        pair_keys = [f"{molecule_id}|{target_id}" for molecule_id, target_id in pairs]

        def fetch_activity_pair(key: str) -> dict[str, Any]:
            molecule_id, target_id = key.split("|", 1)
            return get_chembl_activities(molecule_id, target_id)

        activity_cache = update_cache_parallel(
            activity_cache,
            pair_keys,
            fetch_activity_pair,
            activity_path,
            args.workers,
            "ChEMBL activities",
        )

    summary = summarize(
        asd_rows,
        row_records,
        activity_cache,
        method_by_pdb,
        args,
        aliases,
        inchikeys,
        uniprots,
        pairs,
        chemcomp_cache,
        molecule_cache,
        target_cache,
    )
    write_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
