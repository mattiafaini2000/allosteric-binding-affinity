#!/usr/bin/env python3
"""Recover ChEMBL/BindingDB activity labels for the complete ASD online set.

This script works on ASD's online browse endpoints rather than only the
downloaded allosteric-site complex archive. It crawls the full modulator table
and protein table, explodes modulator -> related protein links into
ligand-target interactions, and then checks whether ChEMBL or BindingDB has
activity data for the exact ASD ligand InChIKey and exact ASD UniProt target.

Row-level outputs are written under data/interim because they contain ASD row
data and third-party activity records. Local summaries contain aggregate counts only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import tarfile
import time
import urllib.parse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .discovery_chembl import (
    ACTIVITY_TYPES,
    STRICT_BINDING_TYPES,
    display_path,
    get_chembl_targets,
    read_json,
    update_cache_parallel,
    write_json,
)
from .verification_ligand_target import (
    DEFAULT_BINDINGDB_CUTOFF_NM,
    bindingdb_sort_key,
    get_bindingdb_ligands_by_uniprot,
    normalize_bindingdb_endpoint,
    parse_bindingdb_affinity,
    pk_from_nm,
    safe_float,
    smiles_to_inchikey_record,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO_ROOT / "data/interim/asd_online"
DEFAULT_CHEMBL_CACHE_DIR = REPO_ROOT / "data/interim/chembl"
DEFAULT_BINDINGDB_CACHE_DIR = REPO_ROOT / "data/interim/ligand_target"
DEFAULT_ASD_AS_ARCHIVE = REPO_ROOT / "data/raw/asd/archives/ASD_Release_202309_AS.tar.gz"
DEFAULT_LABELS_TSV = DEFAULT_CACHE_DIR / "asd_online_chembl_bindingdb_activity_labels.tsv"
DEFAULT_COMPOUNDS_TSV = DEFAULT_CACHE_DIR / "asd_online_compounds_with_activity.tsv"
DEFAULT_INTERACTIONS_TSV = DEFAULT_CACHE_DIR / "asd_online_interactions.tsv"
DEFAULT_SUMMARY_JSON = REPO_ROOT / "outputs/discovery/asd_online_activity_expansion_summary.json"
DEFAULT_SUMMARY_MD = REPO_ROOT / "outputs/discovery/asd_online_activity_expansion_summary.md"

ASD_BASE_URL = "https://mdl.shsmu.edu.cn/ASD"
ASD_MODULATORS_URL = f"{ASD_BASE_URL}/BrowseModulators"
ASD_PROTEINS_URL = f"{ASD_BASE_URL}/BrowseProteins"
CHEMBL_MOLECULE_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
CHEMBL_ACTIVITY_URL = "https://www.ebi.ac.uk/chembl/api/data/activity.json"

ENDPOINT_PRIORITY = {"Kd": 0, "Ki": 1, "IC50": 2, "EC50": 3, "AC50": 4}


def request_json_get(url: str, timeout: int = 45, attempts: int = 4) -> dict[str, Any] | None:
    import requests

    for attempt in range(attempts):
        try:
            response = requests.get(
                url,
                timeout=timeout,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "asd-online-activity-expansion/0.1",
                },
            )
            if response.status_code == 404:
                return None
            if response.status_code in {429, 500, 502, 503, 504}:
                time.sleep(2.0 * (attempt + 1))
                continue
            response.raise_for_status()
            return response.json()
        except Exception:
            if attempt == attempts - 1:
                return None
            time.sleep(2.0 * (attempt + 1))
    return None


def request_json_post(
    url: str,
    data: dict[str, Any],
    timeout: int = 60,
    attempts: int = 5,
) -> dict[str, Any] | None:
    import requests

    for attempt in range(attempts):
        try:
            response = requests.post(
                url,
                data=data,
                timeout=timeout,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "asd-online-activity-expansion/0.1",
                },
            )
            if response.status_code == 404:
                return None
            if response.status_code in {429, 500, 502, 503, 504}:
                time.sleep(2.0 * (attempt + 1))
                continue
            response.raise_for_status()
            return response.json()
        except Exception:
            if attempt == attempts - 1:
                return None
            time.sleep(2.0 * (attempt + 1))
    return None


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() in {"null", "none", "undefined"}:
        return ""
    return text.strip()


def read_as_complex_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    with tarfile.open(path, "r:gz") as tar:
        member = next(
            m for m in tar.getmembers() if Path(m.name).name == "ASD_Release_202309_AS.txt"
        )
        handle = tar.extractfile(member)
        if handle is None:
            return set()
        text = handle.read().decode("utf-8-sig")
    keys: set[tuple[str, str]] = set()
    for row in csv.DictReader(text.splitlines(), delimiter="\t"):
        target_id = clean_text(row.get("target_id"))
        modulator_serial = clean_text(row.get("modulator_serial"))
        if target_id and modulator_serial:
            keys.add((target_id, modulator_serial))
    return keys


def split_uniprots(value: str) -> list[str]:
    tokens = re.split(r";|,|\s+", value or "")
    return sorted({token.strip().upper() for token in tokens if token.strip()})


def first_uniprot(row: dict[str, Any]) -> str:
    for key in ("swissport_id", "trembl_id"):
        value = clean_text(row.get(key))
        if value:
            return value.upper()
    return ""


def crawl_asd_endpoint(
    url: str,
    cache_path: Path,
    page_size: int,
    label: str,
    max_records: int | None = None,
    refresh: bool = False,
    allow_fetch: bool = False,
) -> list[dict[str, Any]]:
    if refresh and not allow_fetch:
        raise ValueError("Refreshing ASD online rows requires --online")
    if cache_path.exists() and not refresh:
        cached = read_json(cache_path, {})
        rows = cached.get("rows") or []
        total_count = int(cached.get("total_count") or len(rows))
        requested_count = min(total_count, max_records) if max_records else total_count
        if rows and len(rows) >= requested_count:
            print(f"{label}: loaded {len(rows)} cached rows from {display_path(cache_path)}")
            return rows[:requested_count]
        if rows:
            print(
                f"{label}: cached rows incomplete ({len(rows)}/{requested_count}); "
                "continuing paged crawl"
            )

    page_cache_path = cache_path.with_suffix(".pages.json")
    page_cache = {} if refresh else read_json(page_cache_path, {})
    pages = page_cache.get("pages", {})
    total_count = int(page_cache.get("total_count") or 0)

    if not total_count:
        if not allow_fetch:
            raise FileNotFoundError(
                f"ASD {label} cache is missing or incomplete: {cache_path}. Use --online to crawl it."
            )
        first = request_json_post(url, {"start": 0, "limit": page_size, "page": 1})
        if not first:
            raise RuntimeError(f"Could not fetch first ASD {label} page from {url}")
        total_count = int(first.get("totalCount") or len(first.get("data") or []))
        pages["0"] = first.get("data") or []
        write_json(page_cache_path, {"total_count": total_count, "pages": pages})

    fetch_count = min(total_count, max_records) if max_records else total_count
    starts = list(range(0, fetch_count, page_size))
    for page_index, start in enumerate(starts, start=1):
        key = str(start)
        expected_page_len = min(page_size, fetch_count - start)
        if key in pages and len(pages.get(key) or []) >= expected_page_len:
            continue
        if not allow_fetch:
            raise RuntimeError(
                f"ASD {label} page cache lacks page starting at {start}; use --online to fetch it."
            )
        page = start // page_size + 1
        data = request_json_post(url, {"start": start, "limit": page_size, "page": page})
        if not data:
            raise RuntimeError(f"Could not fetch ASD {label} page start={start}")
        pages[key] = data.get("data") or []
        if page_index % 5 == 0:
            write_json(page_cache_path, {"total_count": total_count, "pages": pages})
            print(f"{label}: fetched {min(start + page_size, fetch_count)}/{fetch_count}")

    rows: list[dict[str, Any]] = []
    for start in starts:
        rows.extend(pages.get(str(start), []))
    rows = rows[:fetch_count]
    write_json(page_cache_path, {"total_count": total_count, "pages": pages})
    write_json(
        cache_path,
        {
            "source_url": url,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "total_count": total_count,
            "rows": rows,
        },
    )
    print(f"{label}: wrote {len(rows)} rows to {display_path(cache_path)}")
    return rows


def normalize_modulator(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "modulator_id": clean_text(row.get("modulator_id")),
        "db_serial": clean_text(row.get("db_serial")),
        "modulator_name": clean_text(row.get("modulator_name")),
        "modulator_class": clean_text(row.get("modulator_class")),
        "pubchem_id": clean_text(row.get("pubchem_id")),
        "cas_id": clean_text(row.get("cas_id")),
        "iupac": clean_text(row.get("iupac")),
        "smiles": clean_text(row.get("smiles")),
        "formula": clean_text(row.get("formula")),
        "endogenous": clean_text(row.get("endogenous")),
        "drug_phase": clean_text(row.get("drug_phase")),
        "related_proteins_count": clean_text(row.get("related_proteins_count")),
        "related_proteins": row.get("related_proteins") or [],
    }


def normalize_protein(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "mol_id": clean_text(row.get("mol_id")),
        "db_serial": clean_text(row.get("db_serial")),
        "domain": clean_text(row.get("domain")),
        "gene_name": clean_text(row.get("gene_name")),
        "mol_name": clean_text(row.get("mol_name")),
        "organism": clean_text(row.get("organism")),
        "swissport_id": clean_text(row.get("swissport_id")),
        "trembl_id": clean_text(row.get("trembl_id")),
        "uniprot": first_uniprot(row),
        "pdb_id": clean_text(row.get("pdb_id")),
        "allosteric_mechanism": clean_text(row.get("allosteric_mechanism")),
        "pubmed_id": clean_text(row.get("pubmed_id")),
        "ref_title": clean_text(row.get("ref_title")),
    }


def build_online_interactions(
    modulators: list[dict[str, Any]],
    proteins: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    protein_by_serial: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for protein in proteins:
        if protein.get("db_serial"):
            protein_by_serial[protein["db_serial"]].append(protein)
    interactions: list[dict[str, Any]] = []
    interaction_id = 0
    base_related_item_id = 0
    for modulator in modulators:
        related = modulator.get("related_proteins") or []
        if not isinstance(related, list):
            related = []
        for related_index, item in enumerate(related, start=1):
            if not isinstance(item, list) or len(item) < 4:
                continue
            feature = clean_text(item[0])
            related_name = clean_text(item[1])
            related_mol_id = clean_text(item[2])
            related_target_serial = clean_text(item[3])
            base_related_item_id += 1
            matching_proteins = protein_by_serial.get(related_target_serial, [])
            if not matching_proteins:
                matching_proteins = [{}]
            for protein in matching_proteins:
                interaction_id += 1
                interactions.append(
                    {
                        "asd_online_interaction_id": interaction_id,
                        "base_related_item_id": base_related_item_id,
                        "related_item_index_for_modulator": related_index,
                        "asd_modulator_id": modulator.get("modulator_id", ""),
                        "asd_modulator_serial": modulator.get("db_serial", ""),
                        "modulator_name": modulator.get("modulator_name", ""),
                        "modulator_class": modulator.get("modulator_class", ""),
                        "pubchem_id": modulator.get("pubchem_id", ""),
                        "cas_id": modulator.get("cas_id", ""),
                        "smiles": modulator.get("smiles", ""),
                        "formula": modulator.get("formula", ""),
                        "endogenous": modulator.get("endogenous", ""),
                        "drug_phase": modulator.get("drug_phase", ""),
                        "related_feature": feature,
                        "related_protein_name": related_name,
                        "related_mol_id": related_mol_id,
                        "related_target_serial": related_target_serial,
                        "target_serial_match_count": len(matching_proteins)
                        if protein
                        else 0,
                        "domain_expansion_status": "target_serial_matched"
                        if protein
                        else "target_serial_not_found",
                        "target_mol_id": protein.get("mol_id", ""),
                        "target_domain": protein.get("domain", ""),
                        "target_db_serial": protein.get("db_serial", ""),
                        "target_name": protein.get("mol_name", ""),
                        "target_gene": protein.get("gene_name", ""),
                        "organism": protein.get("organism", ""),
                        "uniprot": protein.get("uniprot", ""),
                        "swissport_id": protein.get("swissport_id", ""),
                        "trembl_id": protein.get("trembl_id", ""),
                        "pdb_id": protein.get("pdb_id", ""),
                        "target_pubmed_id": protein.get("pubmed_id", ""),
                    }
                )
    return interactions


def smiles_cache_for_modulators(
    modulators: list[dict[str, Any]],
    cache_path: Path,
) -> dict[str, Any]:
    cache = read_json(cache_path, {})
    smiles_values = sorted({row.get("smiles", "") for row in modulators if row.get("smiles")})
    missing = [smiles for smiles in smiles_values if smiles not in cache]
    if missing:
        print(f"ASD SMILES InChIKey: converting {len(missing)} missing structures")
        for idx, smiles in enumerate(missing, start=1):
            cache[smiles] = smiles_to_inchikey_record(smiles)
            if idx % 5000 == 0:
                write_json(cache_path, cache)
                print(f"ASD SMILES InChIKey: {idx}/{len(missing)} converted")
        write_json(cache_path, cache)
    else:
        print(f"ASD SMILES InChIKey: cache complete ({len(cache)} entries)")
    return cache


def add_ligand_identity(
    modulators: list[dict[str, Any]],
    interactions: list[dict[str, Any]],
    smiles_cache: dict[str, Any],
) -> None:
    modulator_by_id = {row.get("modulator_id", ""): row for row in modulators}
    for modulator in modulators:
        record = smiles_cache.get(modulator.get("smiles", "")) or {}
        modulator["ligand_inchikey"] = record.get("inchikey", "")
        modulator["canonical_smiles"] = record.get("canonical_smiles", "")
        modulator["ligand_identity_status"] = record.get("status", "missing")
    for interaction in interactions:
        modulator = modulator_by_id.get(interaction.get("asd_modulator_id", ""), {})
        interaction["ligand_inchikey"] = modulator.get("ligand_inchikey", "")
        interaction["canonical_smiles"] = modulator.get("canonical_smiles", "")
        interaction["ligand_identity_status"] = modulator.get("ligand_identity_status", "")


def add_as_complex_flags(
    interactions: list[dict[str, Any]],
    as_complex_keys: set[tuple[str, str]],
) -> None:
    for interaction in interactions:
        key = (
            clean_text(interaction.get("target_domain")),
            clean_text(interaction.get("asd_modulator_serial")),
        )
        interaction["has_as_complex_key"] = "1" if key in as_complex_keys else "0"


def fetch_chembl_molecule_batch(inchikeys: list[str]) -> dict[str, Any]:
    query = ",".join(inchikeys)
    url = (
        f"{CHEMBL_MOLECULE_URL}?molecule_structures__standard_inchi_key__in="
        f"{urllib.parse.quote(query, safe=',')}&limit=1000"
    )
    data = request_json_get(url, timeout=60)
    molecules = []
    for molecule in (data or {}).get("molecules", []):
        structures = molecule.get("molecule_structures") or {}
        hierarchy = molecule.get("molecule_hierarchy") or {}
        molecules.append(
            {
                "inchikey": structures.get("standard_inchi_key") or "",
                "molecule_chembl_id": molecule.get("molecule_chembl_id") or "",
                "parent_molecule_chembl_id": hierarchy.get("parent_chembl_id") or "",
                "pref_name": molecule.get("pref_name") or "",
                "molecule_type": molecule.get("molecule_type") or "",
            }
        )
    return {"status": "ok", "molecules": molecules}


def update_chembl_molecule_cache(
    inchikeys: Iterable[str],
    cache_path: Path,
    batch_size: int,
    workers: int,
    allow_fetch: bool = False,
) -> dict[str, Any]:
    cache = read_json(cache_path, {})
    missing = [key for key in sorted(set(inchikeys)) if key and key not in cache]
    if not missing:
        print(f"ChEMBL molecules: cache complete ({len(cache)} entries)")
        return cache
    if not allow_fetch:
        raise RuntimeError(
            f"ChEMBL molecule cache lacks {len(missing)} InChIKeys; use --online to fetch them."
        )

    batches = [missing[idx : idx + batch_size] for idx in range(0, len(missing), batch_size)]
    print(f"ChEMBL molecules: fetching {len(missing)} InChIKeys in {len(batches)} batches")
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_batch = {executor.submit(fetch_chembl_molecule_batch, batch): batch for batch in batches}
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"status": "error", "error": str(exc), "molecules": []}
            by_inchikey: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for molecule in result.get("molecules", []):
                if molecule.get("inchikey"):
                    by_inchikey[molecule["inchikey"]].append(molecule)
            for inchikey in batch:
                molecules = by_inchikey.get(inchikey, [])
                cache[inchikey] = {
                    "status": "ok" if molecules else "missing",
                    "molecules": molecules,
                }
            completed += 1
            if completed % 20 == 0:
                write_json(cache_path, cache)
                print(f"ChEMBL molecules: {completed}/{len(batches)} batches fetched")
    write_json(cache_path, cache)
    return cache


def chembl_molecule_ids_for_inchikey(inchikey: str, cache: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for molecule in (cache.get(inchikey) or {}).get("molecules", []):
        for key in ("molecule_chembl_id", "parent_molecule_chembl_id"):
            value = molecule.get(key)
            if value:
                ids.add(value)
    return ids


def chembl_target_ids_for_uniprot(uniprot: str, cache: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for target in (cache.get(uniprot) or {}).get("targets", []):
        value = target.get("target_chembl_id")
        if value:
            ids.add(value)
    return ids


def activity_sort_key(activity: dict[str, Any]) -> tuple[int, int, int, float, str]:
    endpoint = activity.get("standard_type") or ""
    relation = activity.get("standard_relation") or ""
    assay_type = activity.get("assay_type") or ""
    pchembl = safe_float(activity.get("pchembl_value"))
    return (
        ENDPOINT_PRIORITY.get(endpoint, 99),
        0 if relation == "=" else 1,
        0 if assay_type == "B" else 1,
        -(pchembl if pchembl is not None else -999.0),
        str(activity.get("activity_id") or ""),
    )


def dedupe_activities(activities: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    fallback = 0
    for activity in activities:
        activity_id = str(activity.get("activity_id") or "")
        if not activity_id:
            fallback += 1
            activity_id = f"no-id-{fallback}"
        by_id[activity_id] = activity
    return sorted(by_id.values(), key=activity_sort_key)


def pvalue_summary(activities: list[dict[str, Any]], selected_type: str) -> dict[str, str]:
    values = []
    for activity in activities:
        if selected_type and activity.get("standard_type") != selected_type:
            continue
        value = safe_float(activity.get("pchembl_value"))
        if value is not None:
            values.append(value)
    if not values:
        return {"min": "", "median": "", "max": ""}
    return {
        "min": f"{min(values):.4g}",
        "median": f"{statistics.median(values):.4g}",
        "max": f"{max(values):.4g}",
    }


def fetch_chembl_activity_batch(target_id: str, molecule_ids: list[str]) -> dict[str, Any]:
    activities = []
    offset = 0
    limit = 1000
    standard_types = ",".join(ACTIVITY_TYPES)
    molecule_query = ",".join(molecule_ids)
    while True:
        url = (
            f"{CHEMBL_ACTIVITY_URL}?target_chembl_id={urllib.parse.quote(target_id)}"
            f"&molecule_chembl_id__in={urllib.parse.quote(molecule_query, safe=',')}"
            f"&standard_type__in={standard_types}"
            "&pchembl_value__isnull=false"
            f"&limit={limit}&offset={offset}"
        )
        data = request_json_get(url, timeout=90)
        if not data:
            return {"status": "error", "activities": activities, "offset": offset}
        for activity in data.get("activities", []):
            standard_type = activity.get("standard_type")
            pchembl = activity.get("pchembl_value")
            if standard_type not in ACTIVITY_TYPES or not pchembl:
                continue
            activities.append(
                {
                    "activity_id": activity.get("activity_id") or "",
                    "standard_type": standard_type,
                    "standard_value": activity.get("standard_value") or "",
                    "standard_units": activity.get("standard_units") or "",
                    "pchembl_value": pchembl,
                    "standard_relation": activity.get("standard_relation") or "",
                    "assay_chembl_id": activity.get("assay_chembl_id") or "",
                    "assay_type": activity.get("assay_type") or "",
                    "document_chembl_id": activity.get("document_chembl_id") or "",
                    "document_year": activity.get("document_year") or "",
                    "target_chembl_id": activity.get("target_chembl_id") or "",
                    "molecule_chembl_id": activity.get("molecule_chembl_id") or "",
                }
            )
        page_meta = data.get("page_meta") or {}
        if not page_meta.get("next"):
            break
        offset += limit
    return {"status": "ok" if activities else "missing", "activities": activities}


def batch_key(target_id: str, molecule_ids: list[str]) -> str:
    joined = ",".join(molecule_ids)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]
    return f"{target_id}|{digest}"


def update_chembl_activity_cache(
    target_to_molecules: dict[str, set[str]],
    cache_path: Path,
    batch_size: int,
    workers: int,
    allow_fetch: bool = False,
) -> dict[str, Any]:
    cache = read_json(cache_path, {})
    tasks: list[tuple[str, list[str], str]] = []
    for target_id, molecule_ids in sorted(target_to_molecules.items()):
        sorted_ids = sorted(molecule_ids)
        for idx in range(0, len(sorted_ids), batch_size):
            batch = sorted_ids[idx : idx + batch_size]
            key = batch_key(target_id, batch)
            if key not in cache:
                tasks.append((target_id, batch, key))
    if not tasks:
        print(f"ChEMBL activities: cache complete ({len(cache)} batches)")
        return cache
    if not allow_fetch:
        raise RuntimeError(
            f"ChEMBL activity cache lacks {len(tasks)} target/molecule batches; use --online to fetch them."
        )

    print(f"ChEMBL activities: fetching {len(tasks)} target/molecule batches")
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_task = {
            executor.submit(fetch_chembl_activity_batch, target_id, batch): (target_id, batch, key)
            for target_id, batch, key in tasks
        }
        for future in as_completed(future_to_task):
            target_id, batch, key = future_to_task[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"status": "error", "error": str(exc), "activities": []}
            result["target_chembl_id"] = target_id
            result["molecule_chembl_ids"] = batch
            cache[key] = result
            completed += 1
            if completed % 20 == 0:
                write_json(cache_path, cache)
                print(f"ChEMBL activities: {completed}/{len(tasks)} batches fetched")
    write_json(cache_path, cache)
    return cache


def build_chembl_activity_index(activity_cache: dict[str, Any]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for batch in activity_cache.values():
        for activity in batch.get("activities", []):
            target_id = activity.get("target_chembl_id") or batch.get("target_chembl_id") or ""
            molecule_id = activity.get("molecule_chembl_id") or ""
            if target_id and molecule_id:
                by_pair[(molecule_id, target_id)].append(activity)
    return by_pair


def select_activities(activities: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    deduped = dedupe_activities(activities)
    if not deduped:
        return [], {}
    selected = deduped[0]
    return deduped, selected


def endpoint_group(endpoint_types: set[str]) -> str:
    if endpoint_types & STRICT_BINDING_TYPES:
        return "strict_binding_kd_ki"
    if endpoint_types:
        return "potency_or_functional"
    return ""


def label_base(interaction: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "source": source,
        "label_scope": "asd_online_exact_ligand_target",
        "asd_online_interaction_id": interaction.get("asd_online_interaction_id", ""),
        "base_related_item_id": interaction.get("base_related_item_id", ""),
        "related_item_index_for_modulator": interaction.get("related_item_index_for_modulator", ""),
        "asd_modulator_id": interaction.get("asd_modulator_id", ""),
        "asd_modulator_serial": interaction.get("asd_modulator_serial", ""),
        "modulator_name": interaction.get("modulator_name", ""),
        "modulator_class": interaction.get("modulator_class", ""),
        "pubchem_id": interaction.get("pubchem_id", ""),
        "cas_id": interaction.get("cas_id", ""),
        "smiles": interaction.get("smiles", ""),
        "canonical_smiles": interaction.get("canonical_smiles", ""),
        "ligand_inchikey": interaction.get("ligand_inchikey", ""),
        "ligand_identity_status": interaction.get("ligand_identity_status", ""),
        "related_feature": interaction.get("related_feature", ""),
        "related_protein_name": interaction.get("related_protein_name", ""),
        "related_mol_id": interaction.get("related_mol_id", ""),
        "related_target_serial": interaction.get("related_target_serial", ""),
        "target_serial_match_count": interaction.get("target_serial_match_count", ""),
        "domain_expansion_status": interaction.get("domain_expansion_status", ""),
        "has_as_complex_key": interaction.get("has_as_complex_key", ""),
        "target_mol_id": interaction.get("target_mol_id", ""),
        "target_domain": interaction.get("target_domain", ""),
        "target_db_serial": interaction.get("target_db_serial", ""),
        "target_name": interaction.get("target_name", ""),
        "target_gene": interaction.get("target_gene", ""),
        "organism": interaction.get("organism", ""),
        "uniprot": interaction.get("uniprot", ""),
        "pdb_id": interaction.get("pdb_id", ""),
    }


def build_chembl_labels(
    interactions: list[dict[str, Any]],
    molecule_cache: dict[str, Any],
    target_cache: dict[str, Any],
    activity_index: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    labels = []
    for interaction in interactions:
        inchikey = interaction.get("ligand_inchikey", "")
        uniprot = interaction.get("uniprot", "")
        if not inchikey or not uniprot:
            continue
        molecule_ids = chembl_molecule_ids_for_inchikey(inchikey, molecule_cache)
        target_ids = chembl_target_ids_for_uniprot(uniprot, target_cache)
        activities = []
        for molecule_id in molecule_ids:
            for target_id in target_ids:
                activities.extend(activity_index.get((molecule_id, target_id), []))
        activities, selected = select_activities(activities)
        if not selected:
            continue
        endpoint_types = sorted({activity.get("standard_type", "") for activity in activities if activity})
        binding_activities = [
            activity for activity in activities if activity.get("standard_type") in STRICT_BINDING_TYPES
        ]
        selected_type = selected.get("standard_type", "")
        summary = pvalue_summary(activities, selected_type)
        row = label_base(interaction, "chembl")
        row.update(
            {
                "tier": "tier_b_online",
                "endpoint_group": endpoint_group(set(endpoint_types)),
                "activity_count": len(activities),
                "binding_activity_count": len(binding_activities),
                "endpoint_types": ";".join(endpoint_types),
                "selected_endpoint_type": selected_type,
                "selected_pchembl_or_pk": selected.get("pchembl_value", ""),
                "selected_pchembl_min_same_endpoint": summary["min"],
                "selected_pchembl_median_same_endpoint": summary["median"],
                "selected_pchembl_max_same_endpoint": summary["max"],
                "selected_standard_relation": selected.get("standard_relation", ""),
                "selected_standard_value": selected.get("standard_value", ""),
                "selected_standard_units": selected.get("standard_units", ""),
                "selected_activity_id": selected.get("activity_id", ""),
                "selected_assay_id": selected.get("assay_chembl_id", ""),
                "selected_document_id": selected.get("document_chembl_id", ""),
                "selected_document_year": selected.get("document_year", ""),
                "source_molecule_ids": ";".join(sorted(molecule_ids)),
                "source_target_ids": ";".join(sorted(target_ids)),
                "activity_ids": ";".join(
                    str(activity.get("activity_id")) for activity in activities if activity.get("activity_id")
                ),
                "verification_basis": (
                    "ASD online SMILES converted to InChIKey, ASD online target resolved to UniProt, "
                    "ChEMBL molecule and target matched exactly, and ChEMBL returned selected endpoint activity."
                ),
            }
        )
        labels.append(row)
    return labels


def update_bindingdb_cache(
    uniprots: Iterable[str],
    cache_path: Path,
    cutoff_nm: int,
    workers: int,
    allow_fetch: bool = False,
) -> dict[str, Any]:
    cache = read_json(cache_path, {})
    missing = [key for key in sorted(set(uniprots)) if key and key not in cache]
    if not missing:
        print(f"BindingDB targets: cache complete ({len(cache)} entries)")
        return cache
    if not allow_fetch:
        raise RuntimeError(
            f"BindingDB target cache lacks {len(missing)} UniProt queries; use --online to fetch them."
        )
    print(f"BindingDB targets: fetching {len(missing)} UniProt queries")
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_key = {
            executor.submit(get_bindingdb_ligands_by_uniprot, key, cutoff_nm): key for key in missing
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                cache[key] = future.result()
            except Exception as exc:
                cache[key] = {"status": "error", "error": str(exc), "affinities": []}
            completed += 1
            if completed % 20 == 0:
                write_json(cache_path, cache)
                print(f"BindingDB targets: {completed}/{len(missing)} fetched")
    write_json(cache_path, cache)
    return cache


def update_bindingdb_smiles_cache(
    bindingdb_cache: dict[str, Any],
    cache_path: Path,
) -> dict[str, Any]:
    cache = read_json(cache_path, {})
    smiles_values = sorted(
        {
            record.get("smiles", "")
            for target in bindingdb_cache.values()
            for record in target.get("affinities", [])
            if record.get("smiles")
        }
    )
    missing = [smiles for smiles in smiles_values if smiles not in cache]
    if not missing:
        print(f"BindingDB SMILES InChIKey: cache complete ({len(cache)} entries)")
        return cache
    print(f"BindingDB SMILES InChIKey: converting {len(missing)} missing structures")
    for idx, smiles in enumerate(missing, start=1):
        cache[smiles] = smiles_to_inchikey_record(smiles)
        if idx % 10000 == 0:
            write_json(cache_path, cache)
            print(f"BindingDB SMILES InChIKey: {idx}/{len(missing)} converted")
    write_json(cache_path, cache)
    return cache


def build_bindingdb_activity_index(
    bindingdb_cache: dict[str, Any],
    smiles_cache: dict[str, Any],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    by_pair: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for uniprot, target in bindingdb_cache.items():
        for record in target.get("affinities", []):
            endpoint = normalize_bindingdb_endpoint(record.get("affinity_type", ""))
            if endpoint not in ACTIVITY_TYPES:
                continue
            relation, value_nm = parse_bindingdb_affinity(record.get("affinity", ""))
            if value_nm is None:
                continue
            pk_value = pk_from_nm(value_nm)
            if pk_value is None:
                continue
            smiles = record.get("smiles", "")
            inchikey = (smiles_cache.get(smiles) or {}).get("inchikey", "")
            if not inchikey:
                continue
            monomer_id = record.get("monomerid", "")
            activity_id = f"BDBM{monomer_id}:{uniprot}:{endpoint}:{relation}{value_nm:g}"
            by_pair[(inchikey, uniprot)][activity_id] = {
                "activity_id": activity_id,
                "standard_type": endpoint,
                "standard_value": f"{value_nm:g}",
                "standard_units": "nM",
                "pchembl_value": f"{pk_value:.6g}",
                "standard_relation": relation,
                "assay_chembl_id": "",
                "assay_type": "",
                "document_chembl_id": "",
                "document_year": "",
                "target_chembl_id": uniprot,
                "molecule_chembl_id": f"BDBM{monomer_id}" if monomer_id else "",
                "bindingdb_smiles_inchikey": inchikey,
            }
    return {key: sorted(value.values(), key=bindingdb_sort_key) for key, value in by_pair.items()}


def build_bindingdb_labels(
    interactions: list[dict[str, Any]],
    activity_index: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    labels = []
    for interaction in interactions:
        inchikey = interaction.get("ligand_inchikey", "")
        uniprot = interaction.get("uniprot", "")
        if not inchikey or not uniprot:
            continue
        activities = activity_index.get((inchikey, uniprot), [])
        activities, selected = select_activities(activities)
        if not selected:
            continue
        endpoint_types = sorted({activity.get("standard_type", "") for activity in activities if activity})
        binding_activities = [
            activity for activity in activities if activity.get("standard_type") in STRICT_BINDING_TYPES
        ]
        selected_type = selected.get("standard_type", "")
        summary = pvalue_summary(activities, selected_type)
        row = label_base(interaction, "bindingdb")
        row.update(
            {
                "tier": "tier_b_online",
                "endpoint_group": endpoint_group(set(endpoint_types)),
                "activity_count": len(activities),
                "binding_activity_count": len(binding_activities),
                "endpoint_types": ";".join(endpoint_types),
                "selected_endpoint_type": selected_type,
                "selected_pchembl_or_pk": selected.get("pchembl_value", ""),
                "selected_pchembl_min_same_endpoint": summary["min"],
                "selected_pchembl_median_same_endpoint": summary["median"],
                "selected_pchembl_max_same_endpoint": summary["max"],
                "selected_standard_relation": selected.get("standard_relation", ""),
                "selected_standard_value": selected.get("standard_value", ""),
                "selected_standard_units": selected.get("standard_units", ""),
                "selected_activity_id": selected.get("activity_id", ""),
                "selected_assay_id": "",
                "selected_document_id": "",
                "selected_document_year": "",
                "source_molecule_ids": ";".join(
                    sorted({activity.get("molecule_chembl_id", "") for activity in activities})
                ),
                "source_target_ids": uniprot,
                "activity_ids": ";".join(
                    str(activity.get("activity_id")) for activity in activities if activity.get("activity_id")
                ),
                "verification_basis": (
                    "ASD online SMILES converted to InChIKey, ASD online target resolved to UniProt, "
                    "BindingDB was queried by that UniProt, and a returned BindingDB ligand SMILES "
                    "converted to the same ASD ligand InChIKey."
                ),
            }
        )
        labels.append(row)
    return labels


def build_compound_rows(
    modulators: list[dict[str, Any]],
    interactions: list[dict[str, Any]],
    labels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    interactions_by_modulator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for interaction in interactions:
        interactions_by_modulator[interaction.get("asd_modulator_id", "")].append(interaction)

    labels_by_modulator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for label in labels:
        labels_by_modulator[label.get("asd_modulator_id", "")].append(label)

    rows = []
    for modulator in modulators:
        modulator_id = modulator.get("modulator_id", "")
        mod_labels = labels_by_modulator.get(modulator_id, [])
        if not mod_labels:
            continue
        mod_interactions = interactions_by_modulator.get(modulator_id, [])
        pvalues = [
            value
            for value in (safe_float(row.get("selected_pchembl_or_pk")) for row in mod_labels)
            if value is not None
        ]
        endpoint_types = sorted(
            {
                endpoint
                for row in mod_labels
                for endpoint in (row.get("endpoint_types", "").split(";") if row.get("endpoint_types") else [])
                if endpoint
            }
        )
        source_values = sorted({row.get("source", "") for row in mod_labels if row.get("source")})
        strict_count = sum(row.get("endpoint_group") == "strict_binding_kd_ki" for row in mod_labels)
        rows.append(
            {
                "asd_modulator_id": modulator_id,
                "asd_modulator_serial": modulator.get("db_serial", ""),
                "modulator_name": modulator.get("modulator_name", ""),
                "modulator_class": modulator.get("modulator_class", ""),
                "pubchem_id": modulator.get("pubchem_id", ""),
                "cas_id": modulator.get("cas_id", ""),
                "smiles": modulator.get("smiles", ""),
                "canonical_smiles": modulator.get("canonical_smiles", ""),
                "ligand_inchikey": modulator.get("ligand_inchikey", ""),
                "sources": ";".join(source_values),
                "endpoint_types": ";".join(endpoint_types),
                "label_source_rows": len(mod_labels),
                "labeled_asd_online_interactions": len(
                    {row.get("asd_online_interaction_id", "") for row in mod_labels}
                ),
                "total_asd_online_interactions": len(mod_interactions),
                "labeled_uniprots": len({row.get("uniprot", "") for row in mod_labels if row.get("uniprot")}),
                "labeled_target_domains": len(
                    {row.get("target_domain", "") for row in mod_labels if row.get("target_domain")}
                ),
                "strict_binding_label_rows": strict_count,
                "potency_or_functional_label_rows": len(mod_labels) - strict_count,
                "best_selected_pchembl_or_pk": f"{max(pvalues):.4g}" if pvalues else "",
                "median_selected_pchembl_or_pk": f"{statistics.median(pvalues):.4g}" if pvalues else "",
            }
        )
    return sorted(rows, key=lambda row: (-int(row["label_source_rows"]), row["asd_modulator_serial"]))


def count_unique(rows: Iterable[dict[str, Any]], field: str) -> int:
    return len({str(row.get(field, "")).strip() for row in rows if str(row.get(field, "")).strip()})


def count_unique_pairs(rows: Iterable[dict[str, Any]], fields: tuple[str, str]) -> int:
    values = set()
    for row in rows:
        first = str(row.get(fields[0], "")).strip()
        second = str(row.get(fields[1], "")).strip()
        if first and second:
            values.add((first, second))
    return len(values)


def subset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strict_rows = [row for row in rows if row.get("endpoint_group") == "strict_binding_kd_ki"]
    return {
        "label_source_rows": len(rows),
        "unique_domain_modulator_keys": count_unique_pairs(
            rows, ("target_domain", "asd_modulator_serial")
        ),
        "unique_compounds": count_unique(rows, "asd_modulator_serial"),
        "unique_target_domains": count_unique(rows, "target_domain"),
        "strict_binding_label_source_rows": len(strict_rows),
        "strict_binding_unique_domain_modulator_keys": count_unique_pairs(
            strict_rows, ("target_domain", "asd_modulator_serial")
        ),
        "strict_binding_unique_compounds": count_unique(strict_rows, "asd_modulator_serial"),
        "endpoint_counts": dict(Counter(row.get("selected_endpoint_type", "") for row in rows)),
    }


def source_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strict_rows = [row for row in rows if row.get("endpoint_group") == "strict_binding_kd_ki"]
    with_as_complex = [row for row in rows if row.get("has_as_complex_key") == "1"]
    without_as_complex = [row for row in rows if row.get("has_as_complex_key") != "1"]
    return {
        "label_source_rows": len(rows),
        "unique_compounds": count_unique(rows, "asd_modulator_id"),
        "unique_interactions": count_unique(rows, "asd_online_interaction_id"),
        "unique_domain_modulator_keys": count_unique_pairs(
            rows, ("target_domain", "asd_modulator_serial")
        ),
        "unique_uniprots": count_unique(rows, "uniprot"),
        "strict_binding_label_source_rows": len(strict_rows),
        "strict_binding_unique_compounds": count_unique(strict_rows, "asd_modulator_id"),
        "strict_binding_unique_interactions": count_unique(strict_rows, "asd_online_interaction_id"),
        "strict_binding_unique_domain_modulator_keys": count_unique_pairs(
            strict_rows, ("target_domain", "asd_modulator_serial")
        ),
        "with_as_complex_key": subset_summary(with_as_complex),
        "without_as_complex_key": subset_summary(without_as_complex),
        "endpoint_counts": dict(Counter(row.get("selected_endpoint_type", "") for row in rows)),
    }


def build_summary(
    modulators: list[dict[str, Any]],
    proteins: list[dict[str, Any]],
    interactions: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    compound_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    cache_stats: dict[str, Any],
) -> dict[str, Any]:
    by_source = {source: [row for row in labels if row.get("source") == source] for source in sorted({row.get("source", "") for row in labels})}
    strict_labels = [row for row in labels if row.get("endpoint_group") == "strict_binding_kd_ki"]
    source_sets_by_compound: dict[str, set[str]] = defaultdict(set)
    source_sets_by_interaction: dict[str, set[str]] = defaultdict(set)
    for label in labels:
        source_sets_by_compound[label.get("asd_modulator_id", "")].add(label.get("source", ""))
        source_sets_by_interaction[str(label.get("asd_online_interaction_id", ""))].add(label.get("source", ""))
    labels_with_as_complex = [row for row in labels if row.get("has_as_complex_key") == "1"]
    labels_without_as_complex = [row for row in labels if row.get("has_as_complex_key") != "1"]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "asd_modulators_url": ASD_MODULATORS_URL,
            "asd_proteins_url": ASD_PROTEINS_URL,
            "asd_as_archive": display_path(args.asd_as_archive),
            "bindingdb_cutoff_nm": args.bindingdb_cutoff_nm,
            "activity_types": list(ACTIVITY_TYPES),
            "strict_binding_types": sorted(STRICT_BINDING_TYPES),
            "max_modulators": args.max_modulators,
        },
        "outputs": {
            "interaction_labels_tsv": display_path(args.labels_tsv),
            "compound_labels_tsv": display_path(args.compounds_tsv),
            "interactions_tsv": display_path(args.interactions_tsv),
        },
        "online_asd": {
            "modulators": len(modulators),
            "proteins": len(proteins),
            "base_related_items": count_unique(interactions, "base_related_item_id"),
            "domain_expanded_interactions": len(interactions),
            "domain_expanded_rows_with_target_serial_match": sum(
                row.get("domain_expansion_status") == "target_serial_matched"
                for row in interactions
            ),
            "base_related_items_without_target_serial_match": count_unique(
                [
                    row
                    for row in interactions
                    if row.get("domain_expansion_status") == "target_serial_not_found"
                ],
                "base_related_item_id",
            ),
            "base_related_items_with_multiple_domain_matches": count_unique(
                [
                    row
                    for row in interactions
                    if int(str(row.get("target_serial_match_count") or "0")) > 1
                ],
                "base_related_item_id",
            ),
            "modulators_with_smiles": sum(1 for row in modulators if row.get("smiles")),
            "modulators_with_inchikey": sum(1 for row in modulators if row.get("ligand_inchikey")),
            "interactions_with_uniprot": sum(1 for row in interactions if row.get("uniprot")),
            "unique_uniprots": count_unique(interactions, "uniprot"),
            "domain_expanded_rows_with_as_complex_key": sum(
                row.get("has_as_complex_key") == "1" for row in interactions
            ),
            "domain_expanded_rows_without_as_complex_key": sum(
                row.get("has_as_complex_key") != "1" for row in interactions
            ),
        },
        "combined_labels": {
            "label_source_rows": len(labels),
            "unique_compounds": count_unique(labels, "asd_modulator_id"),
            "unique_interactions": count_unique(labels, "asd_online_interaction_id"),
            "unique_uniprots": count_unique(labels, "uniprot"),
            "compound_rows": len(compound_rows),
            "strict_binding_label_source_rows": len(strict_labels),
            "strict_binding_unique_compounds": count_unique(strict_labels, "asd_modulator_id"),
            "strict_binding_unique_interactions": count_unique(strict_labels, "asd_online_interaction_id"),
            "source_overlap_by_compound": dict(
                Counter("+".join(sorted(sources)) for sources in source_sets_by_compound.values())
            ),
            "source_overlap_by_interaction": dict(
                Counter("+".join(sorted(sources)) for sources in source_sets_by_interaction.values())
            ),
            "endpoint_counts": dict(Counter(row.get("selected_endpoint_type", "") for row in labels)),
        },
        "as_complex_overlap": {
            "with_as_complex_key": subset_summary(labels_with_as_complex),
            "without_as_complex_key": subset_summary(labels_without_as_complex),
        },
        "sources": {source: source_summary(rows) for source, rows in by_source.items()},
        "cache_stats": cache_stats,
        "notes": [
            "Labels require exact ASD online ligand InChIKey and exact ASD online UniProt target.",
            "ASD online related-protein items are expanded through ASD protein db_serial/domain, not the numeric related-protein field.",
            "ChEMBL labels require exact ChEMBL molecule and target mapping and pChEMBL endpoints.",
            "BindingDB labels require a target-side UniProt query and exact returned-ligand InChIKey match.",
            "This is ligand-target activity evidence, not exact experimental complex or pose evidence.",
            "Row-level TSVs stay under data/interim and are not versioned.",
        ],
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ASD Online Activity Expansion",
        "",
        "## Question",
        "",
        (
            "For the complete ASD online modulator set, how many ASD compounds/interactions "
            "have exact ligand-target activity labels recoverable from ChEMBL or BindingDB?"
        ),
        "",
        "## Inputs",
        "",
        f"- ASD online modulators endpoint: `{ASD_MODULATORS_URL}`",
        f"- ASD online proteins endpoint: `{ASD_PROTEINS_URL}`",
        "- ChEMBL molecule, target, and activity APIs.",
        "- BindingDB `getLigandsByUniprot` REST endpoint.",
        "",
        "## ASD Online Universe",
        "",
    ]
    for key, value in summary["online_asd"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Recovered Labels", ""])
    for key, value in summary["combined_labels"].items():
        if isinstance(value, dict):
            continue
        lines.append(f"- {key}: {value}")
    lines.extend(["", "Endpoint counts over selected labels:", ""])
    for endpoint, count in summary["combined_labels"]["endpoint_counts"].items():
        lines.append(f"- `{endpoint}`: {count}")
    lines.extend(["", "Source overlap by compound:", ""])
    for overlap, count in sorted(summary["combined_labels"]["source_overlap_by_compound"].items()):
        lines.append(f"- `{overlap}`: {count}")
    lines.extend(["", "## AS Complex Overlap", ""])
    for label, data in (
        ("With AS complex key", summary["as_complex_overlap"]["with_as_complex_key"]),
        ("Without AS complex key", summary["as_complex_overlap"]["without_as_complex_key"]),
    ):
        lines.append(f"### {label}")
        lines.append("")
        for key, value in data.items():
            if isinstance(value, dict):
                continue
            lines.append(f"- {key}: {value}")
        lines.append("")
        lines.append("Selected endpoint counts:")
        lines.append("")
        for endpoint, count in data["endpoint_counts"].items():
            lines.append(f"- `{endpoint}`: {count}")
        lines.append("")
    lines.extend(["", "## Source Summaries", ""])
    for source, source_data in summary["sources"].items():
        lines.append(f"### {source}")
        lines.append("")
        for key, value in source_data.items():
            if isinstance(value, dict):
                continue
            lines.append(f"- {key}: {value}")
        lines.append("")
        lines.append("Selected endpoint counts:")
        lines.append("")
        for endpoint, count in source_data["endpoint_counts"].items():
            lines.append(f"- `{endpoint}`: {count}")
        lines.append("")
    lines.extend(
        [
            "## Outputs",
            "",
            f"- Interaction-level labels: `{summary['outputs']['interaction_labels_tsv']}`",
            f"- Compound-level labels: `{summary['outputs']['compound_labels_tsv']}`",
            f"- Exploded ASD online interactions: `{summary['outputs']['interactions_tsv']}`",
            "",
            "## Interpretation",
            "",
            (
                "The recovered rows are exact ligand-target activity labels for ASD online "
                "modulator-protein interactions. They are suitable as future predicted-pose "
                "SBALIGN/DiffDock label candidates, but they are not exact crystallographic "
                "complex labels unless independently linked to a resolved structure."
            ),
            "",
            "## Notes",
            "",
        ]
    )
    for note in summary["notes"]:
        lines.append(f"- {note}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


LABEL_FIELDNAMES = [
    "source",
    "tier",
    "label_scope",
    "endpoint_group",
    "asd_online_interaction_id",
    "base_related_item_id",
    "related_item_index_for_modulator",
    "asd_modulator_id",
    "asd_modulator_serial",
    "modulator_name",
    "modulator_class",
    "pubchem_id",
    "cas_id",
    "smiles",
    "canonical_smiles",
    "ligand_inchikey",
    "ligand_identity_status",
    "related_feature",
    "related_protein_name",
    "related_mol_id",
    "related_target_serial",
    "target_serial_match_count",
    "domain_expansion_status",
    "has_as_complex_key",
    "target_mol_id",
    "target_domain",
    "target_db_serial",
    "target_name",
    "target_gene",
    "organism",
    "uniprot",
    "pdb_id",
    "activity_count",
    "binding_activity_count",
    "endpoint_types",
    "selected_endpoint_type",
    "selected_pchembl_or_pk",
    "selected_pchembl_min_same_endpoint",
    "selected_pchembl_median_same_endpoint",
    "selected_pchembl_max_same_endpoint",
    "selected_standard_relation",
    "selected_standard_value",
    "selected_standard_units",
    "selected_activity_id",
    "selected_assay_id",
    "selected_document_id",
    "selected_document_year",
    "source_molecule_ids",
    "source_target_ids",
    "activity_ids",
    "verification_basis",
]

COMPOUND_FIELDNAMES = [
    "asd_modulator_id",
    "asd_modulator_serial",
    "modulator_name",
    "modulator_class",
    "pubchem_id",
    "cas_id",
    "smiles",
    "canonical_smiles",
    "ligand_inchikey",
    "sources",
    "endpoint_types",
    "label_source_rows",
    "labeled_asd_online_interactions",
    "total_asd_online_interactions",
    "labeled_uniprots",
    "labeled_target_domains",
    "strict_binding_label_rows",
    "potency_or_functional_label_rows",
    "best_selected_pchembl_or_pk",
    "median_selected_pchembl_or_pk",
]

INTERACTION_FIELDNAMES = [
    "asd_online_interaction_id",
    "base_related_item_id",
    "related_item_index_for_modulator",
    "asd_modulator_id",
    "asd_modulator_serial",
    "modulator_name",
    "modulator_class",
    "pubchem_id",
    "cas_id",
    "smiles",
    "canonical_smiles",
    "ligand_inchikey",
    "ligand_identity_status",
    "related_feature",
    "related_protein_name",
    "related_mol_id",
    "related_target_serial",
    "target_serial_match_count",
    "domain_expansion_status",
    "has_as_complex_key",
    "target_mol_id",
    "target_domain",
    "target_db_serial",
    "target_name",
    "target_gene",
    "organism",
    "uniprot",
    "swissport_id",
    "trembl_id",
    "pdb_id",
    "target_pubmed_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--chembl-cache-dir", type=Path, default=DEFAULT_CHEMBL_CACHE_DIR)
    parser.add_argument("--bindingdb-cache-dir", type=Path, default=DEFAULT_BINDINGDB_CACHE_DIR)
    parser.add_argument("--asd-as-archive", type=Path, default=DEFAULT_ASD_AS_ARCHIVE)
    parser.add_argument("--labels-tsv", type=Path, default=DEFAULT_LABELS_TSV)
    parser.add_argument("--compounds-tsv", type=Path, default=DEFAULT_COMPOUNDS_TSV)
    parser.add_argument("--interactions-tsv", type=Path, default=DEFAULT_INTERACTIONS_TSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--max-modulators", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bindingdb-workers", type=int, default=4)
    parser.add_argument("--chembl-molecule-batch-size", type=int, default=100)
    parser.add_argument("--chembl-activity-batch-size", type=int, default=80)
    parser.add_argument("--bindingdb-cutoff-nm", type=int, default=DEFAULT_BINDINGDB_CUTOFF_NM)
    parser.add_argument("--refresh-asd", action="store_true", help="Recrawl ASD online endpoint caches.")
    parser.add_argument(
        "--online",
        action="store_true",
        help="Fetch missing ASD, ChEMBL, and BindingDB records.",
    )
    parser.add_argument("--skip-chembl", action="store_true")
    parser.add_argument("--skip-bindingdb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.refresh_asd and not args.online:
        raise ValueError("--refresh-asd requires --online")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.chembl_cache_dir.mkdir(parents=True, exist_ok=True)
    args.bindingdb_cache_dir.mkdir(parents=True, exist_ok=True)

    modulators_raw = crawl_asd_endpoint(
        ASD_MODULATORS_URL,
        args.cache_dir / "asd_online_modulators.json",
        args.page_size,
        "ASD online modulators",
        args.max_modulators,
        args.refresh_asd,
        args.online,
    )
    proteins_raw = crawl_asd_endpoint(
        ASD_PROTEINS_URL,
        args.cache_dir / "asd_online_proteins.json",
        3000,
        "ASD online proteins",
        None,
        args.refresh_asd,
        args.online,
    )
    modulators = [normalize_modulator(row) for row in modulators_raw]
    proteins = [normalize_protein(row) for row in proteins_raw]
    interactions = build_online_interactions(modulators, proteins)

    asd_smiles_cache = smiles_cache_for_modulators(
        modulators, args.cache_dir / "asd_online_smiles_inchikey_by_smiles.json"
    )
    add_ligand_identity(modulators, interactions, asd_smiles_cache)
    add_as_complex_flags(interactions, read_as_complex_keys(args.asd_as_archive))
    write_tsv(args.interactions_tsv, interactions, INTERACTION_FIELDNAMES)

    labels: list[dict[str, Any]] = []
    cache_stats: dict[str, Any] = {}

    inchikeys = {row.get("ligand_inchikey", "") for row in modulators if row.get("ligand_inchikey")}
    uniprots = {row.get("uniprot", "") for row in interactions if row.get("uniprot")}

    if not args.skip_chembl:
        molecule_cache = update_chembl_molecule_cache(
            inchikeys,
            args.cache_dir / "chembl_molecule_by_inchikey.json",
            args.chembl_molecule_batch_size,
            args.workers,
            args.online,
        )
        target_cache = read_json(args.chembl_cache_dir / "chembl_target_by_uniprot.json", {})
        missing_targets = uniprots - set(target_cache)
        if missing_targets and not args.online:
            raise RuntimeError(
                f"ChEMBL target cache lacks {len(missing_targets)} UniProt IDs; use --online to fetch them."
            )
        if args.online:
            target_cache = update_cache_parallel(
                target_cache,
                uniprots,
                get_chembl_targets,
                args.chembl_cache_dir / "chembl_target_by_uniprot.json",
                args.workers,
                "ChEMBL targets",
            )
        target_to_molecules: dict[str, set[str]] = defaultdict(set)
        for interaction in interactions:
            inchikey = interaction.get("ligand_inchikey", "")
            uniprot = interaction.get("uniprot", "")
            if not inchikey or not uniprot:
                continue
            molecule_ids = chembl_molecule_ids_for_inchikey(inchikey, molecule_cache)
            target_ids = chembl_target_ids_for_uniprot(uniprot, target_cache)
            for target_id in target_ids:
                target_to_molecules[target_id].update(molecule_ids)
        activity_cache = update_chembl_activity_cache(
            target_to_molecules,
            args.cache_dir / "chembl_activity_by_target_molecule_batch.json",
            args.chembl_activity_batch_size,
            args.workers,
            args.online,
        )
        chembl_index = build_chembl_activity_index(activity_cache)
        labels.extend(build_chembl_labels(interactions, molecule_cache, target_cache, chembl_index))
        cache_stats["chembl_molecule_status"] = dict(
            Counter((record or {}).get("status", "missing") for record in molecule_cache.values())
        )
        cache_stats["chembl_target_status"] = dict(
            Counter((record or {}).get("status", "missing") for record in target_cache.values())
        )
        cache_stats["chembl_activity_batch_status"] = dict(
            Counter((record or {}).get("status", "missing") for record in activity_cache.values())
        )

    if not args.skip_bindingdb:
        bindingdb_cache = update_bindingdb_cache(
            uniprots,
            args.bindingdb_cache_dir / "bindingdb_by_uniprot.json",
            args.bindingdb_cutoff_nm,
            args.bindingdb_workers,
            args.online,
        )
        bindingdb_smiles_cache = update_bindingdb_smiles_cache(
            bindingdb_cache,
            args.bindingdb_cache_dir / "bindingdb_smiles_inchikey_by_smiles.json",
        )
        bindingdb_index = build_bindingdb_activity_index(bindingdb_cache, bindingdb_smiles_cache)
        labels.extend(build_bindingdb_labels(interactions, bindingdb_index))
        cache_stats["bindingdb_target_status"] = dict(
            Counter((record or {}).get("status", "missing") for record in bindingdb_cache.values())
        )
        cache_stats["bindingdb_smiles_status"] = dict(
            Counter((record or {}).get("status", "missing") for record in bindingdb_smiles_cache.values())
        )

    labels = sorted(
        labels,
        key=lambda row: (
            int(row.get("asd_online_interaction_id") or 0),
            row.get("source", ""),
            ENDPOINT_PRIORITY.get(row.get("selected_endpoint_type", ""), 99),
        ),
    )
    compound_rows = build_compound_rows(modulators, interactions, labels)
    write_tsv(args.labels_tsv, labels, LABEL_FIELDNAMES)
    write_tsv(args.compounds_tsv, compound_rows, COMPOUND_FIELDNAMES)

    summary = build_summary(modulators, proteins, interactions, labels, compound_rows, args, cache_stats)
    write_json(args.summary_json, summary)
    write_summary_md(args.summary_md, summary)

    print(f"Wrote {len(labels)} label source rows to {display_path(args.labels_tsv)}")
    print(f"Wrote {len(compound_rows)} compound rows to {display_path(args.compounds_tsv)}")
    print(f"Wrote summary to {display_path(args.summary_md)}")


if __name__ == "__main__":
    main()
