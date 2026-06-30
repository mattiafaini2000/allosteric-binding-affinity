#!/usr/bin/env python3
"""Verify ASD ChEMBL and BindingDB-derived matches as Tier-B labels.

Tier B means exact ligand-target evidence, not exact crystallographic pose.
For ChEMBL, exactness is based on RCSB chemical-component InChIKey mapping to
ChEMBL molecules and ASD UniProt mapping to ChEMBL targets. For the local
DrugWise BindingDB-derived file, exactness is based on PDB-overlap candidates
whose BindingDB-derived SMILES resolves to the same InChIKey as the ASD/RCSB
ligand and whose target sequence is contained in the ASD UniProt sequence.

Row-level outputs stay under data/interim because they contain ASD-derived and
third-party row data. Local summaries contain aggregate counts only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
import urllib.parse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .discovery_chembl import (
    ACTIVITY_TYPES,
    STRICT_BINDING_TYPES,
    build_row_records,
    display_path,
    get_chembl_activities,
    get_chembl_molecules,
    get_chembl_targets,
    get_rcsb_chemcomp,
    read_asd_rows,
    read_json,
    require_chembl_caches,
    split_aliases,
    split_uniprots,
    update_cache_parallel,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASD_ARCHIVE = REPO_ROOT / "data/raw/asd/archives/ASD_Release_202309_AS.tar.gz"
DEFAULT_CHEMBL_CACHE_DIR = REPO_ROOT / "data/interim/chembl"
DEFAULT_DRUGWISE_BDB = REPO_ROOT / "DrugWise-Implementation-main/data/BDB2020+.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/interim/ligand_target"
DEFAULT_STATUS_TSV = DEFAULT_OUTPUT_DIR / "asd_chembl_bindingdb_tier_b_status.tsv"
DEFAULT_VERIFIED_TSV = DEFAULT_OUTPUT_DIR / "asd_chembl_bindingdb_tier_b_verified_labels.tsv"
DEFAULT_SUMMARY_JSON = (
    REPO_ROOT / "outputs/discovery/asd_chembl_bindingdb_tier_b_verification_summary.json"
)
DEFAULT_SUMMARY_MD = (
    REPO_ROOT / "outputs/discovery/asd_chembl_bindingdb_tier_b_verification_summary.md"
)

ENDPOINT_PRIORITY = {"Kd": 0, "Ki": 1, "IC50": 2, "EC50": 3, "AC50": 4}
CHEMBL_SOURCE = "chembl"
BINDINGDB_SOURCE = "bindingdb"
BINDINGDB_DERIVED_SOURCE = "bindingdb_derived"
BINDINGDB_REST_URL = "https://bindingdb.org/rest/getLigandsByUniprot"
DEFAULT_BINDINGDB_CUTOFF_NM = 1_000_000


def request_json(url: str, timeout: int = 30, attempts: int = 3) -> dict[str, Any] | None:
    import requests

    for attempt in range(attempts):
        try:
            response = requests.get(
                url,
                timeout=timeout,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "asd-tier-b-verification/0.1",
                },
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


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def clean_sequence(value: str) -> str:
    return re.sub(r"[^A-Z]", "", (value or "").upper())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def get_pubchem_properties_by_smiles(smiles: str) -> dict[str, Any]:
    encoded = urllib.parse.quote(smiles, safe="")
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
        f"{encoded}/property/InChIKey,CanonicalSMILES/JSON"
    )
    data = request_json(url, timeout=30)
    properties = (data or {}).get("PropertyTable", {}).get("Properties", [])
    if not properties:
        return {"status": "missing", "inchikey": "", "canonical_smiles": "", "cid": ""}
    first = properties[0]
    return {
        "status": "ok",
        "inchikey": first.get("InChIKey") or "",
        "canonical_smiles": first.get("CanonicalSMILES")
        or first.get("ConnectivitySMILES")
        or "",
        "cid": first.get("CID") or "",
    }


def get_uniprot_sequence(uniprot: str) -> dict[str, Any]:
    url = f"https://rest.uniprot.org/uniprotkb/{urllib.parse.quote(uniprot)}.json"
    data = request_json(url, timeout=30)
    if not data:
        return {"status": "missing", "sequence": "", "primary_accession": uniprot}
    sequence = ((data.get("sequence") or {}).get("value") or "").strip()
    organism = ((data.get("organism") or {}).get("scientificName") or "").strip()
    return {
        "status": "ok" if sequence else "missing",
        "sequence": sequence,
        "primary_accession": data.get("primaryAccession") or uniprot,
        "organism": organism,
        "protein_name": (
            ((data.get("proteinDescription") or {}).get("recommendedName") or {})
            .get("fullName", {})
            .get("value", "")
        ),
    }


def get_bindingdb_ligands_by_uniprot(uniprot: str, cutoff_nm: int) -> dict[str, Any]:
    import requests

    params = {
        "uniprot": f"{uniprot};{cutoff_nm}",
        "response": "application/json",
    }
    try:
        response = requests.get(
            BINDINGDB_REST_URL,
            params=params,
            timeout=90,
            headers={
                "Accept": "application/json",
                "User-Agent": "asd-tier-b-bindingdb/0.1",
            },
        )
        if response.status_code == 404:
            return {"status": "missing", "affinities": [], "hit_count": 0}
        if response.status_code in {429, 500, 502, 503, 504}:
            time.sleep(2.0)
            response = requests.get(
                BINDINGDB_REST_URL,
                params=params,
                timeout=90,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "asd-tier-b-bindingdb/0.1",
                },
            )
        response.raise_for_status()
        if not response.text.strip():
            return {"status": "missing", "affinities": [], "hit_count": 0}
        data = response.json()
    except Exception as exc:
        return {"status": "error", "error": str(exc), "affinities": [], "hit_count": 0}

    payload = (
        data.get("getLindsByUniprotResponse")
        or data.get("getLigandsByUniprotResponse")
        or data.get("getLindsByUniprotsResponse")
        or data
    )
    affinities = payload.get("bdb.affinities") or []
    if isinstance(affinities, dict):
        affinities = [affinities]
    normalized_affinities = []
    for record in affinities:
        if not isinstance(record, dict):
            continue
        normalized_affinities.append(
            {
                "monomerid": str(record.get("bdb.monomerid") or ""),
                "smiles": str(record.get("bdb.smile") or ""),
                "affinity_type": str(record.get("bdb.affinity_type") or ""),
                "affinity": str(record.get("bdb.affinity") or ""),
            }
        )
    return {
        "status": "ok" if normalized_affinities else "missing",
        "primary_uniprot": str(payload.get("bdb.primary") or uniprot),
        "alternative_uniprots": payload.get("bdb.alternative") or [],
        "hit_count": int(str(payload.get("bdb.hit") or len(normalized_affinities)) or 0),
        "uniprot_length": str(payload.get("bdb.uniprot_length") or ""),
        "affinities": normalized_affinities,
    }


def strip_cxsmiles(smiles: str) -> str:
    return re.sub(r"\s+\|.*\|\s*$", "", smiles or "").strip()


def smiles_to_inchikey_record(smiles: str) -> dict[str, Any]:
    from rdkit import Chem
    from rdkit.Chem import inchi

    cleaned = strip_cxsmiles(smiles)
    if not cleaned:
        return {"status": "missing", "inchikey": "", "canonical_smiles": ""}
    try:
        mol = Chem.MolFromSmiles(cleaned)
        if mol is None:
            return {"status": "parse_error", "inchikey": "", "canonical_smiles": ""}
        return {
            "status": "ok",
            "inchikey": inchi.MolToInchiKey(mol),
            "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc), "inchikey": "", "canonical_smiles": ""}


def parse_bindingdb_affinity(value: str) -> tuple[str, float | None]:
    text = (value or "").strip()
    match = re.match(r"^(<=|>=|<|>|=)?\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", text)
    if not match:
        return "", None
    relation = match.group(1) or "="
    return relation, safe_float(match.group(2))


def pk_from_nm(value_nm: float | None) -> float | None:
    if value_nm is None or value_nm <= 0:
        return None
    return 9.0 - math.log10(value_nm)


def normalize_bindingdb_endpoint(value: str) -> str:
    token = (value or "").strip()
    for endpoint in ACTIVITY_TYPES:
        if token.upper() == endpoint.upper():
            return endpoint
    return token


def bindingdb_sort_key(activity: dict[str, Any]) -> tuple[int, int, float, str]:
    endpoint = activity.get("standard_type") or ""
    relation = activity.get("standard_relation") or ""
    pk = safe_float(activity.get("pchembl_value"))
    return (
        ENDPOINT_PRIORITY.get(endpoint, 99),
        0 if relation == "=" else 1,
        -(pk if pk is not None else -999.0),
        str(activity.get("activity_id") or ""),
    )


def update_cache_serial(
    cache: dict[str, Any],
    keys: Iterable[str],
    fetch,
    cache_path: Path,
    label: str,
    flush_interval: int = 10,
) -> dict[str, Any]:
    missing = [key for key in sorted(set(keys)) if key not in cache]
    if not missing:
        print(f"{label}: cache complete ({len(cache)} entries)")
        return cache
    print(f"{label}: fetching {len(missing)} missing entries ({len(cache)} cached)")
    for idx, key in enumerate(missing, start=1):
        try:
            cache[key] = fetch(key)
        except Exception as exc:
            cache[key] = {"status": "error", "error": str(exc)}
        if idx % flush_interval == 0:
            write_json(cache_path, cache)
            print(f"{label}: {idx}/{len(missing)} fetched")
    write_json(cache_path, cache)
    return cache


def endpoint_group(activity_types: set[str]) -> str:
    if activity_types & STRICT_BINDING_TYPES:
        return "strict_binding_kd_ki"
    if activity_types:
        return "potency_or_functional"
    return ""


def activity_sort_key(activity: dict[str, Any]) -> tuple[int, int, int, float, str]:
    standard_type = activity.get("standard_type") or ""
    assay_type = activity.get("assay_type") or ""
    relation = activity.get("standard_relation") or ""
    pchembl = safe_float(activity.get("pchembl_value"))
    return (
        ENDPOINT_PRIORITY.get(standard_type, 99),
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


def pchembl_summary(activities: list[dict[str, Any]], selected_type: str) -> dict[str, str]:
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


def source_row_base(
    row: dict[str, str],
    row_index: int,
    source: str,
    source_scope: str,
) -> dict[str, Any]:
    return {
        "source": source,
        "source_scope": source_scope,
        "asd_row_index": row_index,
        "target_id": row.get("target_id", ""),
        "target_gene": row.get("target_gene", ""),
        "organism": row.get("organism", ""),
        "pdb_uniprot": row.get("pdb_uniprot", ""),
        "allosteric_pdb": row.get("allosteric_pdb", "").upper(),
        "modulator_alias": row.get("modulator_alias", ""),
        "modulator_chain": row.get("modulator_chain", ""),
        "modulator_resi": row.get("modulator_resi", ""),
        "modulator_class": row.get("modulator_class", ""),
        "modulator_name": row.get("modulator_name", ""),
        "asd_pubmed_id": row.get("pubmed_id", ""),
    }


def classify_chembl_row(
    asd_row: dict[str, str],
    record: dict[str, Any],
    row_index: int,
    activity_cache: dict[str, Any],
    target_cache: dict[str, Any],
) -> dict[str, Any]:
    aliases = record["aliases"]
    inchikeys = record["inchikeys"]
    molecule_ids = record["molecule_ids"]
    target_ids = record["target_chembl_ids"]

    activities: list[dict[str, Any]] = []
    for molecule_id in molecule_ids:
        for target_id in target_ids:
            key = f"{molecule_id}|{target_id}"
            activities.extend((activity_cache.get(key) or {}).get("activities", []))
    activities = dedupe_activities(activities)
    endpoint_types = sorted({activity.get("standard_type", "") for activity in activities if activity})
    binding_activities = [
        activity for activity in activities if activity.get("standard_type") in STRICT_BINDING_TYPES
    ]

    target_types = []
    for uniprot in record["uniprots"]:
        for target in (target_cache.get(uniprot) or {}).get("targets", []):
            target_id = target.get("target_chembl_id")
            if target_id in target_ids:
                target_types.append(
                    f"{target_id}:{target.get('target_type') or ''}:{target.get('organism') or ''}"
                )

    tier = "not_tier_b"
    status = "not_tier_b_chembl_no_usable_activity_after_exact_mapping"
    label_scope = ""
    basis = "Exact ligand and target mapping did not yield a usable ChEMBL pChEMBL activity."
    if not aliases:
        status = "not_tier_b_chembl_no_component_like_asd_alias"
        basis = "ASD modulator alias did not provide a component-like token for RCSB CCD mapping."
    elif not inchikeys:
        status = "not_tier_b_chembl_no_rcsb_ligand_inchikey"
        basis = "RCSB CCD did not provide an InChIKey for the ASD modulator alias tokens."
    elif not molecule_ids:
        status = "not_tier_b_chembl_no_molecule_for_ligand_inchikey"
        basis = "No ChEMBL molecule was found for the exact RCSB ligand InChIKey."
    elif not record["uniprots"]:
        status = "not_tier_b_chembl_no_asd_uniprot"
        basis = "ASD row did not provide a UniProt accession for target mapping."
    elif not target_ids:
        status = "not_tier_b_chembl_no_target_for_asd_uniprot"
        basis = "No ChEMBL target was found for the ASD UniProt accession."
    elif binding_activities:
        tier = "tier_b"
        status = "tier_b_chembl_exact_ligand_target_binding"
        label_scope = "exact_ligand_target_binding_kd_ki"
        basis = (
            "RCSB ligand InChIKey maps exactly to a ChEMBL molecule, ASD UniProt maps "
            "to a ChEMBL target, and at least one Kd/Ki pChEMBL record exists."
        )
    elif activities:
        tier = "tier_b"
        status = "tier_b_chembl_exact_ligand_target_potency_or_functional"
        label_scope = "exact_ligand_target_potency_or_functional"
        basis = (
            "RCSB ligand InChIKey maps exactly to a ChEMBL molecule, ASD UniProt maps "
            "to a ChEMBL target, and at least one non-Kd/Ki pChEMBL record exists."
        )

    selected = activities[0] if activities else {}
    pchembl_stats = pchembl_summary(activities, selected.get("standard_type") or "")
    result = source_row_base(asd_row, row_index, CHEMBL_SOURCE, "all_asd_rows")
    result.update(
        {
            "tier": tier,
            "status": status,
            "label_scope": label_scope,
            "ligand_alias_tokens": ";".join(aliases),
            "ligand_inchikeys": ";".join(inchikeys),
            "source_molecule_ids": ";".join(molecule_ids),
            "source_target_ids": ";".join(target_ids),
            "source_target_types": ";".join(sorted(set(target_types))),
            "activity_count": len(activities),
            "binding_activity_count": len(binding_activities),
            "endpoint_types": ";".join(endpoint_types),
            "selected_endpoint_type": selected.get("standard_type", ""),
            "selected_pchembl_or_pk": selected.get("pchembl_value", ""),
            "selected_pchembl_min_same_endpoint": pchembl_stats["min"],
            "selected_pchembl_median_same_endpoint": pchembl_stats["median"],
            "selected_pchembl_max_same_endpoint": pchembl_stats["max"],
            "selected_standard_relation": selected.get("standard_relation", ""),
            "selected_standard_value": selected.get("standard_value", ""),
            "selected_standard_units": selected.get("standard_units", ""),
            "selected_activity_id": selected.get("activity_id", ""),
            "selected_assay_id": selected.get("assay_chembl_id", ""),
            "selected_document_id": selected.get("document_chembl_id", ""),
            "selected_document_year": selected.get("document_year", ""),
            "selected_source_molecule_id": selected.get("molecule_chembl_id", ""),
            "selected_source_target_id": selected.get("target_chembl_id", ""),
            "activity_ids": ";".join(
                str(activity.get("activity_id")) for activity in activities if activity.get("activity_id")
            ),
            "target_sequence_match": "",
            "bindingdb_value_nM": "",
            "bindingdb_accurate": "",
            "bindingdb_smiles_inchikey": "",
            "verification_basis": basis,
        }
    )
    return result


def bindingdb_activities_for_row(
    row: dict[str, str],
    bindingdb_cache: dict[str, Any],
    smiles_inchikey_cache: dict[str, Any],
    chemcomp_cache: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], list[str], list[str], list[str]]:
    aliases = split_aliases(row.get("modulator_alias", ""))
    uniprots = split_uniprots(row.get("pdb_uniprot", ""))
    ligand_inchikeys = sorted(
        {
            (chemcomp_cache.get(alias) or {}).get("inchikey")
            for alias in aliases
            if (chemcomp_cache.get(alias) or {}).get("inchikey")
        }
    )
    activities: list[dict[str, Any]] = []
    for uniprot in uniprots:
        for record in (bindingdb_cache.get(uniprot) or {}).get("affinities", []):
            endpoint = normalize_bindingdb_endpoint(record.get("affinity_type", ""))
            if endpoint not in ACTIVITY_TYPES:
                continue
            smiles = record.get("smiles", "")
            inchikey = (smiles_inchikey_cache.get(smiles) or {}).get("inchikey", "")
            if not inchikey or inchikey not in ligand_inchikeys:
                continue
            relation, value_nm = parse_bindingdb_affinity(record.get("affinity", ""))
            pk_value = pk_from_nm(value_nm)
            if value_nm is None or pk_value is None:
                continue
            monomer_id = str(record.get("monomerid") or "")
            activities.append(
                {
                    "activity_id": f"BDBM{monomer_id}:{uniprot}:{endpoint}:{record.get('affinity','').strip()}",
                    "standard_type": endpoint,
                    "standard_relation": relation,
                    "standard_value": f"{value_nm:.12g}",
                    "standard_units": "nM",
                    "pchembl_value": f"{pk_value:.9g}",
                    "molecule_chembl_id": f"BDBM{monomer_id}" if monomer_id else "",
                    "target_chembl_id": uniprot,
                    "bindingdb_smiles": smiles,
                    "bindingdb_ligand_inchikey": inchikey,
                }
            )
    deduped: dict[str, dict[str, Any]] = {}
    for activity in activities:
        deduped[activity["activity_id"]] = activity
    return (
        sorted(deduped.values(), key=bindingdb_sort_key),
        aliases,
        uniprots,
        ligand_inchikeys,
        sorted({activity["bindingdb_ligand_inchikey"] for activity in deduped.values()}),
    )


def classify_bindingdb_full_row(
    asd_row: dict[str, str],
    row_index: int,
    bindingdb_cache: dict[str, Any],
    smiles_inchikey_cache: dict[str, Any],
    chemcomp_cache: dict[str, Any],
) -> dict[str, Any]:
    activities, aliases, uniprots, ligand_inchikeys, matched_inchikeys = bindingdb_activities_for_row(
        asd_row, bindingdb_cache, smiles_inchikey_cache, chemcomp_cache
    )
    endpoint_types = sorted({activity.get("standard_type", "") for activity in activities if activity})
    binding_activities = [
        activity for activity in activities if activity.get("standard_type") in STRICT_BINDING_TYPES
    ]
    target_records = [
        bindingdb_cache.get(uniprot) or {} for uniprot in uniprots if bindingdb_cache.get(uniprot)
    ]

    tier = "not_tier_b"
    status = "not_tier_b_bindingdb_no_usable_activity_after_exact_mapping"
    label_scope = ""
    basis = "BindingDB target records did not yield a usable exact ligand-target activity."
    if not aliases:
        status = "not_tier_b_bindingdb_no_component_like_asd_alias"
        basis = "ASD modulator alias did not provide a component-like token for RCSB CCD mapping."
    elif not ligand_inchikeys:
        status = "not_tier_b_bindingdb_no_rcsb_ligand_inchikey"
        basis = "RCSB CCD did not provide an InChIKey for the ASD modulator alias tokens."
    elif not uniprots:
        status = "not_tier_b_bindingdb_no_asd_uniprot"
        basis = "ASD row did not provide a UniProt accession for target mapping."
    elif not target_records:
        status = "not_tier_b_bindingdb_target_not_queried"
        basis = "No BindingDB target cache record is available for the ASD UniProt accession."
    elif not any(record.get("status") == "ok" for record in target_records):
        status = "not_tier_b_bindingdb_no_target_records"
        basis = "BindingDB returned no target-ligand affinity records for the ASD UniProt accession."
    elif not activities:
        status = "not_tier_b_bindingdb_no_ligand_inchikey_match"
        basis = (
            "BindingDB returned target-ligand records for the ASD UniProt accession, "
            "but no returned ligand SMILES converted to the ASD/RCSB ligand InChIKey."
        )
    elif binding_activities:
        tier = "tier_b"
        status = "tier_b_bindingdb_exact_ligand_target_binding"
        label_scope = "exact_ligand_target_binding_kd_ki"
        basis = (
            "BindingDB target query by ASD UniProt returned a ligand whose SMILES "
            "converts to the ASD/RCSB ligand InChIKey and at least one Kd/Ki record."
        )
    else:
        tier = "tier_b"
        status = "tier_b_bindingdb_exact_ligand_target_potency_or_functional"
        label_scope = "exact_ligand_target_potency_or_functional"
        basis = (
            "BindingDB target query by ASD UniProt returned a ligand whose SMILES "
            "converts to the ASD/RCSB ligand InChIKey and at least one non-Kd/Ki activity record."
        )

    selected = activities[0] if activities else {}
    pchembl_stats = pchembl_summary(activities, selected.get("standard_type") or "")
    result = source_row_base(asd_row, row_index, BINDINGDB_SOURCE, "bindingdb_rest_uniprot")
    result.update(
        {
            "tier": tier,
            "status": status,
            "label_scope": label_scope,
            "ligand_alias_tokens": ";".join(aliases),
            "ligand_inchikeys": ";".join(ligand_inchikeys),
            "source_molecule_ids": ";".join(
                sorted({activity.get("molecule_chembl_id", "") for activity in activities if activity.get("molecule_chembl_id")})
            ),
            "source_target_ids": ";".join(uniprots),
            "source_target_types": "BindingDB UniProt target",
            "activity_count": len(activities),
            "binding_activity_count": len(binding_activities),
            "endpoint_types": ";".join(endpoint_types),
            "selected_endpoint_type": selected.get("standard_type", ""),
            "selected_pchembl_or_pk": selected.get("pchembl_value", ""),
            "selected_pchembl_min_same_endpoint": pchembl_stats["min"],
            "selected_pchembl_median_same_endpoint": pchembl_stats["median"],
            "selected_pchembl_max_same_endpoint": pchembl_stats["max"],
            "selected_standard_relation": selected.get("standard_relation", ""),
            "selected_standard_value": selected.get("standard_value", ""),
            "selected_standard_units": selected.get("standard_units", ""),
            "selected_activity_id": selected.get("activity_id", ""),
            "selected_assay_id": "",
            "selected_document_id": "",
            "selected_document_year": "",
            "selected_source_molecule_id": selected.get("molecule_chembl_id", ""),
            "selected_source_target_id": selected.get("target_chembl_id", ""),
            "activity_ids": ";".join(
                str(activity.get("activity_id")) for activity in activities if activity.get("activity_id")
            ),
            "target_sequence_match": "queried_by_uniprot" if uniprots else "",
            "bindingdb_value_nM": selected.get("standard_value", ""),
            "bindingdb_accurate": "",
            "bindingdb_smiles_inchikey": ";".join(matched_inchikeys),
            "verification_basis": basis,
        }
    )
    return result


def sequence_match_status(bdb_sequence: str, uniprots: list[str], uniprot_cache: dict[str, Any]) -> str:
    bdb_clean = clean_sequence(bdb_sequence)
    if not bdb_clean:
        return "missing_bindingdb_sequence"
    for uniprot in uniprots:
        uni_clean = clean_sequence((uniprot_cache.get(uniprot) or {}).get("sequence", ""))
        if not uni_clean:
            continue
        if bdb_clean in uni_clean:
            return "bindingdb_sequence_is_subsequence_of_uniprot"
        if uni_clean in bdb_clean:
            return "uniprot_sequence_is_subsequence_of_bindingdb"
    return "no_uniprot_sequence_match"


def classify_bindingdb_row(
    asd_row: dict[str, str],
    row_index: int,
    bdb_row: dict[str, str],
    chemcomp_cache: dict[str, Any],
    pubchem_cache: dict[str, Any],
    uniprot_cache: dict[str, Any],
) -> dict[str, Any]:
    aliases = split_aliases(asd_row.get("modulator_alias", ""))
    uniprots = split_uniprots(asd_row.get("pdb_uniprot", ""))
    ligand_inchikeys = sorted(
        {
            (chemcomp_cache.get(alias) or {}).get("inchikey")
            for alias in aliases
            if (chemcomp_cache.get(alias) or {}).get("inchikey")
        }
    )
    smiles = bdb_row.get("smiles", "")
    smiles_record = pubchem_cache.get(smiles) or {}
    bdb_inchikey = smiles_record.get("inchikey") or ""
    chemical_match = bool(bdb_inchikey and bdb_inchikey in ligand_inchikeys)
    seq_status = sequence_match_status(bdb_row.get("seq", ""), uniprots, uniprot_cache)
    target_match = seq_status in {
        "bindingdb_sequence_is_subsequence_of_uniprot",
        "uniprot_sequence_is_subsequence_of_bindingdb",
    }

    tier = "not_tier_b"
    status = "not_tier_b_bindingdb_derived_not_verified"
    label_scope = ""
    basis = "The local BindingDB-derived candidate did not pass exact ligand-target checks."
    if not ligand_inchikeys:
        status = "not_tier_b_bindingdb_no_rcsb_ligand_inchikey"
        basis = "RCSB CCD did not provide an InChIKey for the ASD modulator alias tokens."
    elif not bdb_inchikey:
        status = "not_tier_b_bindingdb_smiles_not_resolved_to_inchikey"
        basis = "BindingDB-derived SMILES could not be resolved to an InChIKey through PubChem."
    elif not chemical_match:
        status = "not_tier_b_bindingdb_ligand_inchikey_mismatch"
        basis = "BindingDB-derived ligand InChIKey differs from the ASD/RCSB ligand InChIKey."
    elif not target_match:
        status = "not_tier_b_bindingdb_target_sequence_mismatch"
        basis = "BindingDB-derived target sequence did not match the ASD UniProt sequence."
    else:
        tier = "tier_b"
        status = "tier_b_bindingdb_derived_exact_ligand_target_endpoint_unknown"
        label_scope = "exact_ligand_target_bindingdb_derived_pk_endpoint_unknown"
        basis = (
            "BindingDB-derived SMILES resolves to the same InChIKey as the ASD/RCSB "
            "ligand, and the BindingDB-derived target sequence matches the ASD UniProt sequence. "
            "The local file provides pK but not original endpoint type."
        )

    result = source_row_base(
        asd_row, row_index, BINDINGDB_DERIVED_SOURCE, "drugwise_bdb2020plus_pdb_overlap"
    )
    result.update(
        {
            "tier": tier,
            "status": status,
            "label_scope": label_scope,
            "ligand_alias_tokens": ";".join(aliases),
            "ligand_inchikeys": ";".join(ligand_inchikeys),
            "source_molecule_ids": f"PubChem:{smiles_record.get('cid', '')}"
            if smiles_record.get("cid")
            else "",
            "source_target_ids": ";".join(uniprots),
            "source_target_types": "UniProt sequence",
            "activity_count": 1 if bdb_row.get("pK") else 0,
            "binding_activity_count": "",
            "endpoint_types": "unknown",
            "selected_endpoint_type": "unknown",
            "selected_pchembl_or_pk": bdb_row.get("pK", ""),
            "selected_pchembl_min_same_endpoint": "",
            "selected_pchembl_median_same_endpoint": "",
            "selected_pchembl_max_same_endpoint": "",
            "selected_standard_relation": "",
            "selected_standard_value": bdb_row.get("value", ""),
            "selected_standard_units": "nM",
            "selected_activity_id": "",
            "selected_assay_id": "",
            "selected_document_id": "",
            "selected_document_year": "",
            "selected_source_molecule_id": f"PubChem:{smiles_record.get('cid', '')}"
            if smiles_record.get("cid")
            else "",
            "selected_source_target_id": ";".join(uniprots),
            "activity_ids": "",
            "target_sequence_match": seq_status,
            "bindingdb_value_nM": bdb_row.get("value", ""),
            "bindingdb_accurate": bdb_row.get("accurate", ""),
            "bindingdb_smiles_inchikey": bdb_inchikey,
            "verification_basis": basis,
        }
    )
    return result


def build_summary(
    status_rows: list[dict[str, Any]],
    status_tsv: Path,
    verified_tsv: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in status_rows:
        by_source[row["source"]].append(row)

    def unique_count(rows: list[dict[str, Any]], field: str) -> int:
        return len({str(row.get(field, "")) for row in rows if row.get(field, "") != ""})

    def source_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        tier_b_rows = [row for row in rows if row.get("tier") == "tier_b"]
        strict_rows = [
            row
            for row in tier_b_rows
            if row.get("label_scope") == "exact_ligand_target_binding_kd_ki"
        ]
        potency_rows = [
            row
            for row in tier_b_rows
            if row.get("label_scope") == "exact_ligand_target_potency_or_functional"
        ]
        endpoint_counter: Counter[str] = Counter()
        for row in tier_b_rows:
            for endpoint in str(row.get("endpoint_types", "")).split(";"):
                if endpoint:
                    endpoint_counter[endpoint] += 1
        return {
            "evaluated_rows": len(rows),
            "tier_counts": dict(Counter(row.get("tier") for row in rows)),
            "status_counts": dict(Counter(row.get("status") for row in rows)),
            "tier_b": {
                "asd_rows": len(tier_b_rows),
                "unique_pdb_ids": unique_count(tier_b_rows, "allosteric_pdb"),
                "unique_target_ids": unique_count(tier_b_rows, "target_id"),
                "strict_binding_kd_ki_rows": len(strict_rows),
                "potency_or_functional_rows": len(potency_rows),
                "endpoint_row_counts": dict(endpoint_counter.most_common()),
            },
        }

    tier_b_rows = [row for row in status_rows if row.get("tier") == "tier_b"]
    strict_rows = [
        row for row in tier_b_rows if row.get("label_scope") == "exact_ligand_target_binding_kd_ki"
    ]
    bindingdb_rows = [
        row
        for row in tier_b_rows
        if row.get("label_scope")
        == "exact_ligand_target_bindingdb_derived_pk_endpoint_unknown"
    ]
    by_asd_row: dict[str, set[str]] = defaultdict(set)
    for row in tier_b_rows:
        by_asd_row[str(row["asd_row_index"])].add(row["source"])

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "asd_archive": display_path(args.asd_archive),
            "chembl_cache_dir": display_path(args.chembl_cache_dir),
            "bindingdb_cutoff_nm": args.bindingdb_cutoff_nm,
            "bindingdb_uniprot_cache": display_path(status_tsv.parent / "bindingdb_by_uniprot.json"),
            "bindingdb_smiles_inchikey_cache": display_path(
                status_tsv.parent / "bindingdb_smiles_inchikey_by_smiles.json"
            ),
            "drugwise_bdb_csv": display_path(args.drugwise_bdb_csv),
            "status_tsv": display_path(status_tsv),
            "verified_tsv": display_path(verified_tsv),
        },
        "scope": {
            "total_status_rows": len(status_rows),
            "chembl_status_rows": len(by_source.get(CHEMBL_SOURCE, [])),
            "bindingdb_status_rows": len(by_source.get(BINDINGDB_SOURCE, [])),
            "bindingdb_derived_status_rows": len(by_source.get(BINDINGDB_DERIVED_SOURCE, [])),
        },
        "sources": {
            source: source_summary(rows) for source, rows in sorted(by_source.items())
        },
        "combined_tier_b": {
            "source_rows": len(tier_b_rows),
            "unique_asd_rows": len(by_asd_row),
            "unique_pdb_ids": unique_count(tier_b_rows, "allosteric_pdb"),
            "unique_target_ids": unique_count(tier_b_rows, "target_id"),
            "strict_binding_kd_ki_source_rows": len(strict_rows),
            "strict_binding_kd_ki_unique_asd_rows": unique_count(strict_rows, "asd_row_index"),
            "strict_binding_kd_ki_unique_pdb_ids": unique_count(strict_rows, "allosteric_pdb"),
            "strict_binding_kd_ki_unique_target_ids": unique_count(strict_rows, "target_id"),
            "bindingdb_endpoint_unknown_source_rows": len(bindingdb_rows),
            "source_overlap_by_asd_row": {
                "+".join(sorted(sources)): count
                for sources, count in Counter(
                    tuple(sorted(sources)) for sources in by_asd_row.values()
                ).items()
            },
        },
        "notes": [
            "Tier B means exact ligand-target evidence, not exact PDB-complex or pose evidence.",
            "ChEMBL Tier-B rows require RCSB ligand InChIKey to ChEMBL molecule mapping, ASD UniProt to ChEMBL target mapping, and at least one pChEMBL activity.",
            "Full BindingDB Tier-B rows query BindingDB by ASD UniProt, convert returned BindingDB SMILES to InChIKey with RDKit, and require an exact match to the ASD/RCSB ligand InChIKey.",
            "BindingDB-derived Tier-B rows use the local DrugWise BDB2020+ file and require BindingDB-derived SMILES to match the ASD/RCSB InChIKey plus target sequence agreement with ASD UniProt.",
            "BindingDB-derived rows keep endpoint_type=unknown because the local BDB2020+ file does not preserve the original BindingDB endpoint type.",
            "The local row-level TSVs are written under data/interim and are not versioned because they contain ASD/ChEMBL/BindingDB-derived row data.",
        ],
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# ASD ChEMBL/BindingDB Tier-B Verification",
        "",
        f"Generated at UTC: `{summary['generated_at_utc']}`.",
        "",
        "## Purpose",
        "",
        "This report summarizes ligand-target verification for ChEMBL, full BindingDB "
        "REST target queries, and the local DrugWise BindingDB-derived `BDB2020+` file. "
        "Tier B means exact molecule/target evidence, not exact crystallographic pose evidence.",
        "",
        "## Inputs",
        "",
    ]
    for key, value in summary["inputs"].items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(
        [
            "",
            "## Scope",
            "",
            f"- Total status rows: {summary['scope']['total_status_rows']}",
            f"- ChEMBL status rows: {summary['scope']['chembl_status_rows']}",
            f"- BindingDB status rows: {summary['scope']['bindingdb_status_rows']}",
            f"- BindingDB-derived status rows: {summary['scope']['bindingdb_derived_status_rows']}",
            "",
            "## Combined Tier B",
            "",
            f"- Tier-B source rows: {summary['combined_tier_b']['source_rows']}",
            f"- Unique ASD rows with at least one Tier-B source: {summary['combined_tier_b']['unique_asd_rows']}",
            f"- Unique PDB IDs: {summary['combined_tier_b']['unique_pdb_ids']}",
            f"- Unique ASD target IDs: {summary['combined_tier_b']['unique_target_ids']}",
            f"- Strict `Kd`/`Ki` source rows: {summary['combined_tier_b']['strict_binding_kd_ki_source_rows']}",
            f"- Strict `Kd`/`Ki` unique ASD rows: {summary['combined_tier_b']['strict_binding_kd_ki_unique_asd_rows']}",
            f"- Strict `Kd`/`Ki` unique PDB IDs: {summary['combined_tier_b']['strict_binding_kd_ki_unique_pdb_ids']}",
            f"- Strict `Kd`/`Ki` unique ASD target IDs: {summary['combined_tier_b']['strict_binding_kd_ki_unique_target_ids']}",
            f"- BindingDB-derived endpoint-unknown source rows: {summary['combined_tier_b']['bindingdb_endpoint_unknown_source_rows']}",
            "",
            "Source overlap by ASD row:",
            "",
        ]
    )
    for key, count in summary["combined_tier_b"]["source_overlap_by_asd_row"].items():
        lines.append(f"- `{key}`: {count}")

    for source, source_summary in summary["sources"].items():
        lines.extend(
            [
                "",
                f"## {source}",
                "",
                f"- Evaluated rows: {source_summary['evaluated_rows']}",
                f"- Tier-B rows: {source_summary['tier_b']['asd_rows']}",
                f"- Unique PDB IDs: {source_summary['tier_b']['unique_pdb_ids']}",
                f"- Unique ASD target IDs: {source_summary['tier_b']['unique_target_ids']}",
                f"- Strict `Kd`/`Ki` rows: {source_summary['tier_b']['strict_binding_kd_ki_rows']}",
                f"- Potency/functional rows: {source_summary['tier_b']['potency_or_functional_rows']}",
                "",
                "Tier counts:",
                "",
            ]
        )
        for tier, count in source_summary["tier_counts"].items():
            lines.append(f"- `{tier}`: {count}")
        lines.extend(["", "Status counts:", ""])
        for status, count in source_summary["status_counts"].items():
            lines.append(f"- `{status}`: {count}")
        lines.extend(["", "Endpoint row counts among Tier-B rows:", ""])
        for endpoint, count in source_summary["tier_b"]["endpoint_row_counts"].items():
            lines.append(f"- `{endpoint}`: {count}")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "ChEMBL rows are verified exact ligand-target labels after chemical InChIKey "
            "and UniProt target mapping. Full BindingDB rows are verified by querying "
            "BindingDB with the ASD UniProt target, converting returned BindingDB SMILES "
            "to InChIKey with RDKit, and requiring exact agreement with the ASD/RCSB ligand "
            "InChIKey. Local BindingDB-derived rows are also ligand-target verified when "
            "their local SMILES and sequence evidence pass, but the local `BDB2020+` file "
            "does not expose the original endpoint type, so those labels should stay "
            "endpoint-unknown until traced to original BindingDB records.",
            "",
            "Use the local status TSV as the worklist for downstream label resolution. "
            "Keep source, endpoint type, and tier fields in any modeling table.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


STATUS_FIELDNAMES = [
    "source",
    "source_scope",
    "asd_row_index",
    "target_id",
    "target_gene",
    "organism",
    "pdb_uniprot",
    "allosteric_pdb",
    "modulator_alias",
    "modulator_chain",
    "modulator_resi",
    "modulator_class",
    "modulator_name",
    "asd_pubmed_id",
    "tier",
    "status",
    "label_scope",
    "ligand_alias_tokens",
    "ligand_inchikeys",
    "source_molecule_ids",
    "source_target_ids",
    "source_target_types",
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
    "selected_source_molecule_id",
    "selected_source_target_id",
    "activity_ids",
    "target_sequence_match",
    "bindingdb_value_nM",
    "bindingdb_accurate",
    "bindingdb_smiles_inchikey",
    "verification_basis",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asd-archive", type=Path, default=DEFAULT_ASD_ARCHIVE)
    parser.add_argument("--chembl-cache-dir", type=Path, default=DEFAULT_CHEMBL_CACHE_DIR)
    parser.add_argument("--drugwise-bdb-csv", type=Path, default=DEFAULT_DRUGWISE_BDB)
    parser.add_argument("--status-tsv", type=Path, default=DEFAULT_STATUS_TSV)
    parser.add_argument("--verified-tsv", type=Path, default=DEFAULT_VERIFIED_TSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bindingdb-cutoff-nm", type=int, default=DEFAULT_BINDINGDB_CUTOFF_NM)
    parser.add_argument(
        "--online",
        action="store_true",
        help="Fetch missing ligand, target, and activity evidence.",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Use existing local caches only; do not query RCSB/ChEMBL/BindingDB/PubChem/UniProt.",
    )
    args = parser.parse_args()
    if args.online and args.skip_fetch:
        raise ValueError("--online and --skip-fetch cannot be used together")
    fetch_missing = args.online and not args.skip_fetch
    if not fetch_missing:
        require_chembl_caches(args.chembl_cache_dir)

    asd_rows = read_asd_rows(args.asd_archive)

    aliases = {
        alias for row in asd_rows for alias in split_aliases(row.get("modulator_alias", ""))
    }
    uniprots = {
        uniprot for row in asd_rows for uniprot in split_uniprots(row.get("pdb_uniprot", ""))
    }

    chemcomp_path = args.chembl_cache_dir / "rcsb_chemcomp_by_alias.json"
    molecule_path = args.chembl_cache_dir / "chembl_molecule_by_inchikey.json"
    target_path = args.chembl_cache_dir / "chembl_target_by_uniprot.json"
    activity_path = args.chembl_cache_dir / "chembl_activity_by_molecule_target.json"

    chemcomp_cache = read_json(chemcomp_path, {})
    molecule_cache = read_json(molecule_path, {})
    target_cache = read_json(target_path, {})
    activity_cache = read_json(activity_path, {})

    if fetch_missing:
        chemcomp_cache = update_cache_parallel(
            chemcomp_cache,
            aliases,
            get_rcsb_chemcomp,
            chemcomp_path,
            args.workers,
            "RCSB CCD",
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

    chembl_status_rows = [
        classify_chembl_row(asd_rows[record["row_index"]], record, record["row_index"], activity_cache, target_cache)
        for record in row_records
    ]

    bindingdb_cache_path = args.status_tsv.parent / "bindingdb_by_uniprot.json"
    bindingdb_smiles_cache_path = args.status_tsv.parent / "bindingdb_smiles_inchikey_by_smiles.json"
    if not fetch_missing and (not bindingdb_cache_path.exists() or not bindingdb_smiles_cache_path.exists()):
        raise FileNotFoundError(
            "Local BindingDB target and SMILES caches are required; use --online to populate them."
        )
    bindingdb_cache = read_json(bindingdb_cache_path, {})
    bindingdb_smiles_cache = read_json(bindingdb_smiles_cache_path, {})

    if fetch_missing:
        bindingdb_cache = {
            key: value
            for key, value in bindingdb_cache.items()
            if (value or {}).get("status") != "error"
        }
        bindingdb_workers = max(1, min(args.workers, 4))

        def fetch_bindingdb_uniprot(uniprot: str) -> dict[str, Any]:
            return get_bindingdb_ligands_by_uniprot(uniprot, args.bindingdb_cutoff_nm)

        bindingdb_cache = update_cache_parallel(
            bindingdb_cache,
            uniprots,
            fetch_bindingdb_uniprot,
            bindingdb_cache_path,
            bindingdb_workers,
            "BindingDB UniProt target records",
        )

        bindingdb_smiles = {
            record.get("smiles", "")
            for target_record in bindingdb_cache.values()
            for record in (target_record or {}).get("affinities", [])
            if normalize_bindingdb_endpoint(record.get("affinity_type", "")) in ACTIVITY_TYPES
            and record.get("smiles", "")
        }
        bindingdb_smiles_cache = update_cache_serial(
            bindingdb_smiles_cache,
            bindingdb_smiles,
            smiles_to_inchikey_record,
            bindingdb_smiles_cache_path,
            "BindingDB SMILES InChIKey conversion",
            flush_interval=1000,
        )

    bindingdb_full_status_rows = [
        classify_bindingdb_full_row(
            row, idx, bindingdb_cache, bindingdb_smiles_cache, chemcomp_cache
        )
        for idx, row in enumerate(asd_rows)
    ]

    bdb_rows = read_csv(args.drugwise_bdb_csv) if args.drugwise_bdb_csv.exists() else []
    bdb_by_pdb = {
        row.get("PDBID", "").strip().upper(): row
        for row in bdb_rows
        if row.get("PDBID", "").strip()
    }
    bdb_candidates = [
        (idx, row, bdb_by_pdb[row.get("allosteric_pdb", "").strip().upper()])
        for idx, row in enumerate(asd_rows)
        if row.get("allosteric_pdb", "").strip().upper() in bdb_by_pdb
    ]

    pubchem_cache_path = args.status_tsv.parent / "pubchem_by_smiles.json"
    uniprot_cache_path = args.status_tsv.parent / "uniprot_sequence_by_accession.json"
    if bdb_candidates and not fetch_missing and (
        not pubchem_cache_path.exists() or not uniprot_cache_path.exists()
    ):
        raise FileNotFoundError(
            "Local PubChem and UniProt caches are required for BDB-derived candidates; "
            "use --online to populate them."
        )
    pubchem_cache = read_json(pubchem_cache_path, {})
    uniprot_cache = read_json(uniprot_cache_path, {})

    if fetch_missing:
        pubchem_cache = update_cache_serial(
            pubchem_cache,
            (bdb_row.get("smiles", "") for _, _, bdb_row in bdb_candidates if bdb_row.get("smiles")),
            get_pubchem_properties_by_smiles,
            pubchem_cache_path,
            "PubChem SMILES properties",
        )
        candidate_uniprots = {
            uniprot
            for _, row, _ in bdb_candidates
            for uniprot in split_uniprots(row.get("pdb_uniprot", ""))
        }
        uniprot_cache = update_cache_serial(
            uniprot_cache,
            candidate_uniprots,
            get_uniprot_sequence,
            uniprot_cache_path,
            "UniProt sequences",
        )

    bindingdb_derived_status_rows = [
        classify_bindingdb_row(row, idx, bdb_row, chemcomp_cache, pubchem_cache, uniprot_cache)
        for idx, row, bdb_row in bdb_candidates
    ]

    status_rows = chembl_status_rows + bindingdb_full_status_rows + bindingdb_derived_status_rows
    verified_rows = [row for row in status_rows if row.get("tier") == "tier_b"]
    write_tsv(args.status_tsv, status_rows, STATUS_FIELDNAMES)
    write_tsv(args.verified_tsv, verified_rows, STATUS_FIELDNAMES)

    summary = build_summary(status_rows, args.status_tsv, args.verified_tsv, args)
    write_json(args.summary_json, summary)
    write_summary_md(args.summary_md, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
