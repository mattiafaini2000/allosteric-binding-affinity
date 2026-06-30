#!/usr/bin/env python3
"""Map the complete ASD online modulator universe to PDBbind v2020.

The ASD online modulator endpoint lists about 100k modulators and their
related ASD target serials. To test how much of that universe can be connected
to PDBbind, this script expands each related target serial through the ASD
online protein cache, joins the resulting target PDB IDs to local PDBbind
v2020 labels, and verifies whether the PDBbind-selected ligand has the same
InChIKey as the ASD modulator.

Row-level outputs stay under data/interim because they contain ASD-derived and
PDBbind-derived rows. Local summaries contain aggregate counts only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .verification_ligand_target import (
    safe_float,
    smiles_to_inchikey_record,
)
from .discovery_chembl import get_rcsb_chemcomp
from .verification_pdbbind import (
    display_path,
    fetch_pdbbind_plus_record,
    read_csv,
    read_json,
    update_cache_parallel,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASD_ONLINE_DIR = REPO_ROOT / "data/interim/asd_online"
DEFAULT_ASD_MODULATORS_JSON = DEFAULT_ASD_ONLINE_DIR / "asd_online_modulators.json"
DEFAULT_ASD_PROTEINS_JSON = DEFAULT_ASD_ONLINE_DIR / "asd_online_proteins.json"
DEFAULT_ASD_SMILES_CACHE = DEFAULT_ASD_ONLINE_DIR / "asd_online_smiles_inchikey_by_smiles.json"
DEFAULT_HITLEAD_ARCHIVE = REPO_ROOT / "data/raw/asd/archives/ASD_Release_202306_HL.tar.gz"
DEFAULT_PDBBIND_CSV = REPO_ROOT / "DrugWise-Implementation-main/data/PDBbindv2020_General.csv"
DEFAULT_PDBBIND_PLUS_CACHE = REPO_ROOT / "data/interim/pdbbind/pdbbind_plus_browser_by_pdb.json"
DEFAULT_PDBBIND_SMILES_CACHE = (
    DEFAULT_ASD_ONLINE_DIR / "pdbbind_v2020_ligand_smiles_inchikey_by_smiles.json"
)
DEFAULT_PDBBIND_CHEMCOMP_CACHE = (
    DEFAULT_ASD_ONLINE_DIR / "pdbbind_v2020_selected_ligand_chemcomp_by_id.json"
)
DEFAULT_STATUS_TSV = DEFAULT_ASD_ONLINE_DIR / "asd_online_pdbbind_mapping_status.tsv"
DEFAULT_EXACT_TSV = DEFAULT_ASD_ONLINE_DIR / "asd_online_pdbbind_exact_labels.tsv"
DEFAULT_SUMMARY_JSON = REPO_ROOT / "outputs/discovery/asd_online_pdbbind_mapping_summary.json"
DEFAULT_SUMMARY_MD = REPO_ROOT / "outputs/discovery/asd_online_pdbbind_mapping_summary.md"

PDB_ID_RE = re.compile(r"^[0-9A-Z]{4}$")
CHEMCOMP_ID_RE = re.compile(r"^[0-9A-Z]{1,5}$")
ACTIVITY_TYPES = ("Kd", "Ki", "IC50", "EC50", "AC50")
STRICT_BINDING_TYPES = {"Kd", "Ki"}
ENDPOINT_PRIORITY = {"Kd": 0, "Ki": 1, "IC50": 2, "EC50": 3, "AC50": 4}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"null", "none", "undefined", "nan"}:
        return ""
    return text


def read_online_rows(path: Path, label: str) -> list[dict[str, Any]]:
    payload = read_json(path, {})
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError(
            f"Could not read ASD online {label} rows from {display_path(path)}. "
            "Run scripts/asd_expand_online_activity_labels.py once to create the caches."
        )
    return rows


def normalize_modulator(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "modulator_id": clean_text(row.get("modulator_id")),
        "db_serial": clean_text(row.get("db_serial")),
        "modulator_name": clean_text(row.get("modulator_name")),
        "modulator_class": clean_text(row.get("modulator_class")),
        "pubchem_id": clean_text(row.get("pubchem_id")),
        "cas_id": clean_text(row.get("cas_id")),
        "smiles": clean_text(row.get("smiles")),
        "formula": clean_text(row.get("formula")),
        "endogenous": clean_text(row.get("endogenous")),
        "drug_phase": clean_text(row.get("drug_phase")),
        "related_proteins": row.get("related_proteins") or [],
    }


def normalize_protein(row: dict[str, Any]) -> dict[str, Any]:
    swiss = clean_text(row.get("swissport_id")).upper()
    trembl = clean_text(row.get("trembl_id")).upper()
    return {
        "mol_id": clean_text(row.get("mol_id")),
        "db_serial": clean_text(row.get("db_serial")),
        "domain": clean_text(row.get("domain")),
        "gene_name": clean_text(row.get("gene_name")),
        "mol_name": clean_text(row.get("mol_name")),
        "organism": clean_text(row.get("organism")),
        "swissport_id": swiss,
        "trembl_id": trembl,
        "uniprot": swiss or trembl,
        "pdb_id": clean_text(row.get("pdb_id")).upper(),
        "pubmed_id": clean_text(row.get("pubmed_id")),
    }


def update_smiles_cache(
    modulators: list[dict[str, Any]],
    cache_path: Path,
) -> dict[str, Any]:
    cache = read_json(cache_path, {})
    smiles_values = sorted({row["smiles"] for row in modulators if row.get("smiles")})
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


def add_ligand_identity(modulators: list[dict[str, Any]], cache: dict[str, Any]) -> None:
    for row in modulators:
        record = cache.get(row.get("smiles", "")) or {}
        row["canonical_smiles"] = record.get("canonical_smiles", "")
        row["ligand_inchikey"] = record.get("inchikey", "")
        row["ligand_identity_status"] = record.get("status", "missing")


def valid_pdb_id(value: str) -> bool:
    pdb_id = clean_text(value).upper()
    return bool(PDB_ID_RE.fullmatch(pdb_id)) and not pdb_id.startswith("ASD")


def normalize_pdb_id(value: str) -> str:
    pdb_id = clean_text(value).upper()
    return pdb_id if valid_pdb_id(pdb_id) else ""


def build_domain_interactions(
    modulators: list[dict[str, Any]],
    proteins: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    protein_by_serial: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for protein in proteins:
        if protein.get("db_serial"):
            protein_by_serial[protein["db_serial"]].append(protein)

    rows: list[dict[str, Any]] = []
    stats = Counter()
    status_id = 0
    base_related_item_id = 0

    for modulator in modulators:
        related_items = modulator.get("related_proteins") or []
        if not isinstance(related_items, list):
            related_items = []
        for related_index, item in enumerate(related_items, start=1):
            if not isinstance(item, list) or len(item) < 4:
                stats["invalid_related_items"] += 1
                continue
            base_related_item_id += 1
            feature = clean_text(item[0])
            related_name = clean_text(item[1])
            related_numeric_id = clean_text(item[2])
            related_target_serial = clean_text(item[3])
            matches = protein_by_serial.get(related_target_serial, [])
            if not matches:
                status_id += 1
                stats["related_items_without_target_serial_match"] += 1
                rows.append(
                    interaction_row(
                        status_id=status_id,
                        base_related_item_id=base_related_item_id,
                        related_index=related_index,
                        modulator=modulator,
                        feature=feature,
                        related_name=related_name,
                        related_numeric_id=related_numeric_id,
                        related_target_serial=related_target_serial,
                        protein={},
                        match_count=0,
                    )
                )
                continue
            stats["related_items_with_target_serial_match"] += 1
            if len(matches) > 1:
                stats["related_items_with_multiple_domain_matches"] += 1
            for protein in matches:
                status_id += 1
                rows.append(
                    interaction_row(
                        status_id=status_id,
                        base_related_item_id=base_related_item_id,
                        related_index=related_index,
                        modulator=modulator,
                        feature=feature,
                        related_name=related_name,
                        related_numeric_id=related_numeric_id,
                        related_target_serial=related_target_serial,
                        protein=protein,
                        match_count=len(matches),
                    )
                )

    stats["base_related_items_total"] = base_related_item_id
    stats["domain_expanded_status_rows_total"] = len(rows)
    stats["domain_expanded_rows_with_target_serial_match"] = sum(
        1 for row in rows if row.get("domain_expansion_status") == "target_serial_matched"
    )
    return rows, dict(stats)


def interaction_row(
    *,
    status_id: int,
    base_related_item_id: int,
    related_index: int,
    modulator: dict[str, Any],
    feature: str,
    related_name: str,
    related_numeric_id: str,
    related_target_serial: str,
    protein: dict[str, Any],
    match_count: int,
) -> dict[str, Any]:
    pdb_id = clean_text(protein.get("pdb_id")).upper()
    return {
        "asd_online_pdbbind_status_id": status_id,
        "base_related_item_id": base_related_item_id,
        "related_item_index_for_modulator": related_index,
        "asd_modulator_id": modulator.get("modulator_id", ""),
        "asd_modulator_serial": modulator.get("db_serial", ""),
        "modulator_name": modulator.get("modulator_name", ""),
        "modulator_class": modulator.get("modulator_class", ""),
        "pubchem_id": modulator.get("pubchem_id", ""),
        "cas_id": modulator.get("cas_id", ""),
        "smiles": modulator.get("smiles", ""),
        "canonical_smiles": modulator.get("canonical_smiles", ""),
        "ligand_inchikey": modulator.get("ligand_inchikey", ""),
        "ligand_identity_status": modulator.get("ligand_identity_status", ""),
        "related_feature": feature,
        "related_protein_name": related_name,
        "related_numeric_id": related_numeric_id,
        "related_target_serial": related_target_serial,
        "target_serial_match_count": match_count,
        "domain_expansion_status": "target_serial_matched"
        if match_count
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
        "target_pdb_id": pdb_id,
        "target_valid_pdb_id": normalize_pdb_id(pdb_id),
        "target_pubmed_id": protein.get("pubmed_id", ""),
    }


def read_hitlead_rows(path: Path) -> list[dict[str, str]]:
    with tarfile.open(path, "r:gz") as tar:
        member = next(
            m for m in tar.getmembers() if Path(m.name).name == "ASD_Release_202306_HL.txt"
        )
        handle = tar.extractfile(member)
        if handle is None:
            raise RuntimeError(f"Could not extract {member.name} from {display_path(path)}")
        text = handle.read().decode("utf-8-sig")
    return list(csv.DictReader(text.splitlines(), delimiter="\t"))


def hitlead_value_field(rows: list[dict[str, str]]) -> str:
    for row in rows:
        for key in row:
            if key.startswith("Activity_Value"):
                return key
    return "Activity_Value(uM)"


def numeric(value: Any) -> bool:
    result = safe_float(value)
    return result is not None and math.isfinite(result)


def hl_px_from_um(value: Any) -> str:
    result = safe_float(value)
    if result is None or result <= 0:
        return ""
    return f"{6.0 - math.log10(result):.4g}"


def hitlead_key(row: dict[str, str]) -> tuple[str, str]:
    return (clean_text(row.get("Target_ID")), clean_text(row.get("Modulator_ID")))


def build_hitlead_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = hitlead_key(row)
        if key[0] and key[1]:
            by_key[key].append(row)
    return by_key


def summarize_hitlead_rows(rows: list[dict[str, str]], value_field: str) -> dict[str, str]:
    if not rows:
        return {
            "has_hitlead_activity": "0",
            "hitlead_row_count": "0",
            "hitlead_numeric_activity_row_count": "0",
            "hitlead_strict_binding_row_count": "0",
            "hitlead_activity_types": "",
            "hitlead_activity_values_uM": "",
            "hitlead_pvalues": "",
            "hitlead_pubmed_ids": "",
        }

    activity_types = [clean_text(row.get("Activity_Type")) for row in rows]
    numeric_rows = [row for row in rows if numeric(row.get(value_field))]
    strict_rows = [
        row
        for row in numeric_rows
        if clean_text(row.get("Activity_Type")) in STRICT_BINDING_TYPES
    ]
    typed_values = []
    pvalues = []
    for row in numeric_rows:
        activity_type = clean_text(row.get("Activity_Type")) or "unknown"
        value = clean_text(row.get(value_field))
        typed_values.append(f"{activity_type}:{value}")
        px = hl_px_from_um(value)
        if px:
            pvalues.append(f"p{activity_type}:{px}")

    return {
        "has_hitlead_activity": "1",
        "hitlead_row_count": str(len(rows)),
        "hitlead_numeric_activity_row_count": str(len(numeric_rows)),
        "hitlead_strict_binding_row_count": str(len(strict_rows)),
        "hitlead_activity_types": ";".join(sorted({item for item in activity_types if item})),
        "hitlead_activity_values_uM": "|".join(typed_values),
        "hitlead_pvalues": "|".join(pvalues),
        "hitlead_pubmed_ids": ";".join(
            sorted({clean_text(row.get("PubMed_ID")) for row in rows if clean_text(row.get("PubMed_ID"))})
        ),
    }


def pdbbind_by_pdb(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        pdb_id = clean_text(row.get("PDBID")).upper()
        if pdb_id:
            by_id[pdb_id] = row
    return by_id


def update_pdbbind_ligand_smiles_cache(
    pdb_ids: Iterable[str],
    pdbbind_plus_cache: dict[str, Any],
    smiles_cache_path: Path,
) -> dict[str, Any]:
    smiles_cache = read_json(smiles_cache_path, {})
    smiles_values = []
    for pdb_id in pdb_ids:
        record = pdbbind_plus_cache.get(pdb_id) or {}
        info = record.get("info") or {}
        smiles = clean_text(info.get("smilestring"))
        if smiles:
            smiles_values.append(smiles)
    missing = [smiles for smiles in sorted(set(smiles_values)) if smiles not in smiles_cache]
    if missing:
        print(f"PDBbind selected-ligand SMILES InChIKey: converting {len(missing)} missing structures")
        for idx, smiles in enumerate(missing, start=1):
            smiles_cache[smiles] = smiles_to_inchikey_record(smiles)
            if idx % 200 == 0:
                write_json(smiles_cache_path, smiles_cache)
        write_json(smiles_cache_path, smiles_cache)
    else:
        print(f"PDBbind selected-ligand SMILES InChIKey: cache complete ({len(smiles_cache)} entries)")
    return smiles_cache


def update_pdbbind_chemcomp_cache(
    pdb_ids: Iterable[str],
    pdbbind_plus_cache: dict[str, Any],
    chemcomp_cache_path: Path,
    workers: int,
    allow_fetch: bool = False,
) -> dict[str, Any]:
    chemcomp_ids = []
    for pdb_id in pdb_ids:
        record = pdbbind_plus_cache.get(pdb_id) or {}
        info = record.get("info") or {}
        ligand_name = clean_text(info.get("ligandname")).upper()
        if CHEMCOMP_ID_RE.fullmatch(ligand_name):
            chemcomp_ids.append(ligand_name)
    chemcomp_cache = read_json(chemcomp_cache_path, {})
    missing = set(chemcomp_ids) - set(chemcomp_cache)
    if missing and not allow_fetch:
        raise RuntimeError(
            f"PDBbind selected-ligand CCD cache lacks {len(missing)} component IDs; "
            "use --online to fetch them."
        )
    if not allow_fetch:
        return chemcomp_cache
    return update_cache_parallel(
        chemcomp_cache,
        chemcomp_ids,
        get_rcsb_chemcomp,
        chemcomp_cache_path,
        workers,
        "RCSB CCD for PDBbind selected ligands",
    )


def inchikey_connectivity(inchikey: str) -> str:
    return clean_text(inchikey).split("-")[0]


def resolve_pdbbind_ligand_identity(
    pdbbind_smiles: str,
    ligand_name: str,
    pdbbind_smiles_cache: dict[str, Any],
    pdbbind_chemcomp_cache: dict[str, Any],
    asd_inchikey: str,
) -> dict[str, str]:
    smiles_record = pdbbind_smiles_cache.get(pdbbind_smiles) or {}
    smiles_inchikey = clean_text(smiles_record.get("inchikey"))
    smiles_status = clean_text(smiles_record.get("status"))
    chemcomp_id = clean_text(ligand_name).upper()
    chemcomp_record = pdbbind_chemcomp_cache.get(chemcomp_id) or {}
    chemcomp_inchikey = clean_text(chemcomp_record.get("inchikey"))
    chemcomp_status = clean_text(chemcomp_record.get("status"))

    candidates = []
    if smiles_inchikey:
        candidates.append(("pdbbind_plus_smiles", smiles_status or "ok", smiles_inchikey))
    if chemcomp_inchikey:
        candidates.append(("rcsb_chemcomp", chemcomp_status or "ok", chemcomp_inchikey))

    for source, status, inchikey in candidates:
        if asd_inchikey and inchikey == asd_inchikey:
            return {
                "inchikey": inchikey,
                "identity_source": source,
                "identity_status": status,
                "smiles_inchikey": smiles_inchikey,
                "smiles_status": smiles_status,
                "chemcomp_status": chemcomp_status,
                "chemcomp_inchikey": chemcomp_inchikey,
            }

    for source, status, inchikey in candidates:
        if (
            asd_inchikey
            and inchikey_connectivity(asd_inchikey)
            and inchikey_connectivity(asd_inchikey) == inchikey_connectivity(inchikey)
        ):
            return {
                "inchikey": inchikey,
                "identity_source": source,
                "identity_status": status,
                "smiles_inchikey": smiles_inchikey,
                "smiles_status": smiles_status,
                "chemcomp_status": chemcomp_status,
                "chemcomp_inchikey": chemcomp_inchikey,
            }

    if candidates:
        source, status, inchikey = candidates[0]
        return {
            "inchikey": inchikey,
            "identity_source": source,
            "identity_status": status,
            "smiles_inchikey": smiles_inchikey,
            "smiles_status": smiles_status,
            "chemcomp_status": chemcomp_status,
            "chemcomp_inchikey": chemcomp_inchikey,
        }

    return {
        "inchikey": "",
        "identity_source": "",
        "identity_status": smiles_status or chemcomp_status or "missing",
        "smiles_inchikey": smiles_inchikey,
        "smiles_status": smiles_status,
        "chemcomp_status": chemcomp_status,
        "chemcomp_inchikey": chemcomp_inchikey,
    }


def classify_pdbbind_row(
    row: dict[str, Any],
    pdbbind_labels: dict[str, dict[str, str]],
    pdbbind_plus_cache: dict[str, Any],
    pdbbind_smiles_cache: dict[str, Any],
    pdbbind_chemcomp_cache: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pdbbind_v2020_has_pdb_id": "0",
        "pdbbind_v2020_pk": "",
        "pdbbind_plus_status": "",
        "pdbbind_ligandname": "",
        "pdbbind_fullligandname": "",
        "pdbbind_kdtype": "",
        "pdbbind_kdoriginal": "",
        "pdbbind_pkd": "",
        "pdbbind_pubmed": "",
        "pdbbind_reference": "",
        "pdbbind_smiles": "",
        "pdbbind_ligand_inchikey": "",
        "pdbbind_ligand_identity_status": "",
        "pdbbind_ligand_identity_source": "",
        "pdbbind_smiles_inchikey": "",
        "pdbbind_smiles_identity_status": "",
        "pdbbind_chemcomp_status": "",
        "pdbbind_chemcomp_inchikey": "",
        "inchikey_full_match": "0",
        "inchikey_connectivity_match": "0",
        "mapping_tier": "not_mapped",
        "mapping_status": "",
        "verification_basis": "",
    }

    if row.get("domain_expansion_status") != "target_serial_matched":
        result["mapping_status"] = "target_serial_not_found"
        result["verification_basis"] = "ASD modulator related target serial was not found in ASD protein cache"
        return result

    raw_pdb = clean_text(row.get("target_pdb_id")).upper()
    pdb_id = clean_text(row.get("target_valid_pdb_id")).upper()
    if not raw_pdb:
        result["mapping_status"] = "no_target_pdb_id"
        result["verification_basis"] = "ASD online protein row has no PDB ID"
        return result
    if not pdb_id:
        result["mapping_status"] = (
            "target_pdb_field_is_asd_internal_id" if raw_pdb.startswith("ASD") else "invalid_target_pdb_id"
        )
        result["verification_basis"] = "ASD online protein PDB field is not a four-character PDB ID"
        return result
    if pdb_id not in pdbbind_labels:
        result["mapping_status"] = "not_in_pdbbind_v2020"
        result["verification_basis"] = "ASD target PDB ID is absent from local PDBbind v2020 general table"
        return result

    result["pdbbind_v2020_has_pdb_id"] = "1"
    result["pdbbind_v2020_pk"] = clean_text(pdbbind_labels[pdb_id].get("pK"))
    result["mapping_tier"] = "pdbbind_pdb_only_candidate"

    plus_record = pdbbind_plus_cache.get(pdb_id) or {}
    plus_status = clean_text(plus_record.get("status"))
    result["pdbbind_plus_status"] = plus_status
    info = plus_record.get("info") or {}
    for output_key, info_key in (
        ("pdbbind_ligandname", "ligandname"),
        ("pdbbind_fullligandname", "fullligandname"),
        ("pdbbind_kdtype", "kdtype"),
        ("pdbbind_kdoriginal", "KDoriginal"),
        ("pdbbind_pkd", "pkd"),
        ("pdbbind_pubmed", "pubmed"),
        ("pdbbind_reference", "reference"),
        ("pdbbind_smiles", "smilestring"),
    ):
        result[output_key] = clean_text(info.get(info_key))

    if not row.get("ligand_inchikey"):
        result["mapping_status"] = "pdbbind_pdb_overlap_no_asd_ligand_inchikey"
        result["verification_basis"] = "PDB ID overlaps PDBbind, but ASD modulator has no resolved InChIKey"
        return result
    if plus_status != "ok":
        result["mapping_status"] = "pdbbind_pdb_overlap_metadata_unavailable"
        result["verification_basis"] = "PDB ID overlaps PDBbind, but PDBbind+ selected-ligand metadata is unavailable"
        return result
    if not result["pdbbind_smiles"]:
        result["verification_basis"] = "PDBbind+ selected-ligand metadata lacks SMILES; checking CCD fallback"

    identity = resolve_pdbbind_ligand_identity(
        result["pdbbind_smiles"],
        result["pdbbind_ligandname"],
        pdbbind_smiles_cache,
        pdbbind_chemcomp_cache,
        clean_text(row.get("ligand_inchikey")),
    )
    result["pdbbind_ligand_inchikey"] = identity["inchikey"]
    result["pdbbind_ligand_identity_status"] = identity["identity_status"]
    result["pdbbind_ligand_identity_source"] = identity["identity_source"]
    result["pdbbind_smiles_inchikey"] = identity["smiles_inchikey"]
    result["pdbbind_smiles_identity_status"] = identity["smiles_status"]
    result["pdbbind_chemcomp_status"] = identity["chemcomp_status"]
    result["pdbbind_chemcomp_inchikey"] = identity["chemcomp_inchikey"]
    if not result["pdbbind_ligand_inchikey"]:
        result["mapping_status"] = "pdbbind_pdb_overlap_selected_ligand_inchikey_unavailable"
        result["verification_basis"] = (
            "PDBbind selected ligand could not be converted to an InChIKey from "
            "PDBbind+ SMILES or RCSB chemical-component fallback"
        )
        return result

    asd_inchikey = clean_text(row.get("ligand_inchikey"))
    pdbbind_inchikey = result["pdbbind_ligand_inchikey"]
    if asd_inchikey == pdbbind_inchikey:
        result["inchikey_full_match"] = "1"
        result["inchikey_connectivity_match"] = "1"
        result["mapping_tier"] = "pdbbind_exact_selected_ligand_target_pdb"
        result["mapping_status"] = "exact_pdbbind_selected_ligand_match"
        result["verification_basis"] = (
            "same target PDB ID and same full InChIKey as PDBbind-selected ligand "
            f"via {result['pdbbind_ligand_identity_source']}"
        )
        return result

    if inchikey_connectivity(asd_inchikey) and inchikey_connectivity(asd_inchikey) == inchikey_connectivity(
        pdbbind_inchikey
    ):
        result["inchikey_connectivity_match"] = "1"
        result["mapping_status"] = "pdbbind_pdb_overlap_ligand_connectivity_only_match"
        result["verification_basis"] = (
            "same target PDB ID and same InChIKey connectivity block, but full InChIKey differs"
            f" via {result['pdbbind_ligand_identity_source']}"
        )
        return result

    result["mapping_status"] = "pdbbind_pdb_overlap_ligand_mismatch"
    result["verification_basis"] = "same target PDB ID, but ASD ligand and PDBbind-selected ligand InChIKeys differ"
    return result


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def unique_count(rows: Iterable[dict[str, Any]], key: str) -> int:
    return len({clean_text(row.get(key)) for row in rows if clean_text(row.get(key))})


def endpoint_counts(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(clean_text(row.get(key)) or "missing" for row in rows).most_common())


def status_subset(rows: list[dict[str, Any]], predicate) -> list[dict[str, Any]]:
    return [row for row in rows if predicate(row)]


def summarize_mapping(
    *,
    modulators: list[dict[str, Any]],
    proteins: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    exact_rows: list[dict[str, Any]],
    hitlead_rows: list[dict[str, str]],
    hitlead_by_key: dict[tuple[str, str], list[dict[str, str]]],
    hitlead_value_field_name: str,
    pdbbind_labels: dict[str, dict[str, str]],
    build_stats: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    matched_domain_rows = status_subset(
        rows, lambda row: row.get("domain_expansion_status") == "target_serial_matched"
    )
    valid_pdb_rows = status_subset(rows, lambda row: bool(row.get("target_valid_pdb_id")))
    pdbbind_overlap_rows = status_subset(rows, lambda row: row.get("pdbbind_v2020_has_pdb_id") == "1")
    hitlead_status_rows = status_subset(rows, lambda row: row.get("has_hitlead_activity") == "1")
    hitlead_valid_pdb_rows = status_subset(hitlead_status_rows, lambda row: bool(row.get("target_valid_pdb_id")))
    hitlead_pdbbind_rows = status_subset(
        hitlead_status_rows, lambda row: row.get("pdbbind_v2020_has_pdb_id") == "1"
    )
    hitlead_exact_rows = status_subset(
        hitlead_status_rows,
        lambda row: row.get("mapping_tier") == "pdbbind_exact_selected_ligand_target_pdb",
    )
    connectivity_rows = status_subset(rows, lambda row: row.get("inchikey_connectivity_match") == "1")

    hitlead_activity_counter = Counter(clean_text(row.get("Activity_Type")) or "missing" for row in hitlead_rows)
    hitlead_numeric_rows = [row for row in hitlead_rows if numeric(row.get(hitlead_value_field_name))]
    hitlead_strict_rows = [
        row
        for row in hitlead_numeric_rows
        if clean_text(row.get("Activity_Type")) in STRICT_BINDING_TYPES
    ]
    hitlead_keys = set(hitlead_by_key)
    online_hitlead_keys = {
        (clean_text(row.get("target_domain")), clean_text(row.get("asd_modulator_serial")))
        for row in matched_domain_rows
    }

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "asd_online_modulators_json": display_path(args.modulators_json),
            "asd_online_proteins_json": display_path(args.proteins_json),
            "asd_smiles_inchikey_cache": display_path(args.asd_smiles_cache),
            "asd_hitlead_archive": display_path(args.hitlead_archive),
            "pdbbind_v2020_csv": display_path(args.pdbbind_csv),
            "pdbbind_plus_cache": display_path(args.pdbbind_plus_cache),
            "pdbbind_selected_ligand_smiles_cache": display_path(args.pdbbind_smiles_cache),
            "pdbbind_selected_ligand_chemcomp_cache": display_path(args.pdbbind_chemcomp_cache),
        },
        "online_asd": {
            "modulators": len(modulators),
            "modulators_with_smiles": sum(1 for row in modulators if row.get("smiles")),
            "modulators_with_inchikey": sum(1 for row in modulators if row.get("ligand_inchikey")),
            "proteins": len(proteins),
            "unique_protein_db_serials": unique_count(proteins, "db_serial"),
            "base_related_items_total": build_stats.get("base_related_items_total", 0),
            "base_related_items_with_target_serial_match": build_stats.get(
                "related_items_with_target_serial_match", 0
            ),
            "base_related_items_without_target_serial_match": build_stats.get(
                "related_items_without_target_serial_match", 0
            ),
            "base_related_items_with_multiple_domain_matches": build_stats.get(
                "related_items_with_multiple_domain_matches", 0
            ),
            "domain_expanded_status_rows": len(rows),
            "domain_expanded_rows_with_target_serial_match": len(matched_domain_rows),
            "domain_expanded_unique_target_domains": unique_count(matched_domain_rows, "target_domain"),
            "domain_expanded_unique_uniprots": unique_count(matched_domain_rows, "uniprot"),
        },
        "hitlead": {
            "rows": len(hitlead_rows),
            "unique_domain_modulator_keys": len(hitlead_keys),
            "unique_modulators": unique_count(hitlead_rows, "Modulator_ID"),
            "unique_target_domains": unique_count(hitlead_rows, "Target_ID"),
            "numeric_activity_rows": len(hitlead_numeric_rows),
            "strict_binding_kd_ki_rows": len(hitlead_strict_rows),
            "endpoint_counts": dict(hitlead_activity_counter.most_common()),
            "domain_modulator_keys_found_in_online_domain_expansion": len(
                hitlead_keys & online_hitlead_keys
            ),
            "domain_modulator_keys_missing_from_online_domain_expansion": len(
                hitlead_keys - online_hitlead_keys
            ),
            "online_domain_rows_with_hitlead_activity": len(hitlead_status_rows),
            "online_domain_rows_with_hitlead_numeric_activity": sum(
                int(clean_text(row.get("hitlead_numeric_activity_row_count")) or 0) > 0
                for row in hitlead_status_rows
            ),
            "online_domain_rows_with_hitlead_strict_binding": sum(
                int(clean_text(row.get("hitlead_strict_binding_row_count")) or 0) > 0
                for row in hitlead_status_rows
            ),
            "online_hitlead_unique_modulators": unique_count(hitlead_status_rows, "asd_modulator_serial"),
            "online_hitlead_unique_target_domains": unique_count(hitlead_status_rows, "target_domain"),
        },
        "pdbbind_mapping": {
            "pdbbind_v2020_unique_pdb_ids": len(pdbbind_labels),
            "rows_with_valid_target_pdb_id": len(valid_pdb_rows),
            "unique_valid_target_pdb_ids": unique_count(valid_pdb_rows, "target_valid_pdb_id"),
            "rows_with_pdbbind_v2020_pdb_id_overlap": len(pdbbind_overlap_rows),
            "unique_pdbbind_v2020_overlap_pdb_ids": unique_count(
                pdbbind_overlap_rows, "target_valid_pdb_id"
            ),
            "compounds_with_pdbbind_v2020_pdb_id_overlap": unique_count(
                pdbbind_overlap_rows, "asd_modulator_serial"
            ),
            "base_related_items_with_pdbbind_v2020_pdb_id_overlap": unique_count(
                pdbbind_overlap_rows, "base_related_item_id"
            ),
            "exact_selected_ligand_rows": len(exact_rows),
            "exact_selected_ligand_unique_compounds": unique_count(exact_rows, "asd_modulator_serial"),
            "exact_selected_ligand_unique_target_domains": unique_count(exact_rows, "target_domain"),
            "exact_selected_ligand_unique_pdb_ids": unique_count(exact_rows, "target_valid_pdb_id"),
            "exact_selected_ligand_unique_uniprots": unique_count(exact_rows, "uniprot"),
            "connectivity_match_rows_including_full_matches": len(connectivity_rows),
            "exact_selected_ligand_endpoint_counts": endpoint_counts(exact_rows, "pdbbind_kdtype"),
            "exact_selected_ligand_strict_kd_ki_rows": sum(
                row.get("pdbbind_kdtype") in STRICT_BINDING_TYPES for row in exact_rows
            ),
            "pdbbind_overlap_ligand_identity_source_counts": dict(
                Counter(
                    clean_text(row.get("pdbbind_ligand_identity_source")) or "unavailable"
                    for row in pdbbind_overlap_rows
                ).most_common()
            ),
            "exact_selected_ligand_identity_source_counts": dict(
                Counter(
                    clean_text(row.get("pdbbind_ligand_identity_source")) or "unavailable"
                    for row in exact_rows
                ).most_common()
            ),
            "mapping_status_counts": dict(
                Counter(row.get("mapping_status", "missing") for row in rows).most_common()
            ),
            "mapping_tier_counts": dict(
                Counter(row.get("mapping_tier", "missing") for row in rows).most_common()
            ),
        },
        "hitlead_pdbbind_overlap": {
            "hitlead_domain_rows": len(hitlead_status_rows),
            "hitlead_rows_with_valid_target_pdb_id": len(hitlead_valid_pdb_rows),
            "hitlead_rows_with_pdbbind_v2020_pdb_id_overlap": len(hitlead_pdbbind_rows),
            "hitlead_unique_pdbbind_overlap_pdb_ids": unique_count(
                hitlead_pdbbind_rows, "target_valid_pdb_id"
            ),
            "hitlead_compounds_with_pdbbind_v2020_pdb_id_overlap": unique_count(
                hitlead_pdbbind_rows, "asd_modulator_serial"
            ),
            "hitlead_exact_selected_ligand_rows": len(hitlead_exact_rows),
            "hitlead_exact_selected_ligand_unique_compounds": unique_count(
                hitlead_exact_rows, "asd_modulator_serial"
            ),
            "hitlead_exact_selected_ligand_unique_pdb_ids": unique_count(
                hitlead_exact_rows, "target_valid_pdb_id"
            ),
            "hitlead_exact_endpoint_counts": endpoint_counts(hitlead_exact_rows, "pdbbind_kdtype"),
        },
        "outputs": {
            "status_tsv": display_path(args.status_tsv),
            "exact_labels_tsv": display_path(args.exact_tsv),
            "summary_json": display_path(args.summary_json),
            "summary_md": display_path(args.summary_md),
        },
        "notes": [
            "ASD online related-protein items are expanded through ASD protein db_serial, not the numeric related item field.",
            "PDBbind mapping by PDB ID alone is a candidate label only.",
            "Exact selected-ligand rows require full InChIKey equality between the ASD modulator and the PDBbind-selected ligand identity from PDBbind+ SMILES or RCSB CCD fallback.",
            "Exact selected-ligand rows are Tier-A-like for the online table, but weaker than AS-archive Tier A because the online modulator table does not provide chain/residue ligand-instance evidence.",
            "Hit-to-Lead labels are joined by exact ASD target domain and modulator serial.",
            "Row-level TSVs stay under data/interim and are not versioned.",
        ],
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Complete ASD Online Mapping To PDBbind v2020",
        "",
        f"Created: `{summary['created_at']}`",
        "",
        "## Question",
        "",
        (
            "Across the complete ASD online modulator universe, including the ASD Hit-to-Lead "
            "activity subset, how many domain-expanded ASD modulator-target rows can be mapped "
            "to PDBbind v2020?"
        ),
        "",
        "## Corrected ASD Online Expansion",
        "",
        (
            "This pass expands ASD online modulator related-target items through the ASD protein "
            "`db_serial`/`domain` fields. The numeric field in the modulator related-protein list "
            "is not used as the protein join key because spot checks showed it can point to an "
            "unrelated protein row."
        ),
        "",
    ]
    for key, value in summary["online_asd"].items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Hit-to-Lead Coverage", ""])
    for key, value in summary["hitlead"].items():
        if isinstance(value, dict):
            continue
        lines.append(f"- {key}: {value}")
    lines.extend(["", "Hit-to-Lead endpoint counts:", ""])
    for endpoint, count in summary["hitlead"]["endpoint_counts"].items():
        lines.append(f"- `{endpoint}`: {count}")

    lines.extend(["", "## PDBbind Mapping", ""])
    for key, value in summary["pdbbind_mapping"].items():
        if isinstance(value, dict):
            continue
        lines.append(f"- {key}: {value}")
    lines.extend(["", "Mapping status counts:", ""])
    for status, count in summary["pdbbind_mapping"]["mapping_status_counts"].items():
        lines.append(f"- `{status}`: {count}")
    lines.extend(["", "Exact selected-ligand endpoint counts:", ""])
    for endpoint, count in summary["pdbbind_mapping"]["exact_selected_ligand_endpoint_counts"].items():
        lines.append(f"- `{endpoint}`: {count}")
    lines.extend(["", "Exact selected-ligand identity sources:", ""])
    for source, count in summary["pdbbind_mapping"]["exact_selected_ligand_identity_source_counts"].items():
        lines.append(f"- `{source}`: {count}")

    lines.extend(["", "## Hit-to-Lead And PDBbind", ""])
    for key, value in summary["hitlead_pdbbind_overlap"].items():
        if isinstance(value, dict):
            continue
        lines.append(f"- {key}: {value}")
    lines.extend(["", "Hit-to-Lead exact PDBbind endpoint counts:", ""])
    for endpoint, count in summary["hitlead_pdbbind_overlap"]["hitlead_exact_endpoint_counts"].items():
        lines.append(f"- `{endpoint}`: {count}")

    lines.extend(["", "## Outputs", ""])
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "Use the PDB-ID overlap counts as broad structure-label candidates only. "
        "For model validation, the stricter exact selected-ligand rows are the useful subset, "
        "because they verify that the ASD compound and the PDBbind-selected ligand are the same "
        "molecule by full InChIKey."
    )
    lines.append("")
    lines.append(
        "These exact online rows are still weaker than the earlier AS-archive Tier-A set: "
        "the online table links a modulator to a target/domain/PDB, but does not provide the "
        "chain/residue ligand-instance evidence available in the ASD allosteric-site complex archive."
    )
    lines.extend(["", "## Notes", ""])
    for note in summary["notes"]:
        lines.append(f"- {note}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


BASE_FIELDNAMES = [
    "asd_online_pdbbind_status_id",
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
    "related_numeric_id",
    "related_target_serial",
    "target_serial_match_count",
    "domain_expansion_status",
    "target_mol_id",
    "target_domain",
    "target_db_serial",
    "target_name",
    "target_gene",
    "organism",
    "uniprot",
    "swissport_id",
    "trembl_id",
    "target_pdb_id",
    "target_valid_pdb_id",
    "target_pubmed_id",
]

HITLEAD_FIELDNAMES = [
    "has_hitlead_activity",
    "hitlead_row_count",
    "hitlead_numeric_activity_row_count",
    "hitlead_strict_binding_row_count",
    "hitlead_activity_types",
    "hitlead_activity_values_uM",
    "hitlead_pvalues",
    "hitlead_pubmed_ids",
]

PDBBIND_FIELDNAMES = [
    "pdbbind_v2020_has_pdb_id",
    "pdbbind_v2020_pk",
    "pdbbind_plus_status",
    "pdbbind_ligandname",
    "pdbbind_fullligandname",
    "pdbbind_kdtype",
    "pdbbind_kdoriginal",
    "pdbbind_pkd",
    "pdbbind_pubmed",
    "pdbbind_reference",
    "pdbbind_smiles",
    "pdbbind_ligand_inchikey",
    "pdbbind_ligand_identity_status",
    "pdbbind_ligand_identity_source",
    "pdbbind_smiles_inchikey",
    "pdbbind_smiles_identity_status",
    "pdbbind_chemcomp_status",
    "pdbbind_chemcomp_inchikey",
    "inchikey_full_match",
    "inchikey_connectivity_match",
    "mapping_tier",
    "mapping_status",
    "verification_basis",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modulators-json", type=Path, default=DEFAULT_ASD_MODULATORS_JSON)
    parser.add_argument("--proteins-json", type=Path, default=DEFAULT_ASD_PROTEINS_JSON)
    parser.add_argument("--asd-smiles-cache", type=Path, default=DEFAULT_ASD_SMILES_CACHE)
    parser.add_argument("--hitlead-archive", type=Path, default=DEFAULT_HITLEAD_ARCHIVE)
    parser.add_argument("--pdbbind-csv", type=Path, default=DEFAULT_PDBBIND_CSV)
    parser.add_argument("--pdbbind-plus-cache", type=Path, default=DEFAULT_PDBBIND_PLUS_CACHE)
    parser.add_argument("--pdbbind-smiles-cache", type=Path, default=DEFAULT_PDBBIND_SMILES_CACHE)
    parser.add_argument("--pdbbind-chemcomp-cache", type=Path, default=DEFAULT_PDBBIND_CHEMCOMP_CACHE)
    parser.add_argument("--status-tsv", type=Path, default=DEFAULT_STATUS_TSV)
    parser.add_argument("--exact-tsv", type=Path, default=DEFAULT_EXACT_TSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--online",
        action="store_true",
        help="Fetch missing PDBbind+ and RCSB selected-ligand metadata.",
    )
    parser.add_argument(
        "--skip-pdbbind-plus-fetch",
        action="store_true",
        help="Use only the existing PDBbind+ cache; missing selected-ligand metadata stays unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    modulators = [normalize_modulator(row) for row in read_online_rows(args.modulators_json, "modulators")]
    proteins = [normalize_protein(row) for row in read_online_rows(args.proteins_json, "proteins")]
    smiles_cache = update_smiles_cache(modulators, args.asd_smiles_cache)
    add_ligand_identity(modulators, smiles_cache)

    rows, build_stats = build_domain_interactions(modulators, proteins)
    print(f"ASD online domain-expanded rows: {len(rows)}")

    hitlead_rows = read_hitlead_rows(args.hitlead_archive)
    hitlead_value_field_name = hitlead_value_field(hitlead_rows)
    hitlead_by_key = build_hitlead_index(hitlead_rows)
    for row in rows:
        key = (clean_text(row.get("target_domain")), clean_text(row.get("asd_modulator_serial")))
        row.update(summarize_hitlead_rows(hitlead_by_key.get(key, []), hitlead_value_field_name))

    pdbbind_labels = pdbbind_by_pdb(args.pdbbind_csv)
    pdbbind_overlap_pdb_ids = {
        row["target_valid_pdb_id"]
        for row in rows
        if row.get("target_valid_pdb_id") and row["target_valid_pdb_id"] in pdbbind_labels
    }
    pdbbind_plus_cache = read_json(args.pdbbind_plus_cache, {})
    missing_plus = pdbbind_overlap_pdb_ids - set(pdbbind_plus_cache)
    if missing_plus and not args.skip_pdbbind_plus_fetch and not args.online:
        raise RuntimeError(
            f"PDBbind+ cache lacks {len(missing_plus)} PDB IDs; use --online to fetch them."
        )
    if args.skip_pdbbind_plus_fetch or not args.online:
        print(f"PDBbind+ browser: using existing cache ({len(pdbbind_plus_cache)} entries)")
    else:
        pdbbind_plus_cache = update_cache_parallel(
            pdbbind_plus_cache,
            pdbbind_overlap_pdb_ids,
            fetch_pdbbind_plus_record,
            args.pdbbind_plus_cache,
            args.workers,
            "PDBbind+ browser",
        )
    pdbbind_smiles_cache = update_pdbbind_ligand_smiles_cache(
        pdbbind_overlap_pdb_ids,
        pdbbind_plus_cache,
        args.pdbbind_smiles_cache,
    )
    pdbbind_chemcomp_cache = update_pdbbind_chemcomp_cache(
        pdbbind_overlap_pdb_ids,
        pdbbind_plus_cache,
        args.pdbbind_chemcomp_cache,
        args.workers,
        args.online,
    )

    for row in rows:
        row.update(
            classify_pdbbind_row(
                row,
                pdbbind_labels,
                pdbbind_plus_cache,
                pdbbind_smiles_cache,
                pdbbind_chemcomp_cache,
            )
        )

    exact_rows = [
        row for row in rows if row.get("mapping_tier") == "pdbbind_exact_selected_ligand_target_pdb"
    ]
    fieldnames = BASE_FIELDNAMES + HITLEAD_FIELDNAMES + PDBBIND_FIELDNAMES
    write_tsv(args.status_tsv, rows, fieldnames)
    write_tsv(args.exact_tsv, exact_rows, fieldnames)

    summary = summarize_mapping(
        modulators=modulators,
        proteins=proteins,
        rows=rows,
        exact_rows=exact_rows,
        hitlead_rows=hitlead_rows,
        hitlead_by_key=hitlead_by_key,
        hitlead_value_field_name=hitlead_value_field_name,
        pdbbind_labels=pdbbind_labels,
        build_stats=build_stats,
        args=args,
    )
    write_json(args.summary_json, summary)
    write_summary_md(args.summary_md, summary)
    print(f"Wrote status rows: {display_path(args.status_tsv)}")
    print(f"Wrote exact PDBbind labels: {display_path(args.exact_tsv)}")
    print(f"Wrote summary: {display_path(args.summary_md)}")


if __name__ == "__main__":
    main()
