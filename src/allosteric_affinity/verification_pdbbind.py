#!/usr/bin/env python3
"""Verify ASD/PDBbind v2020 matches at ligand level.

The earlier PDBbind recovery pass joined ASD allosteric-site rows to PDBbind by
PDB ID only. This script tightens that into a Tier-A verification pass by
checking the PDBbind-selected ligand against the ASD modulator and the RCSB
entry contents.

Row-level output is intentionally written under data/interim/ because ASD and
PDBbind redistribution terms require caution. Local summary output is aggregate only.
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
DEFAULT_PDBBIND_CSV = REPO_ROOT / "DrugWise-Implementation-main/data/PDBbindv2020_General.csv"
DEFAULT_CACHE_DIR = REPO_ROOT / "data/interim/pdbbind"
DEFAULT_STATUS_TSV = DEFAULT_CACHE_DIR / "asd_pdbbind_v2020_tier_a_status.tsv"
DEFAULT_SUMMARY_JSON = REPO_ROOT / "outputs/discovery/asd_pdbbind_tier_a_verification_summary.json"
DEFAULT_SUMMARY_MD = REPO_ROOT / "outputs/discovery/asd_pdbbind_tier_a_verification_summary.md"

PDBBIND_PLUS_BROWSER_URL = "https://www.pdbbind-plus.org.cn/api/browser"
RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
RCSB_NONPOLYMER_ENTITY_URL = (
    "https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_id}/{entity_id}"
)
RCSB_NONPOLYMER_INSTANCE_URL = (
    "https://data.rcsb.org/rest/v1/core/nonpolymer_entity_instance/{pdb_id}/{asym_id}"
)
RCSB_POLYMER_ENTITY_URL = (
    "https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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


def request_json(url: str, *, params: dict[str, str] | None = None, attempts: int = 3) -> Any:
    import requests

    headers = {"Accept": "application/json", "User-Agent": "asd-pdbbind-tier-a/0.1"}
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=45)
            if response.status_code == 404:
                return {"http_status": 404, "status": "missing"}
            if response.status_code in {429, 500, 502, 503, 504}:
                time.sleep(1.5 * (attempt + 1))
                continue
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            if attempt == attempts - 1:
                return {"status": "error", "error": str(exc)}
            time.sleep(1.5 * (attempt + 1))
    return {"status": "error", "error": "unknown request failure"}


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
            except Exception as exc:  # defensive; request_json already catches request failures
                cache[key] = {"status": "error", "error": str(exc)}
            completed += 1
            if completed % 50 == 0:
                write_json(cache_path, cache)
                print(f"{label}: fetched {completed}/{len(missing)}")
    write_json(cache_path, cache)
    return cache


def fetch_pdbbind_plus_record(pdb_id: str) -> dict[str, Any]:
    data = request_json(PDBBIND_PLUS_BROWSER_URL, params={"pdbcode": pdb_id})
    if not isinstance(data, dict):
        return {"status": "error", "error": "non-dict response"}
    code = data.get("code")
    if code != 0:
        return {
            "status": "unavailable",
            "code": code,
            "msg": data.get("msg", ""),
        }

    info = (data.get("data") or {}).get("info") or {}
    returned_pdb = str(info.get("PDBcode", "")).strip().upper()
    if returned_pdb != pdb_id.upper():
        return {
            "status": "pdb_mismatch",
            "requested_pdb": pdb_id.upper(),
            "returned_pdb": returned_pdb,
            "info": info,
        }
    return {
        "status": "ok",
        "info": {
            "PDBcode": returned_pdb,
            "ligandname": info.get("ligandname"),
            "fullligandname": info.get("fullligandname"),
            "KDoriginal": info.get("KDoriginal"),
            "pkd": info.get("pkd"),
            "kdtype": info.get("kdtype"),
            "pubmed": info.get("pubmed"),
            "reference": info.get("reference"),
            "settype": info.get("settype"),
            "settypeold": info.get("settypeold"),
            "CompoundType": info.get("CompoundType"),
            "LigandSequence": info.get("LigandSequence"),
            "smilestring": info.get("smilestring"),
            "Formula": info.get("Formula"),
            "comments": info.get("comments"),
            "originalref": info.get("originalref"),
        },
    }


def fetch_rcsb_ligand_evidence(pdb_id: str) -> dict[str, Any]:
    entry = request_json(RCSB_ENTRY_URL.format(pdb_id=pdb_id))
    if not isinstance(entry, dict) or entry.get("status") in {"error", "missing"}:
        return {
            "status": entry.get("status", "error") if isinstance(entry, dict) else "error",
            "entry": entry,
            "nonpolymer_instances": [],
            "short_polymer_entities": [],
        }

    identifiers = entry.get("rcsb_entry_container_identifiers") or {}
    result: dict[str, Any] = {
        "status": "ok",
        "pubmed_id": identifiers.get("pubmed_id"),
        "entry_id": identifiers.get("entry_id"),
        "nonpolymer_instances": [],
        "short_polymer_entities": [],
    }

    for entity_id in identifiers.get("non_polymer_entity_ids") or []:
        entity = request_json(
            RCSB_NONPOLYMER_ENTITY_URL.format(pdb_id=pdb_id, entity_id=entity_id)
        )
        if not isinstance(entity, dict) or entity.get("status") in {"error", "missing"}:
            continue
        container = entity.get("rcsb_nonpolymer_entity_container_identifiers") or {}
        comp_id = (
            container.get("nonpolymer_comp_id")
            or container.get("chem_ref_def_id")
            or (entity.get("pdbx_entity_nonpoly") or {}).get("comp_id")
        )
        comp_id = str(comp_id or "").upper()
        entity_description = (
            (entity.get("pdbx_entity_nonpoly") or {}).get("name")
            or (entity.get("rcsb_nonpolymer_entity") or {}).get("pdbx_description")
            or ""
        )
        asym_ids = container.get("asym_ids") or []
        if not asym_ids:
            result["nonpolymer_instances"].append(
                {
                    "entity_id": str(entity_id),
                    "comp_id": comp_id,
                    "asym_id": "",
                    "auth_asym_id": ";".join(container.get("auth_asym_ids") or []),
                    "auth_seq_id": "",
                    "description": entity_description,
                }
            )
            continue
        for asym_id in asym_ids:
            instance = request_json(
                RCSB_NONPOLYMER_INSTANCE_URL.format(pdb_id=pdb_id, asym_id=asym_id)
            )
            instance_container = {}
            if isinstance(instance, dict):
                instance_container = (
                    instance.get("rcsb_nonpolymer_entity_instance_container_identifiers")
                    or {}
                )
            result["nonpolymer_instances"].append(
                {
                    "entity_id": str(entity_id),
                    "comp_id": str(instance_container.get("comp_id") or comp_id).upper(),
                    "asym_id": str(instance_container.get("asym_id") or asym_id),
                    "auth_asym_id": str(instance_container.get("auth_asym_id") or ""),
                    "auth_seq_id": str(instance_container.get("auth_seq_id") or ""),
                    "description": entity_description,
                }
            )

    for entity_id in identifiers.get("polymer_entity_ids") or []:
        entity = request_json(RCSB_POLYMER_ENTITY_URL.format(pdb_id=pdb_id, entity_id=entity_id))
        if not isinstance(entity, dict) or entity.get("status") in {"error", "missing"}:
            continue
        container = entity.get("rcsb_polymer_entity_container_identifiers") or {}
        entity_poly = entity.get("entity_poly") or {}
        sequence = str(entity_poly.get("pdbx_seq_one_letter_code_can") or "").replace("\n", "")
        if len(sequence) <= 40:
            result["short_polymer_entities"].append(
                {
                    "entity_id": str(entity_id),
                    "sequence": sequence,
                    "sequence_length": len(sequence),
                    "asym_ids": ";".join(container.get("asym_ids") or []),
                    "auth_asym_ids": ";".join(container.get("auth_asym_ids") or []),
                    "chem_comp_monomers": ";".join(container.get("chem_comp_monomers") or []),
                    "chem_comp_nstd_monomers": ";".join(
                        container.get("chem_comp_nstd_monomers") or []
                    ),
                }
            )
    return result


def split_aliases(alias: str) -> list[str]:
    tokens = re.split(r";|@@|,|\s+|\|", alias or "")
    normalized = []
    for token in tokens:
        token = token.strip().upper()
        if not token or token.startswith("CHAIN"):
            continue
        normalized.append(token)
    return sorted(set(normalized))


def is_component_like(value: str) -> bool:
    value = value.strip().upper()
    return bool(re.fullmatch(r"[A-Z0-9]{1,6}", value))


def peptide_length_from_name(value: str) -> int | None:
    match = re.fullmatch(r"(\d+)-MER", value.strip().upper())
    if not match:
        return None
    return int(match.group(1))


def split_field_values(value: str) -> set[str]:
    return {
        token.strip().upper()
        for token in re.split(r"[;,|\s]+", value or "")
        if token.strip()
    }


def chain_matches(value: str, auth_asym_ids: str) -> bool:
    chains = split_field_values(value)
    if not chains:
        return False
    return bool(chains & split_field_values(auth_asym_ids))


def classify_match(
    asd_row: dict[str, str],
    row_index: int,
    pdbbind_pk: str,
    pdbbind_record: dict[str, Any],
    rcsb_evidence: dict[str, Any],
) -> dict[str, str]:
    pdb_id = asd_row.get("allosteric_pdb", "").strip().upper()
    alias_tokens = split_aliases(asd_row.get("modulator_alias", ""))
    modulator_class = asd_row.get("modulator_class", "")
    modulator_chain = asd_row.get("modulator_chain", "").strip().upper()
    modulator_resi = asd_row.get("modulator_resi", "").strip()
    modulator_chains = split_field_values(modulator_chain)
    modulator_resis = split_field_values(modulator_resi)

    info = pdbbind_record.get("info") or {}
    pdbbind_status = pdbbind_record.get("status", "")
    ligand_name = str(info.get("ligandname") or "").strip()
    ligand_token = ligand_name.upper()
    ligand_component_like = is_component_like(ligand_token)
    peptide_length = peptide_length_from_name(ligand_token)

    nonpolymer_instances = rcsb_evidence.get("nonpolymer_instances") or []
    nonpolymer_comp_ids = sorted(
        {str(item.get("comp_id", "")).upper() for item in nonpolymer_instances if item.get("comp_id")}
    )
    short_polymers = rcsb_evidence.get("short_polymer_entities") or []

    chain_residue_match = "not_checked"
    if ligand_component_like and ligand_token in nonpolymer_comp_ids:
        matching_instances = [
            item for item in nonpolymer_instances if str(item.get("comp_id", "")).upper() == ligand_token
        ]
        if modulator_chain or modulator_resi:
            chain_residue_match = "no"
            for item in matching_instances:
                chain_ok = (
                    not modulator_chains
                    or str(item.get("auth_asym_id", "")).upper() in modulator_chains
                )
                resi_ok = (
                    not modulator_resis
                    or str(item.get("auth_seq_id", "")).upper() in modulator_resis
                )
                if chain_ok and resi_ok:
                    chain_residue_match = "yes"
                    break
        else:
            chain_residue_match = "not_applicable"

    status = "not_tier_a_pdbbind_metadata_unavailable"
    tier = "not_tier_a"
    verification_basis = ""

    if pdbbind_status != "ok":
        if len(nonpolymer_comp_ids) == 1 and nonpolymer_comp_ids[0] in alias_tokens:
            status = "tier_a_inferred_single_rcsb_nonpolymer_component"
            tier = "tier_a_candidate_inferred"
            verification_basis = (
                "PDBbind+ selected-ligand metadata was unavailable, but RCSB reports one "
                "nonpolymer component and it matches the ASD modulator alias."
            )
        else:
            status = f"not_tier_a_pdbbind_metadata_{pdbbind_status or 'unavailable'}"
            verification_basis = "PDBbind+ did not provide selected-ligand metadata for this PDB ID."
    elif ligand_component_like:
        if ligand_token not in nonpolymer_comp_ids:
            status = "not_tier_a_pdbbind_selected_component_not_in_rcsb_entry"
            verification_basis = (
                "PDBbind+ selected ligand is component-like, but RCSB nonpolymer evidence "
                "does not contain that component."
            )
        elif ligand_token not in alias_tokens:
            status = "not_tier_a_pdbbind_selected_ligand_differs_from_asd_modulator"
            verification_basis = (
                "PDBbind+ selected ligand component exists in RCSB, but it is not one of "
                "the ASD modulator aliases for this row."
            )
        elif chain_residue_match == "yes":
            status = "tier_a_verified_nonpolymer_component_instance"
            tier = "tier_a"
            verification_basis = (
                "PDBbind+ selected ligand component matches ASD alias and an RCSB "
                "nonpolymer instance matches the ASD author chain/residue."
            )
        else:
            status = "tier_a_verified_nonpolymer_component_entry"
            tier = "tier_a"
            verification_basis = (
                "PDBbind+ selected ligand component matches ASD alias and is present as "
                "an RCSB nonpolymer component in the same PDB entry."
            )
    elif peptide_length is not None and "PEP" in modulator_class.upper():
        peptide_candidates = [
            item
            for item in short_polymers
            if int(item.get("sequence_length") or 0) == peptide_length
            and (not modulator_chain or chain_matches(modulator_chain, item.get("auth_asym_ids", "")))
        ]
        if len(peptide_candidates) == 1:
            status = "tier_a_verified_peptide_polymer_chain_length"
            tier = "tier_a"
            verification_basis = (
                "PDBbind+ selected ligand is an n-mer peptide, ASD row is a peptide "
                "modulator, and exactly one short RCSB polymer ligand candidate matches "
                "the peptide length and ASD chain evidence."
            )
            chain_residue_match = "yes" if modulator_chain else "not_applicable"
        else:
            status = "not_tier_a_peptide_ligand_not_uniquely_resolved"
            verification_basis = (
                "PDBbind+ selected ligand is an n-mer peptide, but RCSB/ASD chain-length "
                "evidence did not identify one unique peptide ligand."
            )
    else:
        status = "not_tier_a_pdbbind_selected_ligand_not_component_or_verified_peptide"
        verification_basis = (
            "PDBbind+ selected ligand name is not an RCSB component-style identifier "
            "and did not satisfy the peptide verification rule."
        )

    return {
        "asd_row_index": str(row_index),
        "target_id": asd_row.get("target_id", ""),
        "target_gene": asd_row.get("target_gene", ""),
        "organism": asd_row.get("organism", ""),
        "pdb_uniprot": asd_row.get("pdb_uniprot", ""),
        "allosteric_pdb": pdb_id,
        "modulator_alias": asd_row.get("modulator_alias", ""),
        "modulator_alias_tokens": ";".join(alias_tokens),
        "modulator_chain": asd_row.get("modulator_chain", ""),
        "modulator_resi": asd_row.get("modulator_resi", ""),
        "modulator_class": modulator_class,
        "modulator_name": asd_row.get("modulator_name", ""),
        "asd_pubmed_id": asd_row.get("pubmed_id", ""),
        "pdbbind_v2020_pk": pdbbind_pk,
        "pdbbind_plus_status": pdbbind_status,
        "pdbbind_plus_message": str(pdbbind_record.get("msg") or pdbbind_record.get("error") or ""),
        "pdbbind_ligandname": ligand_name,
        "pdbbind_kdtype": str(info.get("kdtype") or ""),
        "pdbbind_kdoriginal": str(info.get("KDoriginal") or ""),
        "pdbbind_pkd": str(info.get("pkd") or ""),
        "pdbbind_pubmed": str(info.get("pubmed") or ""),
        "pdbbind_settype": str(info.get("settype") or ""),
        "pdbbind_settypeold": str(info.get("settypeold") or ""),
        "rcsb_pubmed_id": str(rcsb_evidence.get("pubmed_id") or ""),
        "rcsb_nonpolymer_comp_ids": ";".join(nonpolymer_comp_ids),
        "rcsb_short_polymer_ligands": ";".join(
            f"{item.get('auth_asym_ids','')}:{item.get('sequence','')}"
            for item in short_polymers
        ),
        "chain_residue_match": chain_residue_match,
        "tier": tier,
        "status": status,
        "verification_basis": verification_basis,
    }


def summarize_status_rows(status_rows: list[dict[str, str]], inputs: dict[str, str]) -> dict[str, Any]:
    tier_a_rows = [row for row in status_rows if row["tier"] == "tier_a"]
    inferred_rows = [row for row in status_rows if row["tier"] == "tier_a_candidate_inferred"]
    not_tier_a_rows = [row for row in status_rows if row["tier"] == "not_tier_a"]

    def unique_count(rows: list[dict[str, str]], field: str) -> int:
        return len({row[field] for row in rows if row.get(field)})

    status_counts = Counter(row["status"] for row in status_rows)
    tier_counts = Counter(row["tier"] for row in status_rows)
    endpoint_counts = Counter(row["pdbbind_kdtype"] or "unknown" for row in status_rows)
    tier_endpoint_counts: dict[str, dict[str, int]] = {}
    for tier, rows in (
        ("tier_a", tier_a_rows),
        ("tier_a_candidate_inferred", inferred_rows),
        ("not_tier_a", not_tier_a_rows),
    ):
        tier_endpoint_counts[tier] = dict(Counter(row["pdbbind_kdtype"] or "unknown" for row in rows))

    by_pdb: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in status_rows:
        by_pdb[row["allosteric_pdb"]].append(row)
    pdb_status_counts = Counter()
    for pdb_id, rows in by_pdb.items():
        tiers = {row["tier"] for row in rows}
        if "tier_a" in tiers:
            pdb_status_counts["has_tier_a_row"] += 1
        elif "tier_a_candidate_inferred" in tiers:
            pdb_status_counts["has_inferred_tier_a_candidate_row_only"] += 1
        else:
            pdb_status_counts["no_tier_a_row"] += 1

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "scope": {
            "matched_asd_rows_evaluated": len(status_rows),
            "matched_unique_pdb_ids_evaluated": unique_count(status_rows, "allosteric_pdb"),
            "matched_unique_target_ids_evaluated": unique_count(status_rows, "target_id"),
        },
        "tier_counts_by_asd_row": dict(tier_counts),
        "status_counts_by_asd_row": dict(status_counts),
        "pdb_counts": dict(pdb_status_counts),
        "tier_a": {
            "asd_rows": len(tier_a_rows),
            "unique_pdb_ids": unique_count(tier_a_rows, "allosteric_pdb"),
            "unique_target_ids": unique_count(tier_a_rows, "target_id"),
            "status_counts": dict(Counter(row["status"] for row in tier_a_rows)),
        },
        "tier_a_candidate_inferred": {
            "asd_rows": len(inferred_rows),
            "unique_pdb_ids": unique_count(inferred_rows, "allosteric_pdb"),
            "unique_target_ids": unique_count(inferred_rows, "target_id"),
            "status_counts": dict(Counter(row["status"] for row in inferred_rows)),
        },
        "not_tier_a": {
            "asd_rows": len(not_tier_a_rows),
            "unique_pdb_ids": unique_count(not_tier_a_rows, "allosteric_pdb"),
            "unique_target_ids": unique_count(not_tier_a_rows, "target_id"),
            "status_counts": dict(Counter(row["status"] for row in not_tier_a_rows)),
        },
        "pdbbind_endpoint_counts_by_asd_row": dict(endpoint_counts),
        "pdbbind_endpoint_counts_by_tier": tier_endpoint_counts,
        "notes": [
            "Tier A rows require ligand-level evidence, not only PDB-ID overlap.",
            "tier_a_candidate_inferred rows are separated from Tier A because PDBbind+ selected-ligand metadata was unavailable, although RCSB reports one nonpolymer component matching the ASD modulator.",
            "The local row-level TSV is written under data/interim and is not versioned because it contains ASD/PDBbind-derived row data.",
        ],
    }


def write_status_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict[str, Any], status_tsv: Path) -> None:
    lines = [
        "# ASD/PDBbind Tier-A Ligand Verification",
        "",
        f"Generated at UTC: `{summary['generated_at_utc']}`.",
        "",
        "## Purpose",
        "",
        "This report summarizes a ligand-level verification pass for ASD allosteric-site rows that matched PDBbind v2020 by PDB ID. The row-level status table is local and git-ignored; this local summary contains aggregate counts only.",
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
            f"- ASD rows evaluated: {summary['scope']['matched_asd_rows_evaluated']}",
            f"- Unique PDB IDs evaluated: {summary['scope']['matched_unique_pdb_ids_evaluated']}",
            f"- Unique ASD target IDs evaluated: {summary['scope']['matched_unique_target_ids_evaluated']}",
            f"- Local row-level status TSV: `{display_path(status_tsv)}`",
            "",
            "## Tier Counts",
            "",
        ]
    )
    for tier, count in summary["tier_counts_by_asd_row"].items():
        lines.append(f"- `{tier}`: {count} ASD rows")
    lines.extend(
        [
            "",
            "## Tier A",
            "",
            f"- ASD rows: {summary['tier_a']['asd_rows']}",
            f"- Unique PDB IDs: {summary['tier_a']['unique_pdb_ids']}",
            f"- Unique ASD target IDs: {summary['tier_a']['unique_target_ids']}",
            "",
            "Tier-A status counts:",
            "",
        ]
    )
    for status, count in summary["tier_a"]["status_counts"].items():
        lines.append(f"- `{status}`: {count}")
    lines.extend(
        [
            "",
            "## Inferred Candidates",
            "",
            "These are separated from Tier A because PDBbind+ selected-ligand metadata was unavailable, even though the RCSB structure has a single nonpolymer component matching the ASD modulator.",
            "",
            f"- ASD rows: {summary['tier_a_candidate_inferred']['asd_rows']}",
            f"- Unique PDB IDs: {summary['tier_a_candidate_inferred']['unique_pdb_ids']}",
            "",
            "## Not Tier A",
            "",
            f"- ASD rows: {summary['not_tier_a']['asd_rows']}",
            f"- Unique PDB IDs: {summary['not_tier_a']['unique_pdb_ids']}",
            "",
            "Not-Tier-A status counts:",
            "",
        ]
    )
    for status, count in summary["not_tier_a"]["status_counts"].items():
        lines.append(f"- `{status}`: {count}")
    lines.extend(
        [
            "",
            "## PDB-Level Counts",
            "",
        ]
    )
    for status, count in summary["pdb_counts"].items():
        lines.append(f"- `{status}`: {count} unique PDB IDs")
    lines.extend(
        [
            "",
            "## Endpoint Counts",
            "",
        ]
    )
    for endpoint, count in summary["pdbbind_endpoint_counts_by_asd_row"].items():
        lines.append(f"- `{endpoint}`: {count} ASD rows")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Rows promoted to Tier A have PDBbind selected-ligand evidence aligned to the ASD modulator and RCSB structure evidence. Rows that remain outside Tier A have explicit reasons such as selected-ligand mismatch, unavailable selected-ligand metadata, or peptide/non-component ligand names that could not be uniquely resolved from public RCSB/PDBbind+ metadata.",
            "",
            "The local TSV should be used as the next worklist for manual or higher-cost verification, but it should not be redistributed without checking ASD and PDBbind terms.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asd-archive", type=Path, default=DEFAULT_ASD_ARCHIVE)
    parser.add_argument("--pdbbind-csv", type=Path, default=DEFAULT_PDBBIND_CSV)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--status-tsv", type=Path, default=DEFAULT_STATUS_TSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--online",
        action="store_true",
        help="Fetch missing PDBbind+ and RCSB ligand evidence.",
    )
    parser.add_argument("--refresh-pdbbind-plus", action="store_true")
    parser.add_argument("--refresh-rcsb", action="store_true")
    args = parser.parse_args()
    if (args.refresh_pdbbind_plus or args.refresh_rcsb) and not args.online:
        raise ValueError("Refreshing remote evidence requires --online")

    asd_rows = read_asd_rows(args.asd_archive)
    pdbbind_rows = read_csv(args.pdbbind_csv)
    pdb_to_pk = {
        row.get("PDBID", "").strip().upper(): row.get("pK", "").strip()
        for row in pdbbind_rows
        if row.get("PDBID", "").strip() and row.get("pK", "").strip()
    }
    matched = [
        (index, row)
        for index, row in enumerate(asd_rows, start=1)
        if row.get("allosteric_pdb", "").strip().upper() in pdb_to_pk
    ]
    matched_pdbs = sorted({row.get("allosteric_pdb", "").strip().upper() for _, row in matched})

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    pdbbind_cache_path = args.cache_dir / "pdbbind_plus_browser_by_pdb.json"
    rcsb_cache_path = args.cache_dir / "rcsb_ligand_evidence_by_pdb.json"
    pdbbind_cache = {} if args.refresh_pdbbind_plus else read_json(pdbbind_cache_path, {})
    rcsb_cache = {} if args.refresh_rcsb else read_json(rcsb_cache_path, {})

    if args.online:
        pdbbind_cache = update_cache_parallel(
            pdbbind_cache,
            matched_pdbs,
            fetch_pdbbind_plus_record,
            pdbbind_cache_path,
            args.workers,
            "PDBbind+ browser metadata",
        )
        rcsb_cache = update_cache_parallel(
            rcsb_cache,
            matched_pdbs,
            fetch_rcsb_ligand_evidence,
            rcsb_cache_path,
            args.workers,
            "RCSB ligand/entity evidence",
        )
    else:
        missing_pdbbind = set(matched_pdbs) - set(pdbbind_cache)
        missing_rcsb = set(matched_pdbs) - set(rcsb_cache)
        if missing_pdbbind or missing_rcsb:
            raise RuntimeError(
                "Local Tier-A evidence is incomplete: "
                f"{len(missing_pdbbind)} PDBbind+ and {len(missing_rcsb)} RCSB PDB IDs missing. "
                "Use --online to fetch them."
            )

    status_rows = [
        classify_match(
            asd_row=row,
            row_index=index,
            pdbbind_pk=pdb_to_pk[row.get("allosteric_pdb", "").strip().upper()],
            pdbbind_record=pdbbind_cache.get(row.get("allosteric_pdb", "").strip().upper(), {}),
            rcsb_evidence=rcsb_cache.get(row.get("allosteric_pdb", "").strip().upper(), {}),
        )
        for index, row in matched
    ]
    write_status_tsv(args.status_tsv, status_rows)

    inputs = {
        "asd_archive": display_path(args.asd_archive),
        "pdbbind_csv": display_path(args.pdbbind_csv),
        "pdbbind_plus_cache": display_path(pdbbind_cache_path),
        "rcsb_ligand_evidence_cache": display_path(rcsb_cache_path),
        "status_tsv": display_path(args.status_tsv),
    }
    summary = summarize_status_rows(status_rows, inputs)
    write_json(args.summary_json, summary)
    write_markdown(args.summary_md, summary, args.status_tsv)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
