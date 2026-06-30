#!/usr/bin/env python3
"""Run pretrained DrugWise/GMI inference on ASD Tier-A test-overlap rows.

This script prepares the 21 ASD Tier-A rows that overlap the DrugWise/PDBbind
v2020 appendix test-set predictions, extracts DrugWise-style GGL/MB features,
loads the local pretrained model, and writes row-level inference outputs.

Primary structure source
------------------------
The script uses the PDBbind++ 2020 refined-set archive from Hugging Face because
it provides PDBbind-style prepared `protein.pdb` and `ligand.mol2` files, which
are much closer to the DrugWise evaluation inputs than reconstructing ligand
MOL2 files from raw RCSB coordinates.

Important compatibility note
----------------------------
The public model asks for 11,000 selected feature columns. Some selected MB
feature names carry pandas merge suffixes (`_x`/`_y`) but the released feature
code does not preserve the original source labels that created those duplicate
columns. For reproducible inference with the released code, suffixed duplicate
columns are filled from their unsuffixed base descriptor value.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import site
import sys
import sysconfig
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from allosteric_affinity.drugwise import (  # noqa: E402
    canonicalize_mol2_atom_types,
    compute_base_descriptors,
    drugwise_import_context,
    strip_pandas_suffix,
)

DEFAULT_OVERLAP_TSV = (
    REPO_ROOT / "data/interim/drugwise/asd_drugwise_test_set_prediction_overlap.tsv"
)
DEFAULT_PDBBINDPP_ZIP = REPO_ROOT / "data/raw/pdbbindpp/pbpp-2020.zip"
DEFAULT_MODEL = REPO_ROOT / "DrugWise-Implementation-main/model/pretraiened_ggl_mb_score.pkl"
DEFAULT_DRUGWISE_DIR = REPO_ROOT / "DrugWise-Implementation-main"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/interim/drugwise/tier_a_inference"
DEFAULT_RCSB_CACHE_DIR = REPO_ROOT / "data/raw/rcsb/pdb"
DEFAULT_SUMMARY_JSON = REPO_ROOT / "outputs/asd_tier_a_drugwise_inference_summary.json"
DEFAULT_SUMMARY_MD = REPO_ROOT / "outputs/asd_tier_a_drugwise_inference_summary.md"

PDBBINDPP_URL = (
    "https://huggingface.co/datasets/photonmz/pdbbindpp-2020/resolve/main/"
    "pbpp-2020.zip?download=true"
)
RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
APPENDIX_V2020_DATASET = "pdbbind_v2020_appendix_d"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def download_file(url: str, path: Path, *, chunk_size: int = 2**20) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    started = time.time()
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    expected_bytes = int(response.headers.get("content-length") or 0)
    downloaded = 0
    with tmp.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            handle.write(chunk)
            downloaded += len(chunk)
    tmp.replace(path)
    return {
        "url": url,
        "path": display_path(path),
        "bytes": downloaded,
        "expected_bytes": expected_bytes,
        "seconds": time.time() - started,
    }


def ensure_pdbbindpp_zip(path: Path, *, download: bool) -> dict[str, Any]:
    if path.exists():
        return {
            "url": PDBBINDPP_URL,
            "path": display_path(path),
            "bytes": path.stat().st_size,
            "downloaded_this_run": False,
        }
    if not download:
        raise FileNotFoundError(
            f"{path} is missing. Re-run with downloads enabled or provide --pdbbindpp-zip."
        )
    info = download_file(PDBBINDPP_URL, path)
    info["downloaded_this_run"] = True
    return info


def clean_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    return cleaned.strip("_").lower()


def split_tokens(value: str) -> list[str]:
    return [token for token in re.split(r"[;,\s|]+", value.strip()) if token]


def find_obabel() -> Path | None:
    found = shutil.which("obabel")
    if found:
        return Path(found)
    candidates = [
        Path(sysconfig.get_path("scripts")) / "obabel.exe",
        Path(site.USER_BASE) / "Python310" / "Scripts" / "obabel.exe",
        Path(site.USER_BASE) / "Scripts" / "obabel.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_tier_a_targets(overlap_tsv: Path) -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_tsv(overlap_tsv)
        if row.get("prediction_dataset") == APPENDIX_V2020_DATASET
        and row.get("tier") == "tier_a"
    ]
    rows.sort(key=lambda row: (row["pdb_id"].lower(), int(row["asd_row_index"])))
    for row in rows:
        pdb_id = row["pdb_id"].lower()
        row["row_complex_id"] = f"{pdb_id}_asd{clean_id(row['asd_row_index'])}"
        row["actual_pk_float"] = float(row["actual_pk"])
        row["paper_predicted_pk_float"] = float(row["predicted_pk"])
    if len(rows) != 21:
        raise RuntimeError(f"Expected 21 Tier-A v2020 appendix rows, found {len(rows)}")
    return rows


def find_zip_member(names: Iterable[str], pdb_id: str, kind: str) -> str | None:
    pdb_lower = pdb_id.lower()
    allowed_basenames = {
        f"{kind}.pdb" if kind == "protein" else f"{kind}.mol2",
        f"{pdb_lower}_{kind}.pdb" if kind == "protein" else f"{pdb_lower}_{kind}.mol2",
    }
    candidates: list[str] = []
    for name in names:
        normalized = name.replace("\\", "/")
        parts = [part for part in normalized.split("/") if part]
        if len(parts) < 2:
            continue
        if parts[-1].lower() not in allowed_basenames:
            continue
        if pdb_lower not in {part.lower() for part in parts[:-1]}:
            continue
        candidates.append(name)
    if not candidates:
        return None
    candidates.sort(key=lambda item: (len(item), item))
    return candidates[0]


def download_rcsb_pdb(pdb_id: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{pdb_id.lower()}.pdb"
    if path.exists() and path.stat().st_size > 0:
        return path
    url = RCSB_PDB_URL.format(pdb_id=pdb_id.upper())
    response = requests.get(
        url,
        timeout=60,
        headers={"User-Agent": "asd-drugwise-tier-a-inference/0.1"},
    )
    response.raise_for_status()
    path.write_text(response.text, encoding="utf-8", newline="\n")
    return path


def keep_primary_altloc(line: str) -> str | None:
    if len(line) <= 16:
        return line
    altloc = line[16]
    if altloc not in {" ", "A", "1"}:
        return None
    return f"{line[:16]} {line[17:]}"


def parse_pdb_atom_key(line: str) -> tuple[str, str, str]:
    return line[17:20].strip(), line[21:22].strip(), line[22:26].strip()


def extract_ligand_lines_from_pdb(row: dict[str, Any], pdb_text: str) -> tuple[list[str], str]:
    component_tokens = {
        token.upper()
        for token in [*split_tokens(row.get("pdbbind_ligandname", "")), *split_tokens(row.get("modulator_alias", ""))]
        if token
    }
    chain_tokens = split_tokens(row.get("modulator_chain", ""))
    resi_tokens = split_tokens(row.get("modulator_resi", ""))
    exact_pairs: set[tuple[str, str]] = set()
    if chain_tokens and resi_tokens:
        if len(chain_tokens) == len(resi_tokens):
            exact_pairs = {
                (chain.strip(), resi.strip()) for chain, resi in zip(chain_tokens, resi_tokens)
            }
        elif len(resi_tokens) == 1:
            exact_pairs = {(chain.strip(), resi_tokens[0].strip()) for chain in chain_tokens}
    all_component_lines: list[str] = []
    exact_lines: list[str] = []
    ligand_serials: set[int] = set()
    for raw_line in pdb_text.splitlines():
        if not raw_line.startswith("HETATM"):
            continue
        line = keep_primary_altloc(raw_line)
        if line is None:
            continue
        comp_id, chain_id, resi = parse_pdb_atom_key(line)
        if component_tokens and comp_id.upper() not in component_tokens:
            continue
        all_component_lines.append(line)
        if exact_pairs and (chain_id, resi) in exact_pairs:
            exact_lines.append(line)
    selected_lines = exact_lines or all_component_lines
    for line in selected_lines:
        try:
            ligand_serials.add(int(line[6:11]))
        except ValueError:
            pass
    conect_lines: list[str] = []
    for line in pdb_text.splitlines():
        if not line.startswith("CONECT"):
            continue
        serials: list[int] = []
        for start in range(6, len(line), 5):
            token = line[start : start + 5].strip()
            if not token:
                continue
            try:
                serials.append(int(token))
            except ValueError:
                continue
        if ligand_serials.intersection(serials):
            conect_lines.append(line)
    basis = "component_chain_residue" if exact_lines else "component_only"
    return [*selected_lines, *conect_lines, "END"], basis


def prepare_structure_from_rcsb(
    row: dict[str, Any],
    row_complex_id: str,
    prepared_dir: Path,
    rcsb_cache_dir: Path,
) -> dict[str, Any]:
    obabel = find_obabel()
    if obabel is None:
        return {
            "status": "missing_obabel",
            "message": "RCSB fallback requires obabel.exe, but it was not found.",
        }
    pdb_path = download_rcsb_pdb(row["pdb_id"], rcsb_cache_dir)
    pdb_text = pdb_path.read_text(encoding="utf-8", errors="replace")
    target_dir = prepared_dir / row_complex_id
    target_dir.mkdir(parents=True, exist_ok=True)
    protein_path = target_dir / f"{row_complex_id}_protein.pdb"
    ligand_pdb_path = target_dir / f"{row_complex_id}_ligand.pdb"
    ligand_mol2_path = target_dir / f"{row_complex_id}_ligand.mol2"

    protein_lines: list[str] = []
    for raw_line in pdb_text.splitlines():
        if not raw_line.startswith("ATOM"):
            continue
        line = keep_primary_altloc(raw_line)
        if line is not None:
            protein_lines.append(line)
    protein_path.write_text("\n".join([*protein_lines, "END", ""]), encoding="utf-8", newline="\n")

    ligand_lines, ligand_basis = extract_ligand_lines_from_pdb(row, pdb_text)
    if len([line for line in ligand_lines if line.startswith("HETATM")]) == 0:
        return {
            "status": "rcsb_ligand_not_found",
            "message": "No HETATM ligand lines matched the PDBbind/ASD ligand component.",
            "protein_path": protein_path,
        }
    ligand_pdb_path.write_text("\n".join([*ligand_lines, ""]), encoding="utf-8", newline="\n")
    import subprocess

    completed = subprocess.run(
        [
            str(obabel),
            str(ligand_pdb_path),
            "-O",
            str(ligand_mol2_path),
            "-h",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0 or not ligand_mol2_path.exists():
        return {
            "status": "obabel_failed",
            "message": completed.stderr.strip() or completed.stdout.strip(),
            "protein_path": protein_path,
            "ligand_pdb_path": ligand_pdb_path,
        }
    return {
        "status": "prepared",
        "message": f"RCSB fallback ligand basis: {ligand_basis}",
        "protein_path": protein_path,
        "ligand_mol2_path": ligand_mol2_path,
        "ligand_pdb_path": ligand_pdb_path,
    }


def prepare_structures(
    zip_path: Path,
    targets: list[dict[str, Any]],
    prepared_dir: Path,
    rcsb_cache_dir: Path,
    *,
    enable_rcsb_fallback: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared_rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        member_cache: dict[tuple[str, str], str | None] = {}
        for row in targets:
            pdb_id = row["pdb_id"].lower()
            row_complex_id = row["row_complex_id"]
            protein_member = member_cache.setdefault(
                (pdb_id, "protein"), find_zip_member(names, pdb_id, "protein")
            )
            ligand_member = member_cache.setdefault(
                (pdb_id, "ligand"), find_zip_member(names, pdb_id, "ligand")
            )
            status = "prepared"
            message = ""
            structure_source = "pdbbindpp_2020_refined"
            protein_path = prepared_dir / row_complex_id / f"{row_complex_id}_protein.pdb"
            ligand_path = prepared_dir / row_complex_id / f"{row_complex_id}_ligand.mol2"
            if protein_member is None or ligand_member is None:
                status = "missing_pdbbindpp_member"
                message = "protein.pdb or ligand.mol2 was not found in the PDBbind++ archive"
                if enable_rcsb_fallback:
                    fallback = prepare_structure_from_rcsb(
                        row, row_complex_id, prepared_dir, rcsb_cache_dir
                    )
                    status = fallback["status"]
                    message = fallback["message"]
                    if status == "prepared":
                        structure_source = "rcsb_pdb_obabel_fallback"
                        protein_path = fallback["protein_path"]
                        ligand_path = fallback["ligand_mol2_path"]
            else:
                protein_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(protein_member) as source, protein_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
                with zf.open(ligand_member) as source, ligand_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
            prepared_rows.append(
                {
                    **row,
                    "structure_source": structure_source,
                    "protein_member": protein_member or "",
                    "ligand_member": ligand_member or "",
                    "prepared_protein_pdb": display_path(protein_path) if protein_path.exists() else "",
                    "prepared_ligand_mol2": display_path(ligand_path) if ligand_path.exists() else "",
                    "preparation_status": status,
                    "preparation_message": message,
                }
            )
    source_counts = Counter(row["preparation_status"] for row in prepared_rows)
    return prepared_rows, {
        "prepared_row_count": source_counts.get("prepared", 0),
        "status_counts": dict(sorted(source_counts.items())),
        "structure_source_counts": dict(Counter(row["structure_source"] for row in prepared_rows)),
        "unique_prepared_pdb_ids": len(
            {row["pdb_id"] for row in prepared_rows if row["preparation_status"] == "prepared"}
        ),
    }


def compute_selected_feature_matrix(
    prepared_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    model = joblib.load(args.model)
    feature_names = list(model.feature_names_in_)
    vectors: list[dict[str, float]] = []
    status_rows: list[dict[str, Any]] = []

    with drugwise_import_context(args.drugwise_dir):
        from get_scores import PAIRWISE_SCORE, SYBYL_GGL, TRIPLET_SCORE

        ggl = SYBYL_GGL(args.cutoff, args.tau, args.kernel_type, args.power)
        pairwise = PAIRWISE_SCORE(args.cutoff, args.tau, args.kernel_type, args.power)
        triplet = TRIPLET_SCORE(args.cutoff, args.tau, args.kernel_type, args.power)

        for row in prepared_rows:
            row_complex_id = row["row_complex_id"]
            if row["preparation_status"] != "prepared":
                status_rows.append(
                    {
                        "row_complex_id": row_complex_id,
                        "pdb_id": row["pdb_id"],
                        "feature_status": "skipped_not_prepared",
                        "feature_message": row["preparation_message"],
                        "selected_feature_missing_bases": "",
                    }
                )
                continue
            protein_file = REPO_ROOT / row["prepared_protein_pdb"]
            ligand_file = REPO_ROOT / row["prepared_ligand_mol2"]
            try:
                canonicalize_mol2_atom_types(ligand_file, args.drugwise_dir)
                base_values = compute_base_descriptors(
                    {"ggl": ggl, "pairwise": pairwise, "triplet": triplet},
                    protein_file,
                    ligand_file,
                )

                missing_bases = sorted(
                    {
                        strip_pandas_suffix(feature)
                        for feature in feature_names
                        if strip_pandas_suffix(feature) not in base_values
                    }
                )
                vector = {
                    feature: base_values.get(strip_pandas_suffix(feature), 0.0)
                    for feature in feature_names
                }
                vector["row_complex_id"] = row_complex_id
                vectors.append(vector)
                status_rows.append(
                    {
                        "row_complex_id": row_complex_id,
                        "pdb_id": row["pdb_id"],
                        "feature_status": "ok",
                        "feature_message": "",
                        "selected_feature_missing_bases": ";".join(missing_bases),
                        "selected_feature_missing_base_count": len(missing_bases),
                        "nonzero_selected_features": sum(
                            1 for feature in feature_names if vector[feature] != 0
                        ),
                    }
                )
            except Exception as exc:
                status_rows.append(
                    {
                        "row_complex_id": row_complex_id,
                        "pdb_id": row["pdb_id"],
                        "feature_status": "error",
                        "feature_message": repr(exc),
                        "selected_feature_missing_bases": "",
                    }
                )

    if not vectors:
        return pd.DataFrame(columns=["row_complex_id", *feature_names]), status_rows, {}

    matrix = pd.DataFrame(vectors)
    matrix = matrix[["row_complex_id", *feature_names]]
    duplicate_suffix_count = sum(
        1 for feature in feature_names if feature.endswith("_x") or feature.endswith("_y")
    )
    matrix_info = {
        "model_class": type(model).__name__,
        "model_feature_count": len(feature_names),
        "duplicate_suffixed_model_feature_count": duplicate_suffix_count,
        "successful_feature_rows": len(matrix),
        "feature_status_counts": dict(Counter(row["feature_status"] for row in status_rows)),
    }
    return matrix, status_rows, matrix_info


def predict_with_model(matrix: pd.DataFrame, model_path: Path) -> np.ndarray:
    model = joblib.load(model_path)
    feature_names = list(model.feature_names_in_)
    if matrix.empty:
        return np.array([])
    return model.predict(matrix[feature_names])


def pearson(rows: list[dict[str, Any]], x_key: str, y_key: str) -> float | None:
    if len(rows) < 2:
        return None
    xs = [float(row[x_key]) for row in rows]
    ys = [float(row[y_key]) for row in rows]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / math.sqrt(var_x * var_y)


def mae(rows: list[dict[str, Any]], error_key: str) -> float | None:
    if not rows:
        return None
    return sum(abs(float(row[error_key])) for row in rows) / len(rows)


def rmse(rows: list[dict[str, Any]], error_key: str) -> float | None:
    if not rows:
        return None
    return math.sqrt(sum(float(row[error_key]) ** 2 for row in rows) / len(rows))


def dedupe_by_pdb(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pdb: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_pdb.setdefault(str(row["pdb_id"]).lower(), row)
    return [by_pdb[pdb_id] for pdb_id in sorted(by_pdb)]


def metric_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "pearson_model_vs_actual": pearson(rows, "actual_pk", "model_predicted_pk"),
        "mae_model_vs_actual": mae(rows, "model_error_vs_actual"),
        "rmse_model_vs_actual": rmse(rows, "model_error_vs_actual"),
        "pearson_paper_vs_actual": pearson(rows, "actual_pk", "paper_predicted_pk"),
        "mae_paper_vs_actual": mae(rows, "paper_error_vs_actual"),
        "rmse_paper_vs_actual": rmse(rows, "paper_error_vs_actual"),
        "mae_model_vs_paper_prediction": mae(rows, "model_delta_vs_paper_prediction"),
        "rmse_model_vs_paper_prediction": rmse(rows, "model_delta_vs_paper_prediction"),
        "pdb_ids": sorted({str(row["pdb_id"]).lower() for row in rows}),
    }


def build_prediction_rows(
    prepared_rows: list[dict[str, Any]],
    matrix: pd.DataFrame,
    predictions: np.ndarray,
) -> list[dict[str, Any]]:
    prepared_by_id = {row["row_complex_id"]: row for row in prepared_rows}
    prediction_by_id = {
        row_complex_id: float(predictions[index])
        for index, row_complex_id in enumerate(matrix["row_complex_id"].tolist())
    }
    output_rows: list[dict[str, Any]] = []
    for row_complex_id, model_prediction in prediction_by_id.items():
        row = prepared_by_id[row_complex_id]
        actual = float(row["actual_pk_float"])
        paper_prediction = float(row["paper_predicted_pk_float"])
        output_rows.append(
            {
                "row_complex_id": row_complex_id,
                "asd_row_index": row["asd_row_index"],
                "pdb_id": row["pdb_id"].lower(),
                "allosteric_pdb": row["allosteric_pdb"],
                "target_id": row["target_id"],
                "target_gene": row["target_gene"],
                "organism": row["organism"],
                "pdb_uniprot": row["pdb_uniprot"],
                "modulator_alias": row["modulator_alias"],
                "modulator_chain": row["modulator_chain"],
                "modulator_resi": row["modulator_resi"],
                "pdbbind_ligandname": row["pdbbind_ligandname"],
                "pdbbind_kdtype": row["pdbbind_kdtype"],
                "pdbbind_kdoriginal": row["pdbbind_kdoriginal"],
                "actual_pk": actual,
                "paper_predicted_pk": paper_prediction,
                "model_predicted_pk": model_prediction,
                "paper_error_vs_actual": paper_prediction - actual,
                "model_error_vs_actual": model_prediction - actual,
                "model_delta_vs_paper_prediction": model_prediction - paper_prediction,
                "tier": row["tier"],
                "status": row["status"],
                "structure_source": row["structure_source"],
                "prepared_protein_pdb": row["prepared_protein_pdb"],
                "prepared_ligand_mol2": row["prepared_ligand_mol2"],
            }
        )
    output_rows.sort(key=lambda row: (row["pdb_id"], int(row["asd_row_index"])))
    return output_rows


def fmt_float(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def build_summary(
    args: argparse.Namespace,
    download_info: dict[str, Any],
    prep_info: dict[str, Any],
    matrix_info: dict[str, Any],
    prediction_rows: list[dict[str, Any]],
    feature_status_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    unique_rows = dedupe_by_pdb(prediction_rows)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "overlap_tsv": display_path(args.overlap_tsv),
            "pdbbindpp_zip": display_path(args.pdbbindpp_zip),
            "pdbbindpp_url": PDBBINDPP_URL,
            "rcsb_pdb_url_template": RCSB_PDB_URL,
            "rcsb_cache_dir": display_path(args.rcsb_cache_dir),
            "model": display_path(args.model),
            "drugwise_dir": display_path(args.drugwise_dir),
        },
        "outputs": {
            "output_dir": display_path(args.output_dir),
            "prepared_status_tsv": display_path(args.prepared_status_tsv),
            "feature_status_tsv": display_path(args.feature_status_tsv),
            "feature_matrix_tsv": display_path(args.feature_matrix_tsv),
            "prediction_tsv": display_path(args.prediction_tsv),
            "summary_json": display_path(args.summary_json),
            "summary_md": display_path(args.summary_md),
        },
        "parameters": {
            "kernel_type": args.kernel_type,
            "tau": args.tau,
            "power": args.power,
            "cutoff": args.cutoff,
            "appendix_prediction_dataset": APPENDIX_V2020_DATASET,
        },
        "download": download_info,
        "preparation": prep_info,
        "feature_matrix": matrix_info,
        "metrics": {
            "row_weighted": metric_block(prediction_rows),
            "unique_pdb": metric_block(unique_rows),
        },
        "feature_status_counts": dict(Counter(row["feature_status"] for row in feature_status_rows)),
        "notes": [
            "The 21 ASD Tier-A rows collapse to 20 unique PDB IDs because 3ZT2 appears twice in ASD with two residue annotations.",
            "PDBbind++ provides one prepared ligand.mol2 per PDB entry; duplicated ASD rows therefore receive the same entry-level prepared-structure prediction.",
            "Rows absent from the PDBbind++ refined archive are prepared from RCSB PDB coordinates and Open Babel MOL2 conversion.",
            "Selected model features ending in _x/_y are filled from the unsuffixed base descriptor because the released feature code does not expose the original duplicate-source labels.",
        ],
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    params = summary["parameters"]
    lines = [
        "# ASD Tier-A DrugWise Inference",
        "",
        f"Generated: `{summary['generated_at_utc']}` UTC.",
        "",
        "This report summarizes a local inference run of the downloaded pretrained DrugWise/GMI model on the 21 ASD Tier-A rows that overlap the paper's PDBbind v2020 appendix test predictions.",
        "",
        "## Inputs",
        "",
        f"- ASD/DrugWise overlap table: `{summary['inputs']['overlap_tsv']}`.",
        f"- PDBbind++ 2020 refined archive: `{summary['inputs']['pdbbindpp_zip']}`.",
        f"- PDBbind++ source URL: `{summary['inputs']['pdbbindpp_url']}`.",
        f"- RCSB fallback URL template: `{summary['inputs']['rcsb_pdb_url_template']}`.",
        f"- Pretrained model: `{summary['inputs']['model']}`.",
        "",
        "## Feature Parameters",
        "",
        f"- Kernel type: `{params['kernel_type']}`.",
        f"- Tau: `{params['tau']}`.",
        f"- Power: `{params['power']}`.",
        f"- Cutoff: `{params['cutoff']}` Angstrom.",
        "",
        "## Preparation",
        "",
        f"- Prepared rows: `{summary['preparation']['prepared_row_count']}`.",
        f"- Unique prepared PDB IDs: `{summary['preparation']['unique_prepared_pdb_ids']}`.",
        f"- Preparation status counts: `{summary['preparation']['status_counts']}`.",
        f"- Structure source counts: `{summary['preparation']['structure_source_counts']}`.",
        "",
        "## Feature Matrix",
        "",
        f"- Successful feature rows: `{summary['feature_matrix'].get('successful_feature_rows', 0)}`.",
        f"- Model feature count: `{summary['feature_matrix'].get('model_feature_count', 0)}`.",
        f"- Suffixed duplicate model features filled from base descriptors: `{summary['feature_matrix'].get('duplicate_suffixed_model_feature_count', 0)}`.",
        f"- Feature status counts: `{summary['feature_status_counts']}`.",
        "",
        "## Metrics",
        "",
        "| Scope | n | Pearson model vs actual | MAE model vs actual | RMSE model vs actual | Pearson paper vs actual | MAE model vs paper prediction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("row_weighted", "unique_pdb"):
        block = metrics[label]
        lines.append(
            "| {label} | {n} | {pm} | {mae} | {rmse} | {pp} | {mp} |".format(
                label=label,
                n=block["n"],
                pm=fmt_float(block["pearson_model_vs_actual"]),
                mae=fmt_float(block["mae_model_vs_actual"]),
                rmse=fmt_float(block["rmse_model_vs_actual"]),
                pp=fmt_float(block["pearson_paper_vs_actual"]),
                mp=fmt_float(block["mae_model_vs_paper_prediction"]),
            )
        )
    lines.extend(
        [
            "",
            "## Unique PDB IDs",
            "",
            ", ".join(summary["metrics"]["unique_pdb"]["pdb_ids"]) + ".",
            "",
            "## Local Outputs",
            "",
            f"- Preparation status: `{summary['outputs']['prepared_status_tsv']}`.",
            f"- Feature status: `{summary['outputs']['feature_status_tsv']}`.",
            f"- Selected feature matrix: `{summary['outputs']['feature_matrix_tsv']}`.",
            f"- Row-level predictions: `{summary['outputs']['prediction_tsv']}`.",
            "",
            "The row-level outputs are local `data/interim/` artifacts and are not intended for redistribution.",
            "",
            "## Caveats",
            "",
            "- This is not an independent ASD benchmark: all 20 unique PDB IDs are already in the original DrugWise/PDBbind core-style test appendix.",
            "- The public model feature names contain `_x`/`_y` duplicate columns whose original kernel/source labels are not preserved in the released feature code; this run fills those columns from the corresponding base descriptor value.",
            "- For the duplicated ASD rows of `3ZT2`, PDBbind++ provides one prepared ligand per PDB entry, so both ASD rows receive the same prepared-structure prediction.",
            "- Fallback RCSB/Open Babel rows may not exactly reproduce official PDBbind preprocessing.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlap-tsv", type=Path, default=DEFAULT_OVERLAP_TSV)
    parser.add_argument("--pdbbindpp-zip", type=Path, default=DEFAULT_PDBBINDPP_ZIP)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--drugwise-dir", type=Path, default=DEFAULT_DRUGWISE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rcsb-cache-dir", type=Path, default=DEFAULT_RCSB_CACHE_DIR)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--kernel-type", default="lorentz_kernel")
    parser.add_argument("--tau", type=float, default=1.5)
    parser.add_argument("--power", type=float, default=2.5)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Require the PDBbind++ archive to already exist instead of downloading it.",
    )
    parser.add_argument(
        "--no-rcsb-fallback",
        action="store_true",
        help="Do not prepare PDBbind++-missing rows from RCSB PDB coordinates.",
    )
    args = parser.parse_args()
    args.prepared_dir = args.output_dir / "prepared"
    args.prepared_status_tsv = args.output_dir / "asd_tier_a_preparation_status.tsv"
    args.feature_status_tsv = args.output_dir / "asd_tier_a_feature_status.tsv"
    args.feature_matrix_tsv = args.output_dir / "asd_tier_a_selected_feature_matrix.tsv"
    args.prediction_tsv = args.output_dir / "asd_tier_a_drugwise_inference_predictions.tsv"
    return args


def main() -> None:
    args = parse_args()
    targets = load_tier_a_targets(args.overlap_tsv)
    download_info = ensure_pdbbindpp_zip(args.pdbbindpp_zip, download=not args.no_download)
    prepared_rows, prep_info = prepare_structures(
        args.pdbbindpp_zip,
        targets,
        args.prepared_dir,
        args.rcsb_cache_dir,
        enable_rcsb_fallback=not args.no_rcsb_fallback,
    )
    matrix, feature_status_rows, matrix_info = compute_selected_feature_matrix(prepared_rows, args)
    predictions = predict_with_model(matrix, args.model)
    prediction_rows = build_prediction_rows(prepared_rows, matrix, predictions)

    prepared_fieldnames = [
        "row_complex_id",
        "asd_row_index",
        "pdb_id",
        "allosteric_pdb",
        "target_id",
        "target_gene",
        "modulator_alias",
        "modulator_chain",
        "modulator_resi",
        "pdbbind_ligandname",
        "pdbbind_kdtype",
        "pdbbind_kdoriginal",
        "actual_pk",
        "predicted_pk",
        "structure_source",
        "protein_member",
        "ligand_member",
        "prepared_protein_pdb",
        "prepared_ligand_mol2",
        "preparation_status",
        "preparation_message",
    ]
    prediction_fieldnames = [
        "row_complex_id",
        "asd_row_index",
        "pdb_id",
        "allosteric_pdb",
        "target_id",
        "target_gene",
        "organism",
        "pdb_uniprot",
        "modulator_alias",
        "modulator_chain",
        "modulator_resi",
        "pdbbind_ligandname",
        "pdbbind_kdtype",
        "pdbbind_kdoriginal",
        "actual_pk",
        "paper_predicted_pk",
        "model_predicted_pk",
        "paper_error_vs_actual",
        "model_error_vs_actual",
        "model_delta_vs_paper_prediction",
        "tier",
        "status",
        "structure_source",
        "prepared_protein_pdb",
        "prepared_ligand_mol2",
    ]
    feature_status_fieldnames = [
        "row_complex_id",
        "pdb_id",
        "feature_status",
        "feature_message",
        "selected_feature_missing_base_count",
        "selected_feature_missing_bases",
        "nonzero_selected_features",
    ]

    write_tsv(args.prepared_status_tsv, prepared_rows, prepared_fieldnames)
    write_tsv(args.feature_status_tsv, feature_status_rows, feature_status_fieldnames)
    args.feature_matrix_tsv.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.feature_matrix_tsv, sep="\t", index=False)
    write_tsv(args.prediction_tsv, prediction_rows, prediction_fieldnames)

    summary = build_summary(
        args, download_info, prep_info, matrix_info, prediction_rows, feature_status_rows
    )
    write_json(args.summary_json, summary)
    write_markdown(args.summary_md, summary)

    unique_metrics = summary["metrics"]["unique_pdb"]
    print(
        "ASD Tier-A DrugWise inference complete: "
        f"n={unique_metrics['n']} unique PDB IDs, "
        f"Pearson={unique_metrics['pearson_model_vs_actual']:.4f}, "
        f"MAE={unique_metrics['mae_model_vs_actual']:.4f}, "
        f"RMSE={unique_metrics['rmse_model_vs_actual']:.4f}"
    )
    print(f"Wrote {display_path(args.prediction_tsv)}")
    print(f"Wrote {display_path(args.summary_md)}")


if __name__ == "__main__":
    main()
