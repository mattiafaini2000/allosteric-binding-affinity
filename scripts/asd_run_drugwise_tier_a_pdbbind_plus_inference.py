#!/usr/bin/env python3
"""Run DrugWise inference on ASD Tier-A rows using official PDBbind+ v2020 files.

This script prepares the same 21 strict Tier-A ASD appendix-overlap rows used by
`asd_run_drugwise_tier_a_inference.py`, but it prefers the official PDBbind+
v2020 renewed website packages rather than the PDBbind++ Hugging Face archive or
RCSB/Open Babel fallback.

The script extracts only the needed `*_protein.pdb` and `*_ligand.mol2` files
from the official tarballs, hashes them against the previous local prepared
files, runs the pretrained DrugWise/GMI model, and writes row-level comparison
artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from asd_run_drugwise_tier_a_inference import (
    APPENDIX_V2020_DATASET,
    DEFAULT_DRUGWISE_DIR,
    DEFAULT_MODEL,
    DEFAULT_OVERLAP_TSV,
    build_prediction_rows,
    compute_selected_feature_matrix,
    dedupe_by_pdb,
    display_path,
    fmt_float,
    load_tier_a_targets,
    metric_block,
    predict_with_model,
    write_json,
    write_tsv,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDBBIND_PLUS_REFINED_TAR = (
    REPO_ROOT / "data/raw/pdbbind_plus/PDBbind_v2020_refined.tar.gz"
)
DEFAULT_PDBBIND_PLUS_OTHER_PL_TAR = (
    REPO_ROOT / "data/raw/pdbbind_plus/PDBbind_v2020_other_PL.tar.gz"
)
DEFAULT_PREVIOUS_STATUS_TSV = (
    REPO_ROOT / "data/interim/drugwise/tier_a_inference/asd_tier_a_preparation_status.tsv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/interim/drugwise/tier_a_inference_pdbbind_plus"
DEFAULT_SUMMARY_JSON = (
    REPO_ROOT / "outputs/asd_tier_a_drugwise_pdbbind_plus_inference_summary.json"
)
DEFAULT_SUMMARY_MD = (
    REPO_ROOT / "outputs/asd_tier_a_drugwise_pdbbind_plus_inference_summary.md"
)

PDBBIND_PLUS_URLS = {
    "pdbbind_plus_v2020_refined": (
        "https://static.pdbbind-plus.org.cn/v2020-renew_website/"
        "PDBbind_v2020_refined.tar.gz"
    ),
    "pdbbind_plus_v2020_other_pl": (
        "https://static.pdbbind-plus.org.cn/v2020-renew_website/"
        "PDBbind_v2020_other_PL.tar.gz"
    ),
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path_text: str) -> dict[str, Any]:
    if not path_text:
        return {
            "path": "",
            "exists": False,
            "bytes": 0,
            "sha256": "",
        }
    path = REPO_ROOT / path_text
    if not path.exists():
        return {
            "path": path_text,
            "exists": False,
            "bytes": 0,
            "sha256": "",
        }
    return {
        "path": path_text,
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def tar_source_name(path: Path) -> str:
    name = path.name.lower()
    if "refined" in name:
        return "pdbbind_plus_v2020_refined"
    if "other_pl" in name:
        return "pdbbind_plus_v2020_other_pl"
    return f"pdbbind_plus_{path.stem}"


def build_tar_member_index(
    archive_paths: list[Path],
    pdb_ids: set[str],
) -> dict[str, dict[str, dict[str, str]]]:
    index: dict[str, dict[str, dict[str, str]]] = {
        pdb_id: {} for pdb_id in sorted(pdb_ids)
    }
    wanted_basenames = {
        pdb_id: {
            "protein": f"{pdb_id}_protein.pdb",
            "ligand": f"{pdb_id}_ligand.mol2",
        }
        for pdb_id in pdb_ids
    }
    for archive_path in archive_paths:
        source_name = tar_source_name(archive_path)
        if not archive_path.exists():
            raise FileNotFoundError(f"Missing PDBbind+ archive: {archive_path}")
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                basename = Path(member.name).name.lower()
                for pdb_id, basenames in wanted_basenames.items():
                    for kind, wanted in basenames.items():
                        if basename == wanted and kind not in index[pdb_id]:
                            index[pdb_id][kind] = {
                                "archive": str(archive_path),
                                "archive_source": source_name,
                                "member": member.name,
                            }
    return index


def batch_extract_members(
    specs_by_archive: dict[str, dict[str, list[Path]]],
) -> None:
    for archive_text, member_targets in specs_by_archive.items():
        archive_path = Path(archive_text)
        remaining = set(member_targets)
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar:
                if member.name not in member_targets:
                    continue
                source = tar.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Could not read {member.name} from {archive_path}")
                data = source.read()
                source.close()
                for target_path in member_targets[member.name]:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    target_path.write_bytes(data)
                remaining.discard(member.name)
                if not remaining:
                    break
        if remaining:
            missing = ", ".join(sorted(remaining)[:10])
            raise RuntimeError(
                f"{archive_path} did not contain {len(remaining)} expected members: {missing}"
            )


def load_previous_status(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return {row["row_complex_id"]: row for row in read_tsv(path)}


def prepare_from_pdbbind_plus(
    targets: list[dict[str, Any]],
    prepared_dir: Path,
    archive_paths: list[Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pdb_ids = {row["pdb_id"].lower() for row in targets}
    index = build_tar_member_index(archive_paths, pdb_ids)
    prepared_rows: list[dict[str, Any]] = []
    extraction_specs: dict[str, dict[str, list[Path]]] = {}

    for row in targets:
        pdb_id = row["pdb_id"].lower()
        row_complex_id = row["row_complex_id"]
        members = index.get(pdb_id, {})
        protein_hit = members.get("protein")
        ligand_hit = members.get("ligand")
        protein_path = prepared_dir / row_complex_id / f"{row_complex_id}_protein.pdb"
        ligand_path = prepared_dir / row_complex_id / f"{row_complex_id}_ligand.mol2"
        status = "prepared"
        message = ""
        structure_source = ""
        protein_member = ""
        ligand_member = ""
        protein_archive = ""
        ligand_archive = ""

        if protein_hit is None or ligand_hit is None:
            status = "missing_pdbbind_plus_member"
            missing = []
            if protein_hit is None:
                missing.append("protein")
            if ligand_hit is None:
                missing.append("ligand")
            message = "Missing official PDBbind+ member(s): " + ",".join(missing)
        else:
            protein_archive_path = Path(protein_hit["archive"])
            ligand_archive_path = Path(ligand_hit["archive"])
            extraction_specs.setdefault(str(protein_archive_path), {}).setdefault(
                protein_hit["member"], []
            ).append(protein_path)
            extraction_specs.setdefault(str(ligand_archive_path), {}).setdefault(
                ligand_hit["member"], []
            ).append(ligand_path)
            protein_member = protein_hit["member"]
            ligand_member = ligand_hit["member"]
            protein_archive = display_path(protein_archive_path)
            ligand_archive = display_path(ligand_archive_path)
            if protein_hit["archive_source"] == ligand_hit["archive_source"]:
                structure_source = protein_hit["archive_source"]
            else:
                structure_source = (
                    protein_hit["archive_source"] + "+" + ligand_hit["archive_source"]
                )

        prepared_rows.append(
            {
                **row,
                "structure_source": structure_source,
                "protein_member": protein_member,
                "ligand_member": ligand_member,
                "protein_archive": protein_archive,
                "ligand_archive": ligand_archive,
                "prepared_protein_pdb": display_path(protein_path)
                if status == "prepared"
                else "",
                "prepared_ligand_mol2": display_path(ligand_path)
                if status == "prepared"
                else "",
                "preparation_status": status,
                "preparation_message": message,
            }
        )

    batch_extract_members(extraction_specs)

    status_counts = Counter(row["preparation_status"] for row in prepared_rows)
    return prepared_rows, {
        "prepared_row_count": status_counts.get("prepared", 0),
        "status_counts": dict(sorted(status_counts.items())),
        "structure_source_counts": dict(
            Counter(row["structure_source"] for row in prepared_rows)
        ),
        "unique_prepared_pdb_ids": len(
            {row["pdb_id"] for row in prepared_rows if row["preparation_status"] == "prepared"}
        ),
    }


def build_file_comparison_rows(
    prepared_rows: list[dict[str, Any]],
    previous_by_id: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in prepared_rows:
        row_complex_id = row["row_complex_id"]
        previous = previous_by_id.get(row_complex_id, {})
        old_protein = file_info(previous.get("prepared_protein_pdb", ""))
        old_ligand = file_info(previous.get("prepared_ligand_mol2", ""))
        new_protein = file_info(row.get("prepared_protein_pdb", ""))
        new_ligand = file_info(row.get("prepared_ligand_mol2", ""))
        protein_changed = old_protein["sha256"] != new_protein["sha256"]
        ligand_changed = old_ligand["sha256"] != new_ligand["sha256"]
        rows.append(
            {
                "row_complex_id": row_complex_id,
                "pdb_id": row["pdb_id"].lower(),
                "previous_structure_source": previous.get("structure_source", ""),
                "official_structure_source": row["structure_source"],
                "previous_protein_pdb": old_protein["path"],
                "official_protein_pdb": new_protein["path"],
                "previous_protein_bytes": old_protein["bytes"],
                "official_protein_bytes": new_protein["bytes"],
                "previous_protein_sha256": old_protein["sha256"],
                "official_protein_sha256": new_protein["sha256"],
                "protein_changed": protein_changed,
                "previous_ligand_mol2": old_ligand["path"],
                "official_ligand_mol2": new_ligand["path"],
                "previous_ligand_bytes": old_ligand["bytes"],
                "official_ligand_bytes": new_ligand["bytes"],
                "previous_ligand_sha256": old_ligand["sha256"],
                "official_ligand_sha256": new_ligand["sha256"],
                "ligand_changed": ligand_changed,
                "any_file_changed": protein_changed or ligand_changed,
            }
        )
    rows.sort(key=lambda row: (row["pdb_id"], row["row_complex_id"]))
    return rows


def build_prediction_comparison_rows(
    official_predictions: list[dict[str, Any]],
    previous_prediction_tsv: Path,
) -> list[dict[str, Any]]:
    if not previous_prediction_tsv.exists():
        return []
    previous_rows = {
        row["row_complex_id"]: row for row in read_tsv(previous_prediction_tsv)
    }
    rows: list[dict[str, Any]] = []
    for row in official_predictions:
        previous = previous_rows.get(row["row_complex_id"], {})
        previous_prediction = (
            float(previous["model_predicted_pk"])
            if previous.get("model_predicted_pk")
            else np.nan
        )
        official_prediction = float(row["model_predicted_pk"])
        rows.append(
            {
                "row_complex_id": row["row_complex_id"],
                "pdb_id": row["pdb_id"],
                "actual_pk": row["actual_pk"],
                "paper_predicted_pk": row["paper_predicted_pk"],
                "previous_structure_source": previous.get("structure_source", ""),
                "official_structure_source": row["structure_source"],
                "previous_model_predicted_pk": previous_prediction,
                "official_model_predicted_pk": official_prediction,
                "prediction_delta_official_minus_previous": official_prediction
                - previous_prediction
                if not np.isnan(previous_prediction)
                else np.nan,
            }
        )
    rows.sort(key=lambda row: (row["pdb_id"], row["row_complex_id"]))
    return rows


def build_summary(
    args: argparse.Namespace,
    prep_info: dict[str, Any],
    matrix_info: dict[str, Any],
    prediction_rows: list[dict[str, Any]],
    feature_status_rows: list[dict[str, Any]],
    file_comparison_rows: list[dict[str, Any]],
    prediction_comparison_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    unique_rows = dedupe_by_pdb(prediction_rows)
    changed_count = sum(1 for row in file_comparison_rows if row["any_file_changed"])
    protein_changed_count = sum(1 for row in file_comparison_rows if row["protein_changed"])
    ligand_changed_count = sum(1 for row in file_comparison_rows if row["ligand_changed"])
    deltas = [
        abs(float(row["prediction_delta_official_minus_previous"]))
        for row in prediction_comparison_rows
        if row["prediction_delta_official_minus_previous"] == row["prediction_delta_official_minus_previous"]
    ]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "overlap_tsv": display_path(args.overlap_tsv),
            "pdbbind_plus_archives": [
                {
                    "path": display_path(path),
                    "source": tar_source_name(path),
                    "url": PDBBIND_PLUS_URLS.get(tar_source_name(path), ""),
                    "bytes": path.stat().st_size if path.exists() else 0,
                    "sha256": sha256_file(path) if path.exists() else "",
                }
                for path in args.pdbbind_plus_tar
            ],
            "previous_preparation_status_tsv": display_path(args.previous_status_tsv),
            "previous_prediction_tsv": display_path(args.previous_prediction_tsv),
            "model": display_path(args.model),
            "drugwise_dir": display_path(args.drugwise_dir),
        },
        "outputs": {
            "output_dir": display_path(args.output_dir),
            "prepared_status_tsv": display_path(args.prepared_status_tsv),
            "file_comparison_tsv": display_path(args.file_comparison_tsv),
            "prediction_comparison_tsv": display_path(args.prediction_comparison_tsv),
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
        "preparation": prep_info,
        "file_comparison": {
            "rows_compared": len(file_comparison_rows),
            "any_file_changed_rows": changed_count,
            "protein_changed_rows": protein_changed_count,
            "ligand_changed_rows": ligand_changed_count,
        },
        "prediction_comparison": {
            "rows_compared": len(prediction_comparison_rows),
            "max_abs_delta": max(deltas) if deltas else None,
            "mean_abs_delta": float(np.mean(deltas)) if deltas else None,
        },
        "feature_matrix": matrix_info,
        "metrics": {
            "row_weighted": metric_block(prediction_rows),
            "unique_pdb": metric_block(unique_rows),
        },
        "feature_status_counts": dict(Counter(row["feature_status"] for row in feature_status_rows)),
    }


def write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    params = summary["parameters"]
    lines = [
        "# ASD Tier-A DrugWise Inference With Official PDBbind+ Structures",
        "",
        f"Generated: `{summary['generated_at_utc']}` UTC.",
        "",
        "This report summarizes a local inference run of the downloaded pretrained DrugWise/GMI model on the 21 ASD Tier-A rows that overlap the paper's PDBbind v2020 appendix test predictions, using official PDBbind+ v2020 renewed website prepared structure archives.",
        "",
        "## Inputs",
        "",
        f"- ASD/DrugWise overlap table: `{summary['inputs']['overlap_tsv']}`.",
        f"- Previous preparation status table: `{summary['inputs']['previous_preparation_status_tsv']}`.",
        f"- Previous prediction table: `{summary['inputs']['previous_prediction_tsv']}`.",
        f"- Pretrained model: `{summary['inputs']['model']}`.",
        "",
        "PDBbind+ archives:",
        "",
    ]
    for archive in summary["inputs"]["pdbbind_plus_archives"]:
        lines.append(
            f"- `{archive['path']}` ({archive['bytes']} bytes), source `{archive['source']}`, URL `{archive['url']}`, SHA256 `{archive['sha256']}`."
        )
    lines.extend(
        [
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
            "## File Comparison Against Previous Local Prepared Inputs",
            "",
            f"- Rows compared: `{summary['file_comparison']['rows_compared']}`.",
            f"- Rows with any file changed: `{summary['file_comparison']['any_file_changed_rows']}`.",
            f"- Rows with protein file changed: `{summary['file_comparison']['protein_changed_rows']}`.",
            f"- Rows with ligand MOL2 changed: `{summary['file_comparison']['ligand_changed_rows']}`.",
            "",
            "## Prediction Comparison Against Previous Local Run",
            "",
            f"- Rows compared: `{summary['prediction_comparison']['rows_compared']}`.",
            f"- Mean absolute prediction delta: `{fmt_float(summary['prediction_comparison']['mean_abs_delta'])}`.",
            f"- Max absolute prediction delta: `{fmt_float(summary['prediction_comparison']['max_abs_delta'])}`.",
            "",
            "## Metrics",
            "",
            "| Scope | n | Pearson model vs actual | MAE model vs actual | RMSE model vs actual | Pearson paper vs actual | MAE model vs paper prediction |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
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
            "## Local Outputs",
            "",
            f"- Preparation status: `{summary['outputs']['prepared_status_tsv']}`.",
            f"- File comparison: `{summary['outputs']['file_comparison_tsv']}`.",
            f"- Prediction comparison: `{summary['outputs']['prediction_comparison_tsv']}`.",
            f"- Feature status: `{summary['outputs']['feature_status_tsv']}`.",
            f"- Selected feature matrix: `{summary['outputs']['feature_matrix_tsv']}`.",
            f"- Row-level predictions: `{summary['outputs']['prediction_tsv']}`.",
            "",
            "The row-level outputs are local `data/interim/` artifacts and are not intended for redistribution.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlap-tsv", type=Path, default=DEFAULT_OVERLAP_TSV)
    parser.add_argument(
        "--pdbbind-plus-tar",
        type=Path,
        action="append",
        default=[DEFAULT_PDBBIND_PLUS_REFINED_TAR, DEFAULT_PDBBIND_PLUS_OTHER_PL_TAR],
    )
    parser.add_argument("--previous-status-tsv", type=Path, default=DEFAULT_PREVIOUS_STATUS_TSV)
    parser.add_argument(
        "--previous-prediction-tsv",
        type=Path,
        default=REPO_ROOT
        / "data/interim/drugwise/tier_a_inference/asd_tier_a_drugwise_inference_predictions.tsv",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--drugwise-dir", type=Path, default=DEFAULT_DRUGWISE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--kernel-type", default="lorentz_kernel")
    parser.add_argument("--tau", type=float, default=1.5)
    parser.add_argument("--power", type=float, default=2.5)
    parser.add_argument("--cutoff", type=float, default=6.0)
    args = parser.parse_args()
    args.prepared_dir = args.output_dir / "prepared"
    args.prepared_status_tsv = args.output_dir / "asd_tier_a_pdbbind_plus_preparation_status.tsv"
    args.file_comparison_tsv = args.output_dir / "asd_tier_a_pdbbind_plus_file_comparison.tsv"
    args.prediction_comparison_tsv = (
        args.output_dir / "asd_tier_a_pdbbind_plus_prediction_comparison.tsv"
    )
    args.feature_status_tsv = args.output_dir / "asd_tier_a_pdbbind_plus_feature_status.tsv"
    args.feature_matrix_tsv = args.output_dir / "asd_tier_a_pdbbind_plus_selected_feature_matrix.tsv"
    args.prediction_tsv = args.output_dir / "asd_tier_a_pdbbind_plus_drugwise_predictions.tsv"
    return args


def main() -> None:
    args = parse_args()
    targets = load_tier_a_targets(args.overlap_tsv)
    prepared_rows, prep_info = prepare_from_pdbbind_plus(
        targets, args.prepared_dir, args.pdbbind_plus_tar
    )
    previous_by_id = load_previous_status(args.previous_status_tsv)
    file_comparison_rows = build_file_comparison_rows(prepared_rows, previous_by_id)
    matrix, feature_status_rows, matrix_info = compute_selected_feature_matrix(prepared_rows, args)
    predictions = predict_with_model(matrix, args.model)
    prediction_rows = build_prediction_rows(prepared_rows, matrix, predictions)
    prediction_comparison_rows = build_prediction_comparison_rows(
        prediction_rows, args.previous_prediction_tsv
    )

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
        "protein_archive",
        "ligand_archive",
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
    file_comparison_fieldnames = [
        "row_complex_id",
        "pdb_id",
        "previous_structure_source",
        "official_structure_source",
        "previous_protein_pdb",
        "official_protein_pdb",
        "previous_protein_bytes",
        "official_protein_bytes",
        "previous_protein_sha256",
        "official_protein_sha256",
        "protein_changed",
        "previous_ligand_mol2",
        "official_ligand_mol2",
        "previous_ligand_bytes",
        "official_ligand_bytes",
        "previous_ligand_sha256",
        "official_ligand_sha256",
        "ligand_changed",
        "any_file_changed",
    ]
    prediction_comparison_fieldnames = [
        "row_complex_id",
        "pdb_id",
        "actual_pk",
        "paper_predicted_pk",
        "previous_structure_source",
        "official_structure_source",
        "previous_model_predicted_pk",
        "official_model_predicted_pk",
        "prediction_delta_official_minus_previous",
    ]

    write_tsv(args.prepared_status_tsv, prepared_rows, prepared_fieldnames)
    write_tsv(args.file_comparison_tsv, file_comparison_rows, file_comparison_fieldnames)
    write_tsv(args.feature_status_tsv, feature_status_rows, feature_status_fieldnames)
    args.feature_matrix_tsv.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.feature_matrix_tsv, sep="\t", index=False)
    write_tsv(args.prediction_tsv, prediction_rows, prediction_fieldnames)
    write_tsv(
        args.prediction_comparison_tsv,
        prediction_comparison_rows,
        prediction_comparison_fieldnames,
    )

    summary = build_summary(
        args,
        prep_info,
        matrix_info,
        prediction_rows,
        feature_status_rows,
        file_comparison_rows,
        prediction_comparison_rows,
    )
    write_json(args.summary_json, summary)
    write_summary_markdown(args.summary_md, summary)

    unique_metrics = summary["metrics"]["unique_pdb"]
    changed = summary["file_comparison"]["any_file_changed_rows"]
    print(
        "ASD Tier-A PDBbind+ DrugWise inference complete: "
        f"changed_rows={changed}, "
        f"n={unique_metrics['n']} unique PDB IDs, "
        f"Pearson={unique_metrics['pearson_model_vs_actual']:.4f}, "
        f"MAE={unique_metrics['mae_model_vs_actual']:.4f}, "
        f"RMSE={unique_metrics['rmse_model_vs_actual']:.4f}"
    )
    print(f"Wrote {display_path(args.prediction_tsv)}")
    print(f"Wrote {display_path(args.summary_md)}")


if __name__ == "__main__":
    main()
