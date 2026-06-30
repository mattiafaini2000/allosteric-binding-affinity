#!/usr/bin/env python3
"""Calculate DrugWise appendix test-set performance on ASD-overlap complexes.

The DrugWise paper reports per-complex predictions in PDF appendices rather
than local CSV files. This script extracts those tables from the local PDF,
intersects them with the locally verified ASD/PDBbind Tier-A status table, and
reports Pearson correlations for the ASD complexes present in the paper test
set.

Row-level outputs are written under data/interim/ because they contain
ASD-derived identifiers. Versioned report outputs contain aggregate statistics
and the unique PDB IDs used for the calculations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = REPO_ROOT / "2026.06.05.730001v1.full.pdf"
DEFAULT_ASD_TIER_STATUS = REPO_ROOT / "data/interim/pdbbind/asd_pdbbind_v2020_tier_a_status.tsv"
DEFAULT_INTERIM_DIR = REPO_ROOT / "data/interim/drugwise"
DEFAULT_PREDICTIONS_TSV = DEFAULT_INTERIM_DIR / "drugwise_appendix_test_predictions.tsv"
DEFAULT_OVERLAP_TSV = DEFAULT_INTERIM_DIR / "asd_drugwise_test_set_prediction_overlap.tsv"
DEFAULT_SUMMARY_JSON = (
    REPO_ROOT / "outputs/asd_drugwise_test_prediction_overlap_summary.json"
)
DEFAULT_SUMMARY_MD = (
    REPO_ROOT / "outputs/asd_drugwise_test_prediction_overlap_summary.md"
)

APPENDIX_TABLES = {
    "pdbbind_v2016_appendix_c": {
        "appendix": "C",
        "paper_dataset_label": "PDBbind v2016",
        "first_pdf_page": 21,
        "last_pdf_page": 23,
    },
    "pdbbind_v2020_appendix_d": {
        "appendix": "D",
        "paper_dataset_label": "PDBbind v2020",
        "first_pdf_page": 24,
        "last_pdf_page": 26,
    },
}

PREDICTION_RE = re.compile(
    r"\b([0-9][A-Za-z0-9]{3})\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\b"
)


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


def run_pdftotext(pdf_path: Path, first_page: int, last_page: int) -> str:
    completed = subprocess.run(
        [
            "pdftotext",
            "-layout",
            "-f",
            str(first_page),
            "-l",
            str(last_page),
            str(pdf_path),
            "-",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout


def parse_appendix_predictions(pdf_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_id, meta in APPENDIX_TABLES.items():
        text = run_pdftotext(pdf_path, meta["first_pdf_page"], meta["last_pdf_page"])
        seen: set[str] = set()
        for match in PREDICTION_RE.finditer(text):
            pdb_id = match.group(1).lower()
            if pdb_id == "2026":
                continue
            actual = float(match.group(2))
            predicted = float(match.group(3))
            if pdb_id in seen:
                raise RuntimeError(f"Duplicate prediction row for {pdb_id} in {dataset_id}")
            seen.add(pdb_id)
            rows.append(
                {
                    "prediction_dataset": dataset_id,
                    "paper_dataset_label": meta["paper_dataset_label"],
                    "appendix": meta["appendix"],
                    "pdb_id": pdb_id,
                    "actual_pk": actual,
                    "predicted_pk": predicted,
                    "prediction_error": predicted - actual,
                    "absolute_error": abs(predicted - actual),
                    "squared_error": (predicted - actual) ** 2,
                }
            )
        if len(seen) != 285:
            raise RuntimeError(
                f"Expected 285 parsed rows for {dataset_id}, parsed {len(seen)} rows"
            )
    return rows


def pearson(rows: list[dict[str, Any]]) -> float | None:
    if len(rows) < 2:
        return None
    xs = [float(row["actual_pk"]) for row in rows]
    ys = [float(row["predicted_pk"]) for row in rows]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / math.sqrt(var_x * var_y)


def rmse(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return math.sqrt(sum(float(row["squared_error"]) for row in rows) / len(rows))


def mae(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(float(row["absolute_error"]) for row in rows) / len(rows)


def summarize_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "pearson": pearson(rows),
        "mae": mae(rows),
        "rmse": rmse(rows),
        "pdb_ids": sorted({str(row["pdb_id"]) for row in rows}),
    }


def deduplicate_prediction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pdb: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_pdb.setdefault(str(row["pdb_id"]), row)
    return [by_pdb[pdb_id] for pdb_id in sorted(by_pdb)]


def build_overlap_rows(
    predictions: list[dict[str, Any]], asd_status_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    predictions_by_dataset_pdb = {
        (str(row["prediction_dataset"]), str(row["pdb_id"])): row for row in predictions
    }
    overlap_rows: list[dict[str, Any]] = []
    for asd_row in asd_status_rows:
        pdb_id = asd_row["allosteric_pdb"].lower()
        for dataset_id in APPENDIX_TABLES:
            prediction = predictions_by_dataset_pdb.get((dataset_id, pdb_id))
            if prediction is None:
                continue
            overlap_rows.append(
                {
                    **prediction,
                    "asd_row_index": asd_row.get("asd_row_index", ""),
                    "target_id": asd_row.get("target_id", ""),
                    "target_gene": asd_row.get("target_gene", ""),
                    "organism": asd_row.get("organism", ""),
                    "pdb_uniprot": asd_row.get("pdb_uniprot", ""),
                    "allosteric_pdb": asd_row.get("allosteric_pdb", ""),
                    "modulator_alias": asd_row.get("modulator_alias", ""),
                    "modulator_chain": asd_row.get("modulator_chain", ""),
                    "modulator_resi": asd_row.get("modulator_resi", ""),
                    "modulator_class": asd_row.get("modulator_class", ""),
                    "modulator_name": asd_row.get("modulator_name", ""),
                    "asd_pubmed_id": asd_row.get("asd_pubmed_id", ""),
                    "pdbbind_v2020_pk": asd_row.get("pdbbind_v2020_pk", ""),
                    "pdbbind_ligandname": asd_row.get("pdbbind_ligandname", ""),
                    "pdbbind_kdtype": asd_row.get("pdbbind_kdtype", ""),
                    "pdbbind_kdoriginal": asd_row.get("pdbbind_kdoriginal", ""),
                    "pdbbind_pubmed": asd_row.get("pdbbind_pubmed", ""),
                    "tier": asd_row.get("tier", ""),
                    "status": asd_row.get("status", ""),
                    "verification_basis": asd_row.get("verification_basis", ""),
                }
            )
    return overlap_rows


def build_summary(
    predictions: list[dict[str, Any]],
    overlap_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    by_prediction_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_prediction_dataset[str(row["prediction_dataset"])].append(row)

    overlap_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in overlap_rows:
        overlap_by_dataset[str(row["prediction_dataset"])].append(row)

    full_test_metrics = {
        dataset_id: summarize_metric_rows(rows)
        for dataset_id, rows in sorted(by_prediction_dataset.items())
    }

    asd_overlap_metrics: dict[str, dict[str, Any]] = {}
    for dataset_id, rows in sorted(overlap_by_dataset.items()):
        asd_overlap_metrics[dataset_id] = {}
        for tier_label in ("any", "tier_a", "not_tier_a"):
            tier_rows = [
                row for row in rows if tier_label == "any" or row.get("tier") == tier_label
            ]
            unique_rows = deduplicate_prediction_rows(tier_rows)
            asd_overlap_metrics[dataset_id][tier_label] = {
                "row_weighted": summarize_metric_rows(tier_rows),
                "unique_pdb": summarize_metric_rows(unique_rows),
                "unique_asd_row_count": len({row.get("asd_row_index", "") for row in tier_rows}),
                "unique_target_count": len({row.get("target_id", "") for row in tier_rows}),
            }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "paper_pdf": display_path(args.pdf),
            "asd_tier_status_tsv": display_path(args.asd_tier_status),
        },
        "outputs": {
            "appendix_predictions_tsv": display_path(args.predictions_tsv),
            "asd_overlap_tsv": display_path(args.overlap_tsv),
            "summary_json": display_path(args.summary_json),
            "summary_md": display_path(args.summary_md),
        },
        "notes": [
            "Pearson is calculated between the paper appendix actual pK and predicted pK values.",
            "The headline ASD metric should use unique_pdb rather than row_weighted because one PDB entry can appear in more than one ASD row.",
            "Tier A means the PDBbind-selected ligand was verified against the ASD allosteric modulator by the existing ligand-level ASD/PDBbind workflow.",
        ],
        "full_test_metrics": full_test_metrics,
        "asd_overlap_metrics": asd_overlap_metrics,
    }


def fmt_float(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ASD Complexes In DrugWise Test Predictions",
        "",
        f"Generated: `{summary['generated_at_utc']}` UTC.",
        "",
        "This report parses the per-complex prediction tables embedded in the local DrugWise paper PDF appendices and intersects them with the verified ASD/PDBbind v2020 Tier-A status table.",
        "",
        "## Inputs",
        "",
        f"- Paper PDF: `{summary['inputs']['paper_pdf']}`.",
        f"- ASD/PDBbind Tier-A status table: `{summary['inputs']['asd_tier_status_tsv']}`.",
        "",
        "## Full Appendix Test Tables",
        "",
        "| Prediction table | n | Pearson | MAE | RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset_id, metrics in summary["full_test_metrics"].items():
        lines.append(
            "| {dataset} | {n} | {pearson} | {mae} | {rmse} |".format(
                dataset=dataset_id,
                n=metrics["n"],
                pearson=fmt_float(metrics["pearson"]),
                mae=fmt_float(metrics["mae"]),
                rmse=fmt_float(metrics["rmse"]),
            )
        )
    lines.extend(
        [
            "",
            "## ASD Overlap",
            "",
            "Metrics below are computed only on ASD/PDBbind rows whose PDB ID appears in the paper appendix test prediction table. `unique_pdb` is the preferred statistic because the model prediction table has one row per PDB entry.",
            "",
            "| Prediction table | ASD tier subset | ASD rows | unique PDB IDs | Pearson, unique PDB | MAE, unique PDB | RMSE, unique PDB | Pearson, row-weighted |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset_id, tier_metrics in summary["asd_overlap_metrics"].items():
        for tier_label in ("any", "tier_a", "not_tier_a"):
            metrics = tier_metrics[tier_label]
            row_weighted = metrics["row_weighted"]
            unique_pdb = metrics["unique_pdb"]
            lines.append(
                "| {dataset} | {tier} | {rows} | {pdbs} | {pearson_unique} | {mae_unique} | {rmse_unique} | {pearson_rows} |".format(
                    dataset=dataset_id,
                    tier=tier_label,
                    rows=row_weighted["n"],
                    pdbs=unique_pdb["n"],
                    pearson_unique=fmt_float(unique_pdb["pearson"]),
                    mae_unique=fmt_float(unique_pdb["mae"]),
                    rmse_unique=fmt_float(unique_pdb["rmse"]),
                    pearson_rows=fmt_float(row_weighted["pearson"]),
                )
            )

    lines.extend(["", "## Tier-A Unique PDB IDs", ""])
    for dataset_id, tier_metrics in summary["asd_overlap_metrics"].items():
        pdb_ids = tier_metrics["tier_a"]["unique_pdb"]["pdb_ids"]
        lines.append(f"- `{dataset_id}`: {', '.join(pdb_ids)}.")

    lines.extend(
        [
            "",
            "## Local Row-Level Outputs",
            "",
            f"- Parsed appendix predictions: `{summary['outputs']['appendix_predictions_tsv']}`.",
            f"- ASD overlap rows: `{summary['outputs']['asd_overlap_tsv']}`.",
            "",
            "The row-level TSVs are local `data/interim/` artifacts and are not intended for redistribution.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--asd-tier-status", type=Path, default=DEFAULT_ASD_TIER_STATUS)
    parser.add_argument("--predictions-tsv", type=Path, default=DEFAULT_PREDICTIONS_TSV)
    parser.add_argument("--overlap-tsv", type=Path, default=DEFAULT_OVERLAP_TSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = parse_appendix_predictions(args.pdf)
    asd_status_rows = read_tsv(args.asd_tier_status)
    overlap_rows = build_overlap_rows(predictions, asd_status_rows)

    prediction_fieldnames = [
        "prediction_dataset",
        "paper_dataset_label",
        "appendix",
        "pdb_id",
        "actual_pk",
        "predicted_pk",
        "prediction_error",
        "absolute_error",
        "squared_error",
    ]
    overlap_fieldnames = prediction_fieldnames + [
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
        "pdbbind_v2020_pk",
        "pdbbind_ligandname",
        "pdbbind_kdtype",
        "pdbbind_kdoriginal",
        "pdbbind_pubmed",
        "tier",
        "status",
        "verification_basis",
    ]

    write_tsv(args.predictions_tsv, predictions, prediction_fieldnames)
    write_tsv(args.overlap_tsv, overlap_rows, overlap_fieldnames)
    summary = build_summary(predictions, overlap_rows, args)
    write_json(args.summary_json, summary)
    write_markdown(args.summary_md, summary)

    headline = summary["asd_overlap_metrics"]["pdbbind_v2020_appendix_d"]["tier_a"][
        "unique_pdb"
    ]
    print(
        "Tier-A ASD overlap with DrugWise PDBbind v2020 appendix test table: "
        f"n={headline['n']} unique PDB IDs, Pearson={headline['pearson']:.4f}, "
        f"MAE={headline['mae']:.4f}, RMSE={headline['rmse']:.4f}"
    )
    print(f"Wrote {display_path(args.summary_md)}")


if __name__ == "__main__":
    main()
