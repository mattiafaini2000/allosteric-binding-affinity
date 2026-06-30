#!/usr/bin/env python3
"""Rebuild DrugWise/GMI features and retrain on official PDBbind+ v2020.

The pipeline reconstructs the original paper's PDBbind v2020 train/test setup
from local DrugWise label CSVs and official PDBbind+ v2020 renewed structure
archives:

- train candidates: PDBbind v2020 general set minus CASF-2016/core IDs
- test set: CASF-2016/core IDs
- optional refined split metadata derived from the official refined archive

It extracts only protein PDB and ligand MOL2 files, computes selected DrugWise
features for two Lorentz parameterizations, fuses the two feature matrices into
the downloaded model's `_x`/`_y` feature schema, and retrains a
GradientBoostingRegressor using the hyperparameters stored in the downloaded
pretrained model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from allosteric_affinity.drugwise import (  # noqa: E402
    DEFAULT_SOURCE,
    LORENTZ_PARAMETER_SETS,
    README_SOURCE,
    canonicalize_mol2_atom_types,
    compute_base_descriptors,
    source_for_feature,
    strip_pandas_suffix,
)

DEFAULT_V2020_GENERAL_CSV = REPO_ROOT / "DrugWise-Implementation-main/data/PDBbindv2020_General.csv"
DEFAULT_CORE_CSV = REPO_ROOT / "DrugWise-Implementation-main/data/CASF_2016_CoreSet.csv"
DEFAULT_MODEL = REPO_ROOT / "DrugWise-Implementation-main/model/pretraiened_ggl_mb_score.pkl"
DEFAULT_DRUGWISE_DIR = REPO_ROOT / "DrugWise-Implementation-main"
DEFAULT_PDBBIND_PLUS_REFINED_TAR = (
    REPO_ROOT / "data/raw/pdbbind_plus/PDBbind_v2020_refined.tar.gz"
)
DEFAULT_PDBBIND_PLUS_OTHER_PL_TAR = (
    REPO_ROOT / "data/raw/pdbbind_plus/PDBbind_v2020_other_PL.tar.gz"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/interim/drugwise_paper_retrain/pdbbind_v2020"
DEFAULT_SUMMARY_JSON = (
    REPO_ROOT / "outputs/pdbbind_plus_drugwise_retrain_summary.json"
)
DEFAULT_SUMMARY_MD = (
    REPO_ROOT / "outputs/pdbbind_plus_drugwise_retrain_summary.md"
)

PARAMETER_SETS = LORENTZ_PARAMETER_SETS
FUSION_VARIANTS = {
    "x_default_y_readme_unsuff_default": {
        "x_source": DEFAULT_SOURCE,
        "y_source": README_SOURCE,
        "unsuffixed_source": DEFAULT_SOURCE,
    },
    "x_default_y_readme_unsuff_readme": {
        "x_source": DEFAULT_SOURCE,
        "y_source": README_SOURCE,
        "unsuffixed_source": README_SOURCE,
    },
    "x_readme_y_default_unsuff_default": {
        "x_source": README_SOURCE,
        "y_source": DEFAULT_SOURCE,
        "unsuffixed_source": DEFAULT_SOURCE,
    },
    "x_readme_y_default_unsuff_readme": {
        "x_source": README_SOURCE,
        "y_source": DEFAULT_SOURCE,
        "unsuffixed_source": README_SOURCE,
    },
}

_WORKER: dict[str, Any] = {}


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_pdb_id(value: Any) -> str:
    return str(value).strip().lower()


def load_model_feature_schema(model_path: Path) -> tuple[list[str], list[str], dict[str, int]]:
    model = joblib.load(model_path)
    feature_names = list(model.feature_names_in_)
    base_names = [strip_pandas_suffix(feature) for feature in feature_names]
    suffix_counts = {
        "x": sum(1 for feature in feature_names if feature.endswith("_x")),
        "y": sum(1 for feature in feature_names if feature.endswith("_y")),
        "unsuffixed": sum(
            1 for feature in feature_names if not feature.endswith("_x") and not feature.endswith("_y")
        ),
    }
    return feature_names, base_names, suffix_counts


def load_dataset_rows(
    general_csv: Path,
    core_csv: Path,
    refined_ids: set[str],
) -> pd.DataFrame:
    general = pd.read_csv(general_csv)
    general["pdb_id"] = general["PDBID"].map(normalize_pdb_id)
    general["pK"] = general["pK"].astype(float)
    core = pd.read_csv(core_csv)
    core["pdb_id"] = core["PDBID"].map(normalize_pdb_id)
    core_ids = set(core["pdb_id"])
    core_pk_by_id = dict(zip(core["pdb_id"], core["pK"].astype(float)))
    rows = general[["pdb_id", "pK"]].copy()
    rows["is_core_test"] = rows["pdb_id"].isin(core_ids)
    rows["is_general_train"] = ~rows["is_core_test"]
    rows["is_refined"] = rows["pdb_id"].isin(refined_ids)
    rows["is_refined_train"] = rows["is_refined"] & ~rows["is_core_test"]
    rows["core_pk"] = rows["pdb_id"].map(core_pk_by_id)
    rows["pk_source"] = np.where(rows["is_core_test"], "PDBbindv2020_General_and_CASF2016", "PDBbindv2020_General")
    rows = rows.sort_values("pdb_id").reset_index(drop=True)
    return rows


def tar_source_name(path: Path) -> str:
    name = path.name.lower()
    if "refined" in name:
        return "pdbbind_plus_v2020_refined"
    if "other_pl" in name:
        return "pdbbind_plus_v2020_other_pl"
    return f"pdbbind_plus_{path.stem}"


def build_archive_index(
    archive_paths: list[Path],
) -> tuple[dict[str, dict[str, dict[str, str]]], set[str]]:
    index: dict[str, dict[str, dict[str, str]]] = {}
    refined_ids: set[str] = set()
    for archive_path in archive_paths:
        if not archive_path.exists():
            raise FileNotFoundError(f"Missing PDBbind+ archive: {archive_path}")
        source_name = tar_source_name(archive_path)
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                basename = Path(member.name).name.lower()
                if not (basename.endswith("_protein.pdb") or basename.endswith("_ligand.mol2")):
                    continue
                pdb_id = basename.split("_", 1)[0].lower()
                if source_name == "pdbbind_plus_v2020_refined":
                    refined_ids.add(pdb_id)
                kind = "protein" if basename.endswith("_protein.pdb") else "ligand"
                index.setdefault(pdb_id, {})
                index[pdb_id].setdefault(
                    kind,
                    {
                        "archive": str(archive_path),
                        "archive_source": source_name,
                        "member": member.name,
                    },
                )
    return index, refined_ids


def extract_needed_structures(
    rows: pd.DataFrame,
    archive_index: dict[str, dict[str, dict[str, str]]],
    prepared_dir: Path,
    *,
    force: bool,
) -> list[dict[str, Any]]:
    prepared_dir.mkdir(parents=True, exist_ok=True)
    status_rows: list[dict[str, Any]] = []
    extraction_specs: dict[str, dict[str, list[Path]]] = {}

    for row in rows.itertuples(index=False):
        pdb_id = row.pdb_id
        members = archive_index.get(pdb_id, {})
        protein_hit = members.get("protein")
        ligand_hit = members.get("ligand")
        protein_path = prepared_dir / pdb_id / f"{pdb_id}_protein.pdb"
        ligand_path = prepared_dir / pdb_id / f"{pdb_id}_ligand.mol2"
        status = "prepared"
        message = ""
        source = ""
        protein_member = ""
        ligand_member = ""

        if protein_hit is None or ligand_hit is None:
            status = "missing_pdbbind_plus_member"
            missing = []
            if protein_hit is None:
                missing.append("protein")
            if ligand_hit is None:
                missing.append("ligand")
            message = "Missing official PDBbind+ member(s): " + ",".join(missing)
        else:
            protein_member = protein_hit["member"]
            ligand_member = ligand_hit["member"]
            source = (
                protein_hit["archive_source"]
                if protein_hit["archive_source"] == ligand_hit["archive_source"]
                else protein_hit["archive_source"] + "+" + ligand_hit["archive_source"]
            )
            if force or not protein_path.exists():
                extraction_specs.setdefault(protein_hit["archive"], {}).setdefault(protein_member, []).append(protein_path)
            if force or not ligand_path.exists():
                extraction_specs.setdefault(ligand_hit["archive"], {}).setdefault(ligand_member, []).append(ligand_path)

        status_rows.append(
            {
                "pdb_id": pdb_id,
                "pK": row.pK,
                "is_core_test": bool(row.is_core_test),
                "is_general_train": bool(row.is_general_train),
                "is_refined": bool(row.is_refined),
                "is_refined_train": bool(row.is_refined_train),
                "structure_source": source,
                "protein_member": protein_member,
                "ligand_member": ligand_member,
                "prepared_protein_pdb": display_path(protein_path) if status == "prepared" else "",
                "prepared_ligand_mol2": display_path(ligand_path) if status == "prepared" else "",
                "preparation_status": status,
                "preparation_message": message,
            }
        )

    for archive_text, member_targets in extraction_specs.items():
        archive_path = Path(archive_text)
        remaining = set(member_targets)
        started = time.time()
        print(f"Extracting {len(remaining)} members from {display_path(archive_path)}")
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
            preview = ", ".join(sorted(remaining)[:10])
            raise RuntimeError(f"{archive_path} missing {len(remaining)} requested members: {preview}")
        print(f"Finished {display_path(archive_path)} extraction in {time.time() - started:.1f}s")

    return status_rows


def init_feature_worker(
    drugwise_dir_text: str,
    base_names: list[str],
    feature_names: list[str],
) -> None:
    global _WORKER
    drugwise_dir = Path(drugwise_dir_text)
    os.chdir(drugwise_dir)
    sys.path.insert(0, str(drugwise_dir / "src"))
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from get_scores import PAIRWISE_SCORE, SYBYL_GGL, TRIPLET_SCORE

    scorers: dict[str, dict[str, Any]] = {}
    for source_name, params in PARAMETER_SETS.items():
        scorers[source_name] = {
            "ggl": SYBYL_GGL(params["cutoff"], params["tau"], params["kernel_type"], params["power"]),
            "pairwise": PAIRWISE_SCORE(params["cutoff"], params["tau"], params["kernel_type"], params["power"]),
            "triplet": TRIPLET_SCORE(params["cutoff"], params["tau"], params["kernel_type"], params["power"]),
        }
    _WORKER = {
        "drugwise_dir": drugwise_dir,
        "base_names": base_names,
        "feature_names": feature_names,
        "scorers": scorers,
        "canonicalize_mol2_atom_types": canonicalize_mol2_atom_types,
    }


def compute_param_vector(protein_file: Path, ligand_file: Path, source_name: str) -> np.ndarray:
    scorers = _WORKER["scorers"][source_name]
    base_values = compute_base_descriptors(scorers, protein_file, ligand_file)
    return np.asarray(
        [base_values.get(base_name, 0.0) for base_name in _WORKER["base_names"]],
        dtype=np.float32,
    )


def feature_worker_task(task: dict[str, Any]) -> dict[str, Any]:
    pdb_id = task["pdb_id"]
    protein_file = Path(task["protein_file"])
    ligand_file = Path(task["ligand_file"])
    try:
        _WORKER["canonicalize_mol2_atom_types"](ligand_file, _WORKER["drugwise_dir"])
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with redirect_stdout(sink), redirect_stderr(sink):
                readme = compute_param_vector(protein_file, ligand_file, README_SOURCE)
                default = compute_param_vector(protein_file, ligand_file, DEFAULT_SOURCE)
        return {
            "pdb_id": pdb_id,
            "status": "ok",
            "message": "",
            README_SOURCE: readme,
            DEFAULT_SOURCE: default,
            "nonzero_readme": int(np.count_nonzero(readme)),
            "nonzero_default": int(np.count_nonzero(default)),
        }
    except Exception as exc:
        return {
            "pdb_id": pdb_id,
            "status": "error",
            "message": repr(exc),
            README_SOURCE: None,
            DEFAULT_SOURCE: None,
            "nonzero_readme": 0,
            "nonzero_default": 0,
        }


def expected_memmap_bytes(shape: tuple[int, int], *, dtype: str = "float32") -> int:
    return int(np.prod(shape) * np.dtype(dtype).itemsize)


def ensure_memmap(
    path: Path,
    shape: tuple[int, int],
    *,
    dtype: str = "float32",
    preserve_existing: bool = False,
) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    if preserve_existing and path.exists() and path.stat().st_size == expected_memmap_bytes(shape, dtype=dtype):
        return np.memmap(path, dtype=dtype, mode="r+", shape=shape)
    return np.memmap(path, dtype=dtype, mode="w+", shape=shape)


def compute_feature_matrices(
    dataset: pd.DataFrame,
    prepared_rows: list[dict[str, Any]],
    feature_names: list[str],
    base_names: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    n_rows = len(dataset)
    n_features = len(feature_names)
    feature_dir = args.output_dir / "features"
    readme_path = feature_dir / f"{README_SOURCE}.float32.memmap"
    default_path = feature_dir / f"{DEFAULT_SOURCE}.float32.memmap"
    status_tsv = feature_dir / "feature_status.tsv"

    status_rows: list[dict[str, Any]]
    preserve_existing = False
    if not args.force_features and status_tsv.exists():
        existing_status = read_tsv(status_tsv)
        if len(existing_status) == n_rows and all(row.get("status") == "ok" for row in existing_status):
            print("Feature memmaps already complete; reusing cached matrices.")
            return {README_SOURCE: readme_path, DEFAULT_SOURCE: default_path}, existing_status
        if (
            len(existing_status) == n_rows
            and readme_path.exists()
            and default_path.exists()
            and readme_path.stat().st_size == expected_memmap_bytes((n_rows, n_features))
            and default_path.stat().st_size == expected_memmap_bytes((n_rows, n_features))
        ):
            preserve_existing = True
            status_rows = existing_status
            print(
                "Resuming incomplete feature extraction: "
                f"{sum(1 for row in status_rows if row.get('status') == 'ok')}/{n_rows} rows already ok."
            )
        else:
            status_rows = []
    else:
        status_rows = []

    readme_mm = ensure_memmap(readme_path, (n_rows, n_features), preserve_existing=preserve_existing)
    default_mm = ensure_memmap(default_path, (n_rows, n_features), preserve_existing=preserve_existing)
    prepared_by_id = {row["pdb_id"]: row for row in prepared_rows}
    tasks = []
    for index, row in dataset.reset_index(drop=True).iterrows():
        if status_rows and status_rows[index].get("status") == "ok":
            continue
        prepared = prepared_by_id.get(row["pdb_id"], {})
        tasks.append(
            {
                "row_index": index,
                "pdb_id": row["pdb_id"],
                "protein_file": str(REPO_ROOT / prepared.get("prepared_protein_pdb", "")),
                "ligand_file": str(REPO_ROOT / prepared.get("prepared_ligand_mol2", "")),
            }
        )

    if not status_rows:
        status_rows = [
            {
                "row_index": index,
                "pdb_id": row["pdb_id"],
                "status": "pending",
                "message": "",
                "nonzero_readme": 0,
                "nonzero_default": 0,
            }
            for index, row in dataset.reset_index(drop=True).iterrows()
        ]
    if not tasks:
        print("No incomplete feature rows remain; reusing cached matrices.")
        return {README_SOURCE: readme_path, DEFAULT_SOURCE: default_path}, status_rows
    print(f"Computing features for {len(tasks)}/{n_rows} rows with {args.workers} workers.")
    started = time.time()
    completed = 0
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_feature_worker,
        initargs=(str(args.drugwise_dir), base_names, feature_names),
    ) as pool:
        future_to_task = {pool.submit(feature_worker_task, task): task for task in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            result = future.result()
            index = task["row_index"]
            status_rows[index].update(
                {
                    "status": result["status"],
                    "message": result["message"],
                    "nonzero_readme": result["nonzero_readme"],
                    "nonzero_default": result["nonzero_default"],
                }
            )
            if result["status"] == "ok":
                readme_mm[index, :] = result[README_SOURCE]
                default_mm[index, :] = result[DEFAULT_SOURCE]
            completed += 1
            if completed == 1 or completed % args.progress_every == 0 or completed == n_rows:
                elapsed = time.time() - started
                rate = completed / elapsed if elapsed else 0
                remaining = (n_rows - completed) / rate if rate else float("nan")
                print(
                    f"features {completed}/{n_rows} "
                    f"({rate:.2f} complexes/s, eta {remaining/60:.1f} min)"
                )
                readme_mm.flush()
                default_mm.flush()
                write_tsv(
                    status_tsv,
                    status_rows,
                    [
                        "row_index",
                        "pdb_id",
                        "status",
                        "message",
                        "nonzero_readme",
                        "nonzero_default",
                    ],
                )
    readme_mm.flush()
    default_mm.flush()
    write_tsv(
        status_tsv,
        status_rows,
        ["row_index", "pdb_id", "status", "message", "nonzero_readme", "nonzero_default"],
    )
    return {README_SOURCE: readme_path, DEFAULT_SOURCE: default_path}, status_rows


def column_source_masks(feature_names: list[str], variant: dict[str, str]) -> dict[str, np.ndarray]:
    sources = [source_for_feature(feature, variant) for feature in feature_names]
    return {
        README_SOURCE: np.asarray([source == README_SOURCE for source in sources], dtype=bool),
        DEFAULT_SOURCE: np.asarray([source == DEFAULT_SOURCE for source in sources], dtype=bool),
    }


def build_fused_array(
    readme_mm: np.memmap,
    default_mm: np.memmap,
    row_indices: np.ndarray,
    feature_names: list[str],
    variant: dict[str, str],
) -> np.ndarray:
    masks = column_source_masks(feature_names, variant)
    x = np.empty((len(row_indices), len(feature_names)), dtype=np.float32)
    if masks[README_SOURCE].any():
        x[:, masks[README_SOURCE]] = readme_mm[row_indices][:, masks[README_SOURCE]]
    if masks[DEFAULT_SOURCE].any():
        x[:, masks[DEFAULT_SOURCE]] = default_mm[row_indices][:, masks[DEFAULT_SOURCE]]
    return x


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2:
        return None
    value = float(np.corrcoef(x.astype(float), y.astype(float))[0, 1])
    if math.isnan(value):
        return None
    return value


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    errors = y_pred - y_true
    return {
        "n": int(len(y_true)),
        "pearson": pearson(y_true, y_pred),
        "mae": float(np.mean(np.abs(errors))) if len(errors) else None,
        "rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
    }


def train_and_evaluate(
    dataset: pd.DataFrame,
    feature_paths: dict[str, Path],
    feature_status_rows: list[dict[str, Any]],
    feature_names: list[str],
    suffix_counts: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    model_template = joblib.load(args.model)
    params = model_template.get_params()
    if args.n_estimators_override is not None:
        params["n_estimators"] = args.n_estimators_override
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    n_rows = len(dataset)
    n_features = len(feature_names)
    readme_mm = np.memmap(feature_paths[README_SOURCE], dtype="float32", mode="r", shape=(n_rows, n_features))
    default_mm = np.memmap(feature_paths[DEFAULT_SOURCE], dtype="float32", mode="r", shape=(n_rows, n_features))
    ok_ids = {row["pdb_id"] for row in feature_status_rows if row["status"] == "ok"}
    ok_mask = dataset["pdb_id"].isin(ok_ids).to_numpy()
    train_mask = dataset[args.train_split].astype(bool).to_numpy() & ok_mask
    test_mask = dataset["is_core_test"].astype(bool).to_numpy() & ok_mask
    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(test_mask)
    y_train = dataset.loc[train_indices, "pK"].to_numpy(dtype=np.float64)
    y_test = dataset.loc[test_indices, "pK"].to_numpy(dtype=np.float64)

    variant = FUSION_VARIANTS[args.fusion_variant]
    started = time.time()
    print(
        f"Building fused train/test arrays for {args.train_split}: "
        f"train={len(train_indices)}, test={len(test_indices)}, features={n_features}"
    )
    x_train = build_fused_array(readme_mm, default_mm, train_indices, feature_names, variant)
    x_test = build_fused_array(readme_mm, default_mm, test_indices, feature_names, variant)
    print(f"Training GradientBoostingRegressor with params: {params}")
    model = GradientBoostingRegressor(**params)
    fit_started = time.time()
    model.fit(x_train, y_train)
    fit_seconds = time.time() - fit_started
    test_pred = model.predict(x_test)
    train_pred = model.predict(x_train)
    metrics = {
        "train": evaluate_predictions(y_train, train_pred),
        "test_core": evaluate_predictions(y_test, test_pred),
    }
    prediction_rows = []
    test_frame = dataset.loc[test_indices].copy().reset_index(drop=True)
    for row, prediction in zip(test_frame.to_dict(orient="records"), test_pred):
        prediction_rows.append(
            {
                "pdb_id": row["pdb_id"],
                "actual_pk": row["pK"],
                "predicted_pk": float(prediction),
                "error": float(prediction - row["pK"]),
                "train_split": args.train_split,
                "fusion_variant": args.fusion_variant,
            }
        )
    prediction_tsv = args.output_dir / "predictions" / f"{args.train_split}_{args.fusion_variant}_core_predictions.tsv"
    write_tsv(
        prediction_tsv,
        prediction_rows,
        ["pdb_id", "actual_pk", "predicted_pk", "error", "train_split", "fusion_variant"],
    )
    model_path = model_dir / f"{args.train_split}_{args.fusion_variant}_gbr.pkl"
    joblib.dump(model, model_path, compress=3)
    return {
        "train_split": args.train_split,
        "fusion_variant": args.fusion_variant,
        "variant_sources": variant,
        "model_hyperparameters": params,
        "feature_suffix_counts": suffix_counts,
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "fit_seconds": fit_seconds,
        "total_train_eval_seconds": time.time() - started,
        "metrics": metrics,
        "model_path": display_path(model_path),
        "prediction_tsv": display_path(prediction_tsv),
    }


def fmt_float(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    training = summary.get("training")
    lines = [
        "# PDBbind+ DrugWise Retraining Pipeline",
        "",
        f"Generated: `{summary['generated_at_utc']}` UTC.",
        "",
        "This report summarizes reconstruction of the paper's PDBbind v2020 train/test data from official PDBbind+ renewed website structures, extraction of two Lorentz feature matrices, and retraining with the hyperparameters stored in the downloaded DrugWise/GMI model.",
        "",
        "## Inputs",
        "",
        f"- PDBbind v2020 labels: `{summary['inputs']['v2020_general_csv']}`.",
        f"- Core test labels: `{summary['inputs']['core_csv']}`.",
        f"- Downloaded pretrained model: `{summary['inputs']['model']}`.",
        "",
        "PDBbind+ archives:",
    ]
    for archive in summary["inputs"]["pdbbind_plus_archives"]:
        lines.append(f"- `{archive}`.")
    lines.extend(
        [
            "",
            "## Dataset",
            "",
            f"- PDBbind v2020 general rows: `{summary['dataset']['v2020_general_rows']}`.",
            f"- CASF/core test rows: `{summary['dataset']['core_test_rows']}`.",
            f"- General-minus-core training rows: `{summary['dataset']['general_train_rows']}`.",
            f"- Refined archive rows: `{summary['dataset']['refined_rows']}`.",
            f"- Refined-minus-core training rows: `{summary['dataset']['refined_train_rows']}`.",
            f"- Prepared rows: `{summary['preparation']['prepared_rows']}`.",
            f"- Missing structure rows: `{summary['preparation']['missing_rows']}`.",
            "",
            "## Features",
            "",
            f"- Feature count from downloaded model schema: `{summary['features']['feature_count']}`.",
            f"- `_x` features: `{summary['features']['suffix_counts']['x']}`.",
            f"- `_y` features: `{summary['features']['suffix_counts']['y']}`.",
            f"- Unsuffixed features: `{summary['features']['suffix_counts']['unsuffixed']}`.",
            f"- Feature rows with status `ok`: `{summary['features']['ok_rows']}`.",
            f"- Feature rows with errors: `{summary['features']['error_rows']}`.",
            "",
            "Lorentz parameter sets:",
            "",
            f"- `{README_SOURCE}`: tau=1.5, power=2.5, cutoff=6.0.",
            f"- `{DEFAULT_SOURCE}`: tau=0.5, power=5.0, cutoff=6.0.",
            "",
        ]
    )
    if training:
        lines.extend(
            [
                "## Retraining",
                "",
                f"- Train split: `{training['train_split']}`.",
                f"- Fusion variant: `{training['fusion_variant']}`.",
                f"- Train rows used: `{training['train_rows']}`.",
                f"- Core test rows used: `{training['test_rows']}`.",
                f"- Fit time: `{training['fit_seconds']:.1f}` seconds.",
                f"- Model output: `{training['model_path']}`.",
                f"- Core prediction output: `{training['prediction_tsv']}`.",
                "",
                "| Scope | n | Pearson | MAE | RMSE |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for scope in ("train", "test_core"):
            block = training["metrics"][scope]
            lines.append(
                f"| {scope} | {block['n']} | {fmt_float(block['pearson'])} | {fmt_float(block['mae'])} | {fmt_float(block['rmse'])} |"
            )
    else:
        lines.extend(
            [
                "## Retraining",
                "",
                "Training was skipped for this run.",
            ]
        )
    lines.extend(
        [
            "",
            "## Local Outputs",
            "",
            f"- Dataset manifest: `{summary['outputs']['dataset_tsv']}`.",
            f"- Structure preparation status: `{summary['outputs']['preparation_tsv']}`.",
            f"- Feature status: `{summary['outputs']['feature_status_tsv']}`.",
            f"- Feature memmaps: `{summary['outputs']['feature_dir']}`.",
            "",
            "Large arrays, extracted structures, and model artifacts are stored under ignored `data/interim/` paths.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2020-general-csv", type=Path, default=DEFAULT_V2020_GENERAL_CSV)
    parser.add_argument("--core-csv", type=Path, default=DEFAULT_CORE_CSV)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--drugwise-dir", type=Path, default=DEFAULT_DRUGWISE_DIR)
    parser.add_argument("--pdbbind-plus-tar", type=Path, action="append", default=[DEFAULT_PDBBIND_PLUS_REFINED_TAR, DEFAULT_PDBBIND_PLUS_OTHER_PL_TAR])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--train-split", choices=["is_general_train", "is_refined_train"], default="is_general_train")
    parser.add_argument("--fusion-variant", choices=sorted(FUSION_VARIANTS), default="x_default_y_readme_unsuff_default")
    parser.add_argument("--n-estimators-override", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_names, base_names, suffix_counts = load_model_feature_schema(args.model)

    archive_index, refined_ids = build_archive_index(args.pdbbind_plus_tar)
    dataset = load_dataset_rows(args.v2020_general_csv, args.core_csv, refined_ids)
    dataset_tsv = args.output_dir / "pdbbind_v2020_paper_dataset.tsv"
    dataset.to_csv(dataset_tsv, sep="\t", index=False)

    prepared_dir = args.output_dir / "prepared"
    preparation_rows = extract_needed_structures(
        dataset,
        archive_index,
        prepared_dir,
        force=args.force_extract,
    )
    preparation_tsv = args.output_dir / "pdbbind_v2020_pdbbind_plus_preparation_status.tsv"
    write_tsv(
        preparation_tsv,
        preparation_rows,
        [
            "pdb_id",
            "pK",
            "is_core_test",
            "is_general_train",
            "is_refined",
            "is_refined_train",
            "structure_source",
            "protein_member",
            "ligand_member",
            "prepared_protein_pdb",
            "prepared_ligand_mol2",
            "preparation_status",
            "preparation_message",
        ],
    )

    feature_paths = {
        README_SOURCE: args.output_dir / "features" / f"{README_SOURCE}.float32.memmap",
        DEFAULT_SOURCE: args.output_dir / "features" / f"{DEFAULT_SOURCE}.float32.memmap",
    }
    feature_status_rows: list[dict[str, Any]] = []
    feature_status_tsv = args.output_dir / "features" / "feature_status.tsv"
    if args.skip_features:
        if not feature_status_tsv.exists():
            raise FileNotFoundError("--skip-features requested but feature_status.tsv does not exist")
        feature_status_rows = read_tsv(feature_status_tsv)
    else:
        feature_paths, feature_status_rows = compute_feature_matrices(
            dataset,
            preparation_rows,
            feature_names,
            base_names,
            args,
        )

    training_summary = None
    if not args.skip_training:
        training_summary = train_and_evaluate(
            dataset,
            feature_paths,
            feature_status_rows,
            feature_names,
            suffix_counts,
            args,
        )

    prepared_count = sum(1 for row in preparation_rows if row["preparation_status"] == "prepared")
    ok_feature_count = sum(1 for row in feature_status_rows if row.get("status") == "ok")
    error_feature_count = sum(1 for row in feature_status_rows if row.get("status") != "ok")
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "v2020_general_csv": display_path(args.v2020_general_csv),
            "core_csv": display_path(args.core_csv),
            "model": display_path(args.model),
            "drugwise_dir": display_path(args.drugwise_dir),
            "pdbbind_plus_archives": [display_path(path) for path in args.pdbbind_plus_tar],
        },
        "dataset": {
            "v2020_general_rows": int(len(dataset)),
            "core_test_rows": int(dataset["is_core_test"].sum()),
            "general_train_rows": int(dataset["is_general_train"].sum()),
            "refined_rows": int(dataset["is_refined"].sum()),
            "refined_train_rows": int(dataset["is_refined_train"].sum()),
        },
        "preparation": {
            "prepared_rows": prepared_count,
            "missing_rows": len(preparation_rows) - prepared_count,
            "status_counts": dict(Counter(row["preparation_status"] for row in preparation_rows)),
            "source_counts": dict(Counter(row["structure_source"] for row in preparation_rows)),
        },
        "features": {
            "feature_count": len(feature_names),
            "suffix_counts": suffix_counts,
            "parameter_sets": PARAMETER_SETS,
            "ok_rows": ok_feature_count,
            "error_rows": error_feature_count,
        },
        "training": training_summary,
        "outputs": {
            "dataset_tsv": display_path(dataset_tsv),
            "preparation_tsv": display_path(preparation_tsv),
            "feature_status_tsv": display_path(feature_status_tsv),
            "feature_dir": display_path(args.output_dir / "features"),
            "summary_json": display_path(args.summary_json),
            "summary_md": display_path(args.summary_md),
        },
    }
    write_json(args.summary_json, summary)
    write_summary_md(args.summary_md, summary)
    print(
        "PDBbind+ DrugWise retraining pipeline complete: "
        f"prepared={prepared_count}/{len(preparation_rows)}, "
        f"features_ok={ok_feature_count}, "
        f"training={'done' if training_summary else 'skipped'}"
    )
    if training_summary:
        test_metrics = training_summary["metrics"]["test_core"]
        print(
            f"Core test: n={test_metrics['n']} "
            f"Pearson={fmt_float(test_metrics['pearson'])} "
            f"MAE={fmt_float(test_metrics['mae'])} "
            f"RMSE={fmt_float(test_metrics['rmse'])}"
        )
    print(f"Wrote {display_path(args.summary_md)}")


if __name__ == "__main__":
    main()
