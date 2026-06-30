#!/usr/bin/env python3
"""Train PDBbind v2020 DrugWise/GMI models from existing kernel feature memmaps.

This script assumes `scripts/pdbbind_plus_drugwise_retrain.py` has already prepared
the PDBbind v2020 dataset and extracted the two selected 11,000-feature Lorentz
memmaps. It fits one requested feature variant without recomputing structures or
features:

- `readme_lorentz_all_selected`: all selected features from tau=1.5, power=2.5.
- `default_lorentz_all_selected`: all selected features from tau=0.5, power=5.0.
- `both_lorentz_concat`: 22,000 columns, README features followed by default features.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from allosteric_affinity.drugwise import DEFAULT_SOURCE, README_SOURCE  # noqa: E402

DEFAULT_RUN_DIR = REPO_ROOT / "data/interim/drugwise_paper_retrain/pdbbind_v2020"
DEFAULT_MODEL = REPO_ROOT / "DrugWise-Implementation-main/model/pretraiened_ggl_mb_score.pkl"

SOURCE_LABELS = {
    README_SOURCE: "README-style Lorentz tau=1.5 power=2.5 cutoff=6.0",
    DEFAULT_SOURCE: "code-default Lorentz tau=0.5 power=5.0 cutoff=6.0",
}

VARIANTS = {
    "readme_lorentz_all_selected": {
        "description": "All 11,000 selected model-schema features from README-style Lorentz parameters.",
        "sources": [README_SOURCE],
    },
    "default_lorentz_all_selected": {
        "description": "All 11,000 selected model-schema features from code-default Lorentz parameters.",
        "sources": [DEFAULT_SOURCE],
    },
    "both_lorentz_concat": {
        "description": "Concatenated 22,000-column matrix: README-style Lorentz features followed by code-default Lorentz features.",
        "sources": [README_SOURCE, DEFAULT_SOURCE],
    },
}

PAPER_GB_PRESETS: dict[str, dict[str, Any]] = {
    "model1_template": {},
    "model2_paper": {
        "n_estimators": 15000,
        "max_features": "sqrt",
        "max_depth": 7,
        "min_samples_split": 16,
        "min_samples_leaf": 10,
        "learning_rate": 0.008,
        "subsample": 0.4,
        "loss": "squared_error",
    },
    "model3_paper": {
        "n_estimators": 24500,
        "max_features": "sqrt",
        "max_depth": 7,
        "min_samples_split": 3,
        "min_samples_leaf": 19,
        "learning_rate": 0.03,
        "subsample": 0.4,
        "loss": "squared_error",
    },
    "model4_paper": {
        "n_estimators": 23500,
        "max_features": "sqrt",
        "max_depth": 6,
        "min_samples_split": 16,
        "min_samples_leaf": 8,
        "learning_rate": 0.006,
        "subsample": 0.7,
        "loss": "squared_error",
    },
}

PAPER_GB_PRESET_DESCRIPTIONS = {
    "model1_template": "Downloaded model/template hyperparameters, corresponding to paper Table A1 Model 1.",
    "model2_paper": "Paper Table A1 Model 2 boosting hyperparameters.",
    "model3_paper": "Paper Table A1 Model 3 boosting hyperparameters.",
    "model4_paper": "Paper Table A1 Model 4 boosting hyperparameters.",
}


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    return frame.to_dict(orient="records")


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if rows:
        frame = frame.reindex(columns=fieldnames)
    else:
        frame = pd.DataFrame(columns=fieldnames)
    frame.to_csv(path, sep="\t", index=False)


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2:
        return None
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int | None]:
    error = y_pred - y_true
    return {
        "n": int(len(y_true)),
        "pearson": pearson(y_true, y_pred),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


def load_model_template(
    model_path: Path,
    gb_preset: str,
    n_estimators_override: int | None,
    model_verbose: int | None,
) -> tuple[dict[str, Any], list[str]]:
    model_template = joblib.load(model_path)
    params = model_template.get_params()
    params.update(PAPER_GB_PRESETS[gb_preset])
    if n_estimators_override is not None:
        params["n_estimators"] = n_estimators_override
    if model_verbose is not None:
        params["verbose"] = model_verbose
    feature_names = [str(name) for name in model_template.feature_names_in_]
    del model_template
    gc.collect()
    return params, feature_names


def load_inputs(args: argparse.Namespace, feature_names: list[str]) -> tuple[pd.DataFrame, list[dict[str, str]], dict[str, np.memmap]]:
    dataset_tsv = args.run_dir / "pdbbind_v2020_paper_dataset.tsv"
    feature_status_tsv = args.run_dir / "features/feature_status.tsv"
    dataset = pd.read_csv(dataset_tsv, sep="\t")
    status_rows = read_tsv(feature_status_tsv)

    n_rows = len(dataset)
    n_features = len(feature_names)
    feature_paths = {
        README_SOURCE: args.run_dir / "features" / f"{README_SOURCE}.float32.memmap",
        DEFAULT_SOURCE: args.run_dir / "features" / f"{DEFAULT_SOURCE}.float32.memmap",
    }
    memmaps = {
        name: np.memmap(path, dtype="float32", mode="r", shape=(n_rows, n_features))
        for name, path in feature_paths.items()
    }
    return dataset, status_rows, memmaps


def variant_feature_names(variant: str, base_feature_names: list[str]) -> list[str]:
    sources = VARIANTS[variant]["sources"]
    if len(sources) == 1:
        return list(base_feature_names)
    names: list[str] = []
    for source in sources:
        names.extend([f"{source}__{feature}" for feature in base_feature_names])
    return names


def build_variant_array(
    memmaps: dict[str, np.memmap],
    row_indices: np.ndarray,
    variant: str,
) -> np.ndarray:
    sources = VARIANTS[variant]["sources"]
    if len(sources) == 1:
        return np.asarray(memmaps[sources[0]][row_indices, :], dtype=np.float32)

    n_rows = len(row_indices)
    n_features = memmaps[sources[0]].shape[1]
    x = np.empty((n_rows, n_features * len(sources)), dtype=np.float32)
    offset = 0
    for source in sources:
        x[:, offset : offset + n_features] = memmaps[source][row_indices, :]
        offset += n_features
    return x


def write_feature_schema(path: Path, variant: str, base_feature_names: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    output_index = 0
    for source in VARIANTS[variant]["sources"]:
        for source_index, feature_name in enumerate(base_feature_names):
            rows.append(
                {
                    "output_feature_index": output_index,
                    "source": source,
                    "source_feature_index": source_index,
                    "source_feature_name": feature_name,
                    "output_feature_name": (
                        feature_name
                        if len(VARIANTS[variant]["sources"]) == 1
                        else f"{source}__{feature_name}"
                    ),
                }
            )
            output_index += 1
    write_tsv(
        path,
        rows,
        [
            "output_feature_index",
            "source",
            "source_feature_index",
            "source_feature_name",
            "output_feature_name",
        ],
    )


def train_variant(args: argparse.Namespace) -> dict[str, Any]:
    if args.variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {args.variant}")
    if args.gb_preset not in PAPER_GB_PRESETS:
        raise ValueError(f"Unknown GB preset: {args.gb_preset}")

    params, base_feature_names = load_model_template(
        args.model,
        args.gb_preset,
        args.n_estimators_override,
        args.model_verbose,
    )
    dataset, status_rows, memmaps = load_inputs(args, base_feature_names)

    ok_ids = {row["pdb_id"] for row in status_rows if row.get("status") == "ok"}
    ok_mask = dataset["pdb_id"].isin(ok_ids).to_numpy()
    train_mask = dataset[args.train_split].astype(bool).to_numpy() & ok_mask
    test_mask = dataset["is_core_test"].astype(bool).to_numpy() & ok_mask
    all_ok_indices = np.flatnonzero(ok_mask)
    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(test_mask)

    y_train = dataset.loc[train_indices, "pK"].to_numpy(dtype=np.float64)
    y_test = dataset.loc[test_indices, "pK"].to_numpy(dtype=np.float64)

    output_feature_names = variant_feature_names(args.variant, base_feature_names)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "models"
    prediction_dir = args.output_dir / "predictions"
    schema_dir = args.output_dir / "feature_schemas"
    summary_dir = args.output_dir / "summaries"
    for directory in (model_dir, prediction_dir, schema_dir, summary_dir):
        directory.mkdir(parents=True, exist_ok=True)

    run_name = (
        f"{args.train_split}_{args.variant}"
        if args.gb_preset == "model1_template"
        else f"{args.train_split}_{args.variant}_{args.gb_preset}"
    )

    schema_path = schema_dir / f"{run_name}_features.tsv"
    write_feature_schema(schema_path, args.variant, base_feature_names)

    started = time.time()
    print(
        f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] "
        f"Building arrays for {run_name}: train={len(train_indices)}, "
        f"test={len(test_indices)}, features={len(output_feature_names)}",
        flush=True,
    )
    x_train = build_variant_array(memmaps, train_indices, args.variant)
    x_test = build_variant_array(memmaps, test_indices, args.variant)

    print(
        f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] "
        f"Training GradientBoostingRegressor for {run_name} with params={params}",
        flush=True,
    )
    model = GradientBoostingRegressor(**params)
    fit_started = time.time()
    model.fit(x_train, y_train)
    fit_seconds = time.time() - fit_started

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    metrics = {
        "train": evaluate_predictions(y_train, train_pred),
        "test_core": evaluate_predictions(y_test, test_pred),
    }
    del x_train, x_test
    gc.collect()

    model_path = model_dir / f"{run_name}_gbr.pkl"
    print(
        f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] "
        f"Saving {display_path(model_path)}",
        flush=True,
    )
    joblib.dump(model, model_path, compress=args.joblib_compress)

    test_rows = []
    for row, prediction in zip(dataset.loc[test_indices].to_dict(orient="records"), test_pred):
        test_rows.append(
            {
                "pdb_id": row["pdb_id"],
                "actual_pk": row["pK"],
                "predicted_pk": float(prediction),
                "error": float(prediction - row["pK"]),
                "split": "test_core",
                "train_split": args.train_split,
                "feature_variant": args.variant,
            }
        )

    all_x = build_variant_array(memmaps, all_ok_indices, args.variant)
    all_pred = model.predict(all_x)
    model.feature_names_in_ = np.asarray(output_feature_names, dtype=object)
    all_rows = []
    for row, prediction in zip(dataset.loc[all_ok_indices].to_dict(orient="records"), all_pred):
        split = "test_core" if bool(row["is_core_test"]) else "train"
        all_rows.append(
            {
                "pdb_id": row["pdb_id"],
                "actual_pk": row["pK"],
                "predicted_pk": float(prediction),
                "error": float(prediction - row["pK"]),
                "split": split,
                "train_split": args.train_split,
                "feature_variant": args.variant,
            }
        )

    core_prediction_tsv = prediction_dir / f"{run_name}_core_predictions.tsv"
    all_prediction_tsv = prediction_dir / f"{run_name}_all_ok_predictions.tsv"
    prediction_fields = [
        "pdb_id",
        "actual_pk",
        "predicted_pk",
        "error",
        "split",
        "train_split",
        "feature_variant",
    ]
    write_tsv(core_prediction_tsv, test_rows, prediction_fields)
    write_tsv(all_prediction_tsv, all_rows, prediction_fields)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "variant": args.variant,
        "variant_description": VARIANTS[args.variant]["description"],
        "run_name": run_name,
        "gb_preset": args.gb_preset,
        "gb_preset_description": PAPER_GB_PRESET_DESCRIPTIONS[args.gb_preset],
        "source_labels": {source: SOURCE_LABELS[source] for source in VARIANTS[args.variant]["sources"]},
        "train_split": args.train_split,
        "counts": {
            "dataset_rows": int(len(dataset)),
            "ok_feature_rows": int(ok_mask.sum()),
            "train_rows": int(len(train_indices)),
            "test_core_rows": int(len(test_indices)),
            "input_features": int(len(output_feature_names)),
        },
        "model_hyperparameters": params,
        "metrics": metrics,
        "timing": {
            "fit_seconds": fit_seconds,
            "total_seconds": time.time() - started,
        },
        "inputs": {
            "run_dir": display_path(args.run_dir),
            "model_template": display_path(args.model),
            "feature_status_tsv": display_path(args.run_dir / "features/feature_status.tsv"),
            "dataset_tsv": display_path(args.run_dir / "pdbbind_v2020_paper_dataset.tsv"),
        },
        "outputs": {
            "model_path": display_path(model_path),
            "feature_schema_tsv": display_path(schema_path),
            "core_prediction_tsv": display_path(core_prediction_tsv),
            "all_prediction_tsv": display_path(all_prediction_tsv),
        },
    }

    summary_json = summary_dir / f"{run_name}_summary.json"
    summary_md = summary_dir / f"{run_name}_summary.md"
    summary["outputs"]["summary_json"] = display_path(summary_json)
    summary["outputs"]["summary_md"] = display_path(summary_md)
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary_md.write_text(render_markdown(summary), encoding="utf-8")

    print(
        f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] "
        f"Done {run_name}: test Pearson={fmt_float(metrics['test_core']['pearson'])}, "
        f"MAE={fmt_float(metrics['test_core']['mae'])}, RMSE={fmt_float(metrics['test_core']['rmse'])}",
        flush=True,
    )
    return summary


def fmt_float(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def render_markdown(summary: dict[str, Any]) -> str:
    train = summary["metrics"]["train"]
    test = summary["metrics"]["test_core"]
    lines = [
        f"# PDBbind v2020 kernel-variant training: {summary['run_name']}",
        "",
        summary["variant_description"],
        "",
        f"Gradient-boosting preset: `{summary['gb_preset']}`. {summary['gb_preset_description']}",
        "",
        "## Sources",
        "",
    ]
    for source, label in summary["source_labels"].items():
        lines.append(f"- `{source}`: {label}.")
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- Dataset rows: `{summary['counts']['dataset_rows']}`.",
            f"- Feature-ok rows: `{summary['counts']['ok_feature_rows']}`.",
            f"- Training rows: `{summary['counts']['train_rows']}`.",
            f"- Core test rows: `{summary['counts']['test_core_rows']}`.",
            f"- Input features: `{summary['counts']['input_features']}`.",
            "",
            "## Metrics",
            "",
            "| Scope | n | Pearson | MAE | RMSE |",
            "|---|---:|---:|---:|---:|",
            f"| Train | {train['n']} | {fmt_float(train['pearson'])} | {fmt_float(train['mae'])} | {fmt_float(train['rmse'])} |",
            f"| CASF/core test | {test['n']} | {fmt_float(test['pearson'])} | {fmt_float(test['mae'])} | {fmt_float(test['rmse'])} |",
            "",
            "## Outputs",
            "",
            f"- Model: `{summary['outputs']['model_path']}`.",
            f"- Feature schema: `{summary['outputs']['feature_schema_tsv']}`.",
            f"- Core predictions: `{summary['outputs']['core_prediction_tsv']}`.",
            f"- All feature-ok predictions: `{summary['outputs']['all_prediction_tsv']}`.",
            f"- Fit seconds: `{summary['timing']['fit_seconds']:.1f}`.",
            f"- Total seconds: `{summary['timing']['total_seconds']:.1f}`.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RUN_DIR / "kernel_variant_training",
    )
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument(
        "--gb-preset",
        choices=sorted(PAPER_GB_PRESETS),
        default="model1_template",
    )
    parser.add_argument(
        "--train-split",
        choices=["is_general_train", "is_refined_train"],
        default="is_general_train",
    )
    parser.add_argument("--n-estimators-override", type=int, default=None)
    parser.add_argument("--model-verbose", type=int, default=None)
    parser.add_argument("--joblib-compress", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_variant(args)


if __name__ == "__main__":
    main()
