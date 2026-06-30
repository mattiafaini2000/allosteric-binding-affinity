#!/usr/bin/env python3
"""Fuse two Lorentz-parameter DrugWise feature matrices for ASD Tier-A inference.

The pretrained DrugWise/GMI model contains selected feature names with pandas
merge suffixes (`_x` and `_y`). The public inference script cannot recover which
upstream feature table produced each suffixed column, so the first local
inference runs filled both suffixes from one parameterization at a time.

This diagnostic script tests whether the suffixes could correspond to the two
Lorentz parameterizations already generated locally:

- README-style Lorentz: tau=1.5, power=2.5, cutoff=6.0
- public-code-default Lorentz: tau=0.5, power=5.0, cutoff=6.0

It builds fused selected-feature matrices by assigning one parameter set to
`_x`, the other to `_y`, and then swapping them. Because the pretrained artifact
does not reveal the source for unsuffixed features, both unsuffixed-source
choices are also tested.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from allosteric_affinity.drugwise import (  # noqa: E402
    DEFAULT_SOURCE,
    README_SOURCE,
    source_for_feature,
)

DEFAULT_MODEL = REPO_ROOT / "DrugWise-Implementation-main/model/pretraiened_ggl_mb_score.pkl"
DEFAULT_README_MATRIX = (
    REPO_ROOT
    / "data/interim/drugwise/tier_a_inference/asd_tier_a_selected_feature_matrix.tsv"
)
DEFAULT_DEFAULT_MATRIX = (
    REPO_ROOT
    / "data/interim/drugwise/tier_a_inference_tau0p5_power5/"
    / "asd_tier_a_selected_feature_matrix.tsv"
)
DEFAULT_README_PREDICTIONS = (
    REPO_ROOT
    / "data/interim/drugwise/tier_a_inference/"
    / "asd_tier_a_drugwise_inference_predictions.tsv"
)
DEFAULT_DEFAULT_PREDICTIONS = (
    REPO_ROOT
    / "data/interim/drugwise/tier_a_inference_tau0p5_power5/"
    / "asd_tier_a_drugwise_inference_predictions.tsv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/interim/drugwise/tier_a_lorentz_parameter_fusion"
DEFAULT_SUMMARY_JSON = (
    REPO_ROOT / "outputs/asd_tier_a_drugwise_lorentz_parameter_fusion_summary.json"
)
DEFAULT_SUMMARY_MD = (
    REPO_ROOT / "outputs/asd_tier_a_drugwise_lorentz_parameter_fusion_summary.md"
)

PDBBINDPP_SOURCE = "pdbbindpp_2020_refined"

VARIANTS = [
    {
        "variant_id": "x_readme_y_default_unsuff_readme",
        "x_source": README_SOURCE,
        "y_source": DEFAULT_SOURCE,
        "unsuffixed_source": README_SOURCE,
    },
    {
        "variant_id": "x_default_y_readme_unsuff_readme",
        "x_source": DEFAULT_SOURCE,
        "y_source": README_SOURCE,
        "unsuffixed_source": README_SOURCE,
    },
    {
        "variant_id": "x_readme_y_default_unsuff_default",
        "x_source": README_SOURCE,
        "y_source": DEFAULT_SOURCE,
        "unsuffixed_source": DEFAULT_SOURCE,
    },
    {
        "variant_id": "x_default_y_readme_unsuff_default",
        "x_source": DEFAULT_SOURCE,
        "y_source": README_SOURCE,
        "unsuffixed_source": DEFAULT_SOURCE,
    },
]


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


def load_selected_matrix(path: Path, source_name: str, feature_names: list[str]) -> pd.DataFrame:
    matrix = pd.read_csv(path, sep="\t")
    if "row_complex_id" not in matrix.columns:
        raise ValueError(f"{path} is missing row_complex_id")
    if matrix["row_complex_id"].duplicated().any():
        duplicated = matrix.loc[matrix["row_complex_id"].duplicated(), "row_complex_id"]
        raise ValueError(f"{path} has duplicated row_complex_id values: {duplicated.tolist()}")
    missing = [feature for feature in feature_names if feature not in matrix.columns]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"{path} is missing {len(missing)} model features for {source_name}: {preview}")
    return matrix.set_index("row_complex_id", drop=False)


def align_matrices(
    readme_matrix: pd.DataFrame,
    default_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    readme_ids = set(readme_matrix.index)
    default_ids = set(default_matrix.index)
    if readme_ids != default_ids:
        only_readme = sorted(readme_ids - default_ids)
        only_default = sorted(default_ids - readme_ids)
        raise ValueError(
            "Selected matrices have different row_complex_id sets. "
            f"Only README: {only_readme}; only default: {only_default}"
        )
    default_matrix = default_matrix.loc[readme_matrix.index]
    return readme_matrix, default_matrix


def build_fused_matrix(
    sources: dict[str, pd.DataFrame],
    feature_names: list[str],
    variant: dict[str, str],
) -> pd.DataFrame:
    first_source = sources[variant["unsuffixed_source"]]
    columns: dict[str, pd.Series] = {"row_complex_id": first_source["row_complex_id"]}
    for feature in feature_names:
        source_name = source_for_feature(feature, variant)
        columns[feature] = sources[source_name][feature]
    matrix = pd.DataFrame(columns)
    return matrix[["row_complex_id", *feature_names]]


def pearson(x: pd.Series, y: pd.Series) -> float | None:
    if len(x) < 2:
        return None
    value = float(np.corrcoef(x.astype(float), y.astype(float))[0, 1])
    if math.isnan(value):
        return None
    return value


def compute_metrics(
    frame: pd.DataFrame,
    prediction_column: str,
    label: str,
    scope: str,
) -> dict[str, Any]:
    actual = frame["actual_pk"].astype(float)
    predicted = frame[prediction_column].astype(float)
    errors = predicted - actual
    return {
        "label": label,
        "scope": scope,
        "rows": int(len(frame)),
        "unique_pdb_ids": int(frame["pdb_id"].str.upper().nunique()),
        "pearson_vs_actual": pearson(actual, predicted),
        "mae_vs_actual": float(np.mean(np.abs(errors))),
        "rmse_vs_actual": float(np.sqrt(np.mean(errors**2))),
    }


def compute_unique_pdb_metrics(
    frame: pd.DataFrame,
    prediction_column: str,
    label: str,
    scope: str,
) -> dict[str, Any]:
    grouped = (
        frame.assign(pdb_id=frame["pdb_id"].str.upper())
        .groupby("pdb_id", as_index=False)
        .agg(actual_pk=("actual_pk", "mean"), predicted=(prediction_column, "mean"))
    )
    actual = grouped["actual_pk"].astype(float)
    predicted = grouped["predicted"].astype(float)
    errors = predicted - actual
    return {
        "label": label,
        "scope": scope,
        "rows": int(len(frame)),
        "unique_pdb_ids": int(len(grouped)),
        "pearson_vs_actual": pearson(actual, predicted),
        "mae_vs_actual": float(np.mean(np.abs(errors))),
        "rmse_vs_actual": float(np.sqrt(np.mean(errors**2))),
    }


def metrics_for_prediction_frame(
    frame: pd.DataFrame,
    prediction_column: str,
    label: str,
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    scopes = [
        ("row_weighted_all", frame),
        ("unique_pdb_all", frame),
        ("row_weighted_pdbbindpp_only", frame[frame["structure_source"].eq(PDBBINDPP_SOURCE)]),
        ("unique_pdb_pdbbindpp_only", frame[frame["structure_source"].eq(PDBBINDPP_SOURCE)]),
    ]
    for scope, scoped in scopes:
        if scoped.empty:
            continue
        if scope.startswith("unique_pdb"):
            metrics.append(compute_unique_pdb_metrics(scoped, prediction_column, label, scope))
        else:
            metrics.append(compute_metrics(scoped, prediction_column, label, scope))
    return metrics


def build_prediction_rows(
    template: pd.DataFrame,
    variant: dict[str, str],
    predictions: np.ndarray,
) -> pd.DataFrame:
    rows = template.copy()
    rows.insert(0, "variant_id", variant["variant_id"])
    rows.insert(1, "x_source", variant["x_source"])
    rows.insert(2, "y_source", variant["y_source"])
    rows.insert(3, "unsuffixed_source", variant["unsuffixed_source"])
    rows["model_predicted_pk"] = predictions
    rows["model_error_vs_actual"] = rows["model_predicted_pk"].astype(float) - rows[
        "actual_pk"
    ].astype(float)
    rows["model_delta_vs_paper_prediction"] = rows["model_predicted_pk"].astype(float) - rows[
        "paper_predicted_pk"
    ].astype(float)
    rows["abs_model_error_vs_actual"] = rows["model_error_vs_actual"].abs()
    return rows


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    unique_metrics = [
        metric
        for metric in summary["comparison_metrics"]
        if metric["scope"] == "unique_pdb_all"
    ]
    pdbbindpp_metrics = [
        metric
        for metric in summary["comparison_metrics"]
        if metric["scope"] == "unique_pdb_pdbbindpp_only"
    ]
    fused_unique_metrics = [
        metric for metric in unique_metrics if metric["label"].startswith("x_")
    ]
    fused_pdbbindpp_metrics = [
        metric for metric in pdbbindpp_metrics if metric["label"].startswith("x_")
    ]
    best_all_pearson = max(
        fused_unique_metrics,
        key=lambda metric: metric["pearson_vs_actual"]
        if metric["pearson_vs_actual"] is not None
        else float("-inf"),
    )
    best_all_mae = min(fused_unique_metrics, key=lambda metric: metric["mae_vs_actual"])
    best_pdbbindpp_pearson = max(
        fused_pdbbindpp_metrics,
        key=lambda metric: metric["pearson_vs_actual"]
        if metric["pearson_vs_actual"] is not None
        else float("-inf"),
    )
    best_pdbbindpp_mae = min(
        fused_pdbbindpp_metrics, key=lambda metric: metric["mae_vs_actual"]
    )

    def metric_row(metric: dict[str, Any]) -> str:
        pearson_value = metric["pearson_vs_actual"]
        pearson_text = "n/a" if pearson_value is None else f"{pearson_value:.4f}"
        return (
            f"| {metric['label']} | {metric['rows']} | {metric['unique_pdb_ids']} | "
            f"{pearson_text} | {metric['mae_vs_actual']:.4f} | "
            f"{metric['rmse_vs_actual']:.4f} |"
        )

    lines = [
        "# ASD Tier-A DrugWise Lorentz Parameter Fusion",
        "",
        f"Generated: `{summary['generated_at_utc']}` UTC.",
        "",
        "This diagnostic fuses two local Lorentz feature matrices for the 21 strict Tier-A ASD rows that overlap the paper's PDBbind v2020 appendix test predictions.",
        "",
        "## Feature Sources",
        "",
        f"- `{README_SOURCE}`: Lorentz kernel, tau=1.5, power=2.5, cutoff=6.0.",
        f"- `{DEFAULT_SOURCE}`: Lorentz kernel, tau=0.5, power=5.0, cutoff=6.0.",
        "",
        "## Variant Logic",
        "",
        "For model features ending in `_x`, the value is taken from the variant's x source. For model features ending in `_y`, the value is taken from the variant's y source. Unsuffixed model features are tested with both possible parameter sources because the released model artifact does not identify their origin.",
        "",
        "Feature suffix counts:",
        "",
        f"- `_x`: {summary['feature_suffix_counts']['x']}.",
        f"- `_y`: {summary['feature_suffix_counts']['y']}.",
        f"- unsuffixed: {summary['feature_suffix_counts']['unsuffixed']}.",
        "",
        "## Unique-PDB Metrics",
        "",
        "| Run | Rows | Unique PDB IDs | Pearson vs actual | MAE | RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(metric_row(metric) for metric in unique_metrics)
    lines.extend(
        [
            "",
            "## PDBbind++-Only Unique-PDB Metrics",
            "",
            "| Run | Rows | Unique PDB IDs | Pearson vs actual | MAE | RMSE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(metric_row(metric) for metric in pdbbindpp_metrics)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The fused `_x`/`_y` assignments do not reproduce the paper appendix predictions.",
            "",
            f"On all 20 unique PDB IDs, the best fused Pearson is `{best_all_pearson['pearson_vs_actual']:.4f}` from `{best_all_pearson['label']}`, while the best fused MAE is `{best_all_mae['mae_vs_actual']:.4f}` from `{best_all_mae['label']}`.",
            "",
            f"On the 17 unique PDBbind++-prepared IDs, the best fused Pearson is `{best_pdbbindpp_pearson['pearson_vs_actual']:.4f}` from `{best_pdbbindpp_pearson['label']}`, and the best fused MAE is `{best_pdbbindpp_mae['mae_vs_actual']:.4f}` from `{best_pdbbindpp_mae['label']}`.",
            "",
            "The swap direction `_x=default, _y=README` is consistently better for Pearson than `_x=README, _y=default`, especially on the PDBbind++-only subset. The unsuffixed-source choice mostly controls the prediction scale and error metrics. This is useful evidence, but still not a clean recovery of the original feature provenance.",
            "",
            "## Local Outputs",
            "",
            f"- Long prediction table: `{summary['outputs']['predictions_long_tsv']}`.",
            f"- Wide prediction table: `{summary['outputs']['predictions_wide_tsv']}`.",
            f"- Metrics table: `{summary['outputs']['metrics_tsv']}`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--readme-matrix", type=Path, default=DEFAULT_README_MATRIX)
    parser.add_argument("--default-matrix", type=Path, default=DEFAULT_DEFAULT_MATRIX)
    parser.add_argument("--readme-predictions", type=Path, default=DEFAULT_README_PREDICTIONS)
    parser.add_argument("--default-predictions", type=Path, default=DEFAULT_DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = joblib.load(args.model)
    feature_names = list(model.feature_names_in_)
    readme_matrix = load_selected_matrix(args.readme_matrix, README_SOURCE, feature_names)
    default_matrix = load_selected_matrix(args.default_matrix, DEFAULT_SOURCE, feature_names)
    readme_matrix, default_matrix = align_matrices(readme_matrix, default_matrix)
    sources = {
        README_SOURCE: readme_matrix,
        DEFAULT_SOURCE: default_matrix,
    }

    template = pd.read_csv(args.readme_predictions, sep="\t").set_index("row_complex_id", drop=False)
    template = template.loc[readme_matrix.index].reset_index(drop=True)
    default_predictions = pd.read_csv(args.default_predictions, sep="\t")

    prediction_frames: list[pd.DataFrame] = []
    for variant in VARIANTS:
        matrix = build_fused_matrix(sources, feature_names, variant)
        predictions = model.predict(matrix[feature_names])
        prediction_frames.append(build_prediction_rows(template, variant, predictions))

    predictions_long = pd.concat(prediction_frames, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_long_tsv = args.output_dir / "asd_tier_a_lorentz_parameter_fusion_predictions.tsv"
    predictions_wide_tsv = args.output_dir / "asd_tier_a_lorentz_parameter_fusion_predictions_wide.tsv"
    metrics_tsv = args.output_dir / "asd_tier_a_lorentz_parameter_fusion_metrics.tsv"

    predictions_long.to_csv(predictions_long_tsv, sep="\t", index=False)

    metadata_columns = [
        "row_complex_id",
        "pdb_id",
        "structure_source",
        "actual_pk",
        "paper_predicted_pk",
    ]
    wide = template[metadata_columns].copy()
    for variant_frame in prediction_frames:
        variant_id = variant_frame["variant_id"].iloc[0]
        wide = wide.merge(
            variant_frame[["row_complex_id", "model_predicted_pk"]].rename(
                columns={"model_predicted_pk": f"{variant_id}_pred_pk"}
            ),
            on="row_complex_id",
            how="left",
        )
    wide.to_csv(predictions_wide_tsv, sep="\t", index=False)

    comparison_metrics: list[dict[str, Any]] = []
    comparison_metrics.extend(
        metrics_for_prediction_frame(template, "paper_predicted_pk", "paper_appendix")
    )
    comparison_metrics.extend(
        metrics_for_prediction_frame(
            template, "model_predicted_pk", "all_features_from_readme_lorentz"
        )
    )
    comparison_metrics.extend(
        metrics_for_prediction_frame(
            default_predictions, "model_predicted_pk", "all_features_from_default_lorentz"
        )
    )
    for variant_frame in prediction_frames:
        variant_id = variant_frame["variant_id"].iloc[0]
        comparison_metrics.extend(
            metrics_for_prediction_frame(variant_frame, "model_predicted_pk", variant_id)
        )

    metrics_frame = pd.DataFrame(comparison_metrics)
    metrics_frame.to_csv(metrics_tsv, sep="\t", index=False)

    feature_suffix_counts = {
        "x": sum(1 for feature in feature_names if feature.endswith("_x")),
        "y": sum(1 for feature in feature_names if feature.endswith("_y")),
        "unsuffixed": sum(
            1
            for feature in feature_names
            if not feature.endswith("_x") and not feature.endswith("_y")
        ),
    }
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "inputs": {
            "model": display_path(args.model),
            "readme_matrix": display_path(args.readme_matrix),
            "default_matrix": display_path(args.default_matrix),
            "readme_predictions": display_path(args.readme_predictions),
            "default_predictions": display_path(args.default_predictions),
        },
        "feature_sources": {
            README_SOURCE: {
                "kernel_type": "lorentz_kernel",
                "tau": 1.5,
                "power": 2.5,
                "cutoff": 6.0,
            },
            DEFAULT_SOURCE: {
                "kernel_type": "lorentz_kernel",
                "tau": 0.5,
                "power": 5.0,
                "cutoff": 6.0,
            },
        },
        "feature_suffix_counts": feature_suffix_counts,
        "variants": VARIANTS,
        "comparison_metrics": comparison_metrics,
        "outputs": {
            "predictions_long_tsv": display_path(predictions_long_tsv),
            "predictions_wide_tsv": display_path(predictions_wide_tsv),
            "metrics_tsv": display_path(metrics_tsv),
            "summary_json": display_path(args.summary_json),
            "summary_md": display_path(args.summary_md),
        },
    }
    write_json(args.summary_json, summary)
    write_markdown(args.summary_md, summary)

    unique = metrics_frame[metrics_frame["scope"].eq("unique_pdb_all")].copy()
    print(unique[["label", "rows", "unique_pdb_ids", "pearson_vs_actual", "mae_vs_actual", "rmse_vs_actual"]].to_string(index=False))
    print(f"Wrote {display_path(predictions_long_tsv)}")
    print(f"Wrote {display_path(args.summary_md)}")


if __name__ == "__main__":
    main()
