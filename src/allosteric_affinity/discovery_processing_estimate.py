#!/usr/bin/env python3
"""Estimate storage and runtime for ASD resolved complex processing.

This script summarizes all RCSB-resolved ASD allosteric complex structures by
experimental method and, when requested, measures the compressed/decompressed
RCSB mmCIF payload by streaming `.cif.gz` files without storing the structures.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import io
import json
import pathlib
import statistics
import sys
import tarfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ASD_SITE_MEMBER = "ASD_Release_202309_AS.txt"
DEFAULT_ARCHIVE_PATH = REPO_ROOT / pathlib.Path("data/raw/asd/archives/ASD_Release_202309_AS.tar.gz")
DEFAULT_METHODS_PATH = REPO_ROOT / pathlib.Path("data/interim/asd/rcsb_pdb_methods.tsv")
DEFAULT_SIZE_CACHE_PATH = REPO_ROOT / pathlib.Path("data/interim/asd/rcsb_cif_size_cache.tsv")
DEFAULT_SUMMARY_PATH = REPO_ROOT / pathlib.Path("outputs/discovery/asd_resolved_complex_processing_estimate.json")
RCSB_DOWNLOAD_ROOT = "https://files.rcsb.org/download"
FEATURE_COUNT = 11_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate disk/network and processing scale for ASD resolved structures."
    )
    parser.add_argument("--archive", type=pathlib.Path, default=DEFAULT_ARCHIVE_PATH)
    parser.add_argument("--methods", type=pathlib.Path, default=DEFAULT_METHODS_PATH)
    parser.add_argument("--size-cache", type=pathlib.Path, default=DEFAULT_SIZE_CACHE_PATH)
    parser.add_argument("--summary-out", type=pathlib.Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument(
        "--measure-file-sizes",
        action="store_true",
        help="Stream missing RCSB .cif.gz files to measure compressed and decompressed sizes.",
    )
    parser.add_argument(
        "--refresh-size-cache",
        action="store_true",
        help="Re-measure all RCSB .cif.gz sizes even when a cache exists.",
    )
    parser.add_argument("--workers", type=int, default=12, help="Concurrent RCSB downloads. Default: 12")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-file HTTP timeout in seconds.")
    return parser.parse_args()


def load_asd_site_rows(archive_path: pathlib.Path) -> list[dict[str, str]]:
    if not archive_path.exists():
        raise FileNotFoundError(f"Missing ASD AS archive: {archive_path}")

    with tarfile.open(archive_path, "r:gz") as archive:
        member = archive.getmember(ASD_SITE_MEMBER)
        extracted = archive.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"Could not extract {ASD_SITE_MEMBER}")

        with extracted:
            text_stream = io.TextIOWrapper(extracted, encoding="utf-8-sig", newline="")
            return [dict(row) for row in csv.DictReader(text_stream, delimiter="\t")]


def load_methods(path: pathlib.Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing RCSB methods cache: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["pdb_id"].strip().upper(): dict(row) for row in csv.DictReader(handle, delimiter="\t")}


def load_size_cache(path: pathlib.Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["pdb_id"].strip().upper(): dict(row) for row in csv.DictReader(handle, delimiter="\t")}


def write_size_cache(path: pathlib.Path, rows_by_pdb: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pdb_id",
        "methods",
        "compressed_bytes",
        "decompressed_bytes",
        "elapsed_seconds",
        "status",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for pdb_id in sorted(rows_by_pdb):
            row = rows_by_pdb[pdb_id]
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def method_or_unresolved(methods_by_pdb: dict[str, dict[str, str]], pdb_id: str) -> str:
    value = methods_by_pdb.get(pdb_id, {}).get("methods", "").strip()
    return value if value else "UNRESOLVED"


def count_by(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def stream_cif_size(pdb_id: str, methods: str, timeout: float) -> dict[str, str]:
    url = f"{RCSB_DOWNLOAD_ROOT}/{pdb_id}.cif.gz"
    start = time.perf_counter()
    request = urllib.request.Request(url, headers={"User-Agent": "allosteric-binding-affinity/0.1"})
    compressed_bytes = 0
    decompressed_bytes = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
            compressed_bytes = len(payload)
            decompressed_bytes = len(gzip.decompress(payload))
        status = "ok"
        error = ""
    except (OSError, urllib.error.URLError, gzip.BadGzipFile) as exc:
        status = "error"
        error = str(exc).replace("\t", " ").replace("\n", " ")
    elapsed = time.perf_counter() - start
    return {
        "pdb_id": pdb_id,
        "methods": methods,
        "compressed_bytes": str(compressed_bytes),
        "decompressed_bytes": str(decompressed_bytes),
        "elapsed_seconds": f"{elapsed:.3f}",
        "status": status,
        "error": error,
    }


def measure_missing_sizes(
    resolved_pdb_ids: list[str],
    methods_by_pdb: dict[str, dict[str, str]],
    cache: dict[str, dict[str, str]],
    workers: int,
    timeout: float,
    refresh: bool,
) -> dict[str, dict[str, str]]:
    if refresh:
        cache = {}

    missing = [
        pdb_id
        for pdb_id in resolved_pdb_ids
        if pdb_id not in cache or cache[pdb_id].get("status") != "ok"
    ]
    if not missing:
        return cache

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(stream_cif_size, pdb_id, methods_by_pdb[pdb_id]["methods"], timeout): pdb_id
            for pdb_id in missing
        }
        completed = 0
        for future in as_completed(futures):
            row = future.result()
            cache[row["pdb_id"]] = row
            completed += 1
            if completed % 100 == 0 or completed == len(missing):
                print(f"Measured {completed}/{len(missing)} missing RCSB files...", file=sys.stderr)
    return cache


def int_field(row: dict[str, str], field: str) -> int:
    value = row.get(field, "")
    if not value:
        return 0
    return int(float(value))


def float_field(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if not value:
        return 0.0
    return float(value)


def bytes_to_gib(value: int) -> float:
    return value / (1024**3)


def seconds_for_bytes(total_bytes: int, mbps: float) -> float:
    bytes_per_second = (mbps * 1_000_000) / 8
    return total_bytes / bytes_per_second


def summarize_sizes(cache: dict[str, dict[str, str]], resolved_pdb_ids: list[str]) -> dict[str, object]:
    ok_rows = [cache[pdb_id] for pdb_id in resolved_pdb_ids if cache.get(pdb_id, {}).get("status") == "ok"]
    failed = [pdb_id for pdb_id in resolved_pdb_ids if cache.get(pdb_id, {}).get("status") != "ok"]
    compressed_values = [int_field(row, "compressed_bytes") for row in ok_rows]
    decompressed_values = [int_field(row, "decompressed_bytes") for row in ok_rows]
    elapsed_values = [float_field(row, "elapsed_seconds") for row in ok_rows]

    if not ok_rows:
        return {"measured_files": 0, "failed_files": len(failed), "failed_pdb_ids": failed}

    total_compressed = sum(compressed_values)
    total_decompressed = sum(decompressed_values)
    feature_rows_unique = len(resolved_pdb_ids)
    feature_matrix_float32 = feature_rows_unique * FEATURE_COUNT * 4
    feature_matrix_float64 = feature_rows_unique * FEATURE_COUNT * 8

    return {
        "measured_files": len(ok_rows),
        "failed_files": len(failed),
        "failed_pdb_ids": failed,
        "compressed_cif_total_bytes": total_compressed,
        "compressed_cif_total_gib": round(bytes_to_gib(total_compressed), 3),
        "decompressed_cif_total_bytes": total_decompressed,
        "decompressed_cif_total_gib": round(bytes_to_gib(total_decompressed), 3),
        "compressed_cif_median_bytes": int(statistics.median(compressed_values)),
        "compressed_cif_max_bytes": max(compressed_values),
        "decompressed_cif_median_bytes": int(statistics.median(decompressed_values)),
        "decompressed_cif_max_bytes": max(decompressed_values),
        "measurement_elapsed_seconds_sum": round(sum(elapsed_values), 1),
        "download_time_estimates": {
            "50_mbps_seconds": round(seconds_for_bytes(total_compressed, 50), 1),
            "100_mbps_seconds": round(seconds_for_bytes(total_compressed, 100), 1),
            "500_mbps_seconds": round(seconds_for_bytes(total_compressed, 500), 1),
        },
        "feature_matrix_estimate_unique_structures": {
            "features_per_complex": FEATURE_COUNT,
            "float32_bytes": feature_matrix_float32,
            "float32_gib": round(bytes_to_gib(feature_matrix_float32), 3),
            "float64_bytes": feature_matrix_float64,
            "float64_gib": round(bytes_to_gib(feature_matrix_float64), 3),
        },
    }


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    rows = load_asd_site_rows(args.archive)
    methods_by_pdb = load_methods(args.methods)
    unique_pdb_ids = sorted({row["allosteric_pdb"].strip().upper() for row in rows})
    resolved_pdb_ids = sorted(
        pdb_id for pdb_id in unique_pdb_ids if methods_by_pdb.get(pdb_id, {}).get("methods", "").strip()
    )
    row_methods = [method_or_unresolved(methods_by_pdb, row["allosteric_pdb"].strip().upper()) for row in rows]
    unique_methods = [method_or_unresolved(methods_by_pdb, pdb_id) for pdb_id in unique_pdb_ids]

    size_cache = load_size_cache(args.size_cache)
    if args.measure_file_sizes:
        size_cache = measure_missing_sizes(
            resolved_pdb_ids,
            methods_by_pdb,
            size_cache,
            workers=args.workers,
            timeout=args.timeout,
            refresh=args.refresh_size_cache,
        )
        write_size_cache(args.size_cache, size_cache)

    resolved_row_count = sum(1 for method in row_methods if method != "UNRESOLVED")
    feature_matrix_float32_rows = resolved_row_count * FEATURE_COUNT * 4
    feature_matrix_float64_rows = resolved_row_count * FEATURE_COUNT * 8
    summary = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "asd_archive": str(args.archive),
            "rcsb_methods_cache": str(args.methods),
            "rcsb_cif_size_cache": str(args.size_cache),
        },
        "sources": {
            "rcsb_download_pattern": f"{RCSB_DOWNLOAD_ROOT}/<PDB_ID>.cif.gz",
        },
        "counts": {
            "asd_site_rows": len(rows),
            "unique_allosteric_pdb_ids": len(unique_pdb_ids),
            "resolved_site_rows": resolved_row_count,
            "resolved_unique_pdb_ids": len(resolved_pdb_ids),
            "unresolved_site_rows": len(rows) - resolved_row_count,
            "unresolved_unique_pdb_ids": len(unique_pdb_ids) - len(resolved_pdb_ids),
        },
        "row_counts_by_method": count_by(row_methods),
        "unique_pdb_counts_by_method": count_by(unique_methods),
        "size_summary": summarize_sizes(size_cache, resolved_pdb_ids),
        "feature_matrix_estimate_resolved_rows": {
            "rows": resolved_row_count,
            "features_per_complex": FEATURE_COUNT,
            "float32_bytes": feature_matrix_float32_rows,
            "float32_gib": round(bytes_to_gib(feature_matrix_float32_rows), 3),
            "float64_bytes": feature_matrix_float64_rows,
            "float64_gib": round(bytes_to_gib(feature_matrix_float64_rows), 3),
        },
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "Resolved ASD complexes: "
        f"{summary['counts']['resolved_unique_pdb_ids']} unique PDB IDs, "
        f"{summary['counts']['resolved_site_rows']} ASD rows."
    )
    size_summary = summary["size_summary"]
    if size_summary.get("measured_files"):
        print(
            "Measured RCSB mmCIF payload: "
            f"{size_summary['compressed_cif_total_gib']} GiB compressed, "
            f"{size_summary['decompressed_cif_total_gib']} GiB decompressed."
        )
    else:
        print("RCSB mmCIF sizes not measured. Re-run with --measure-file-sizes.")
    print(f"Wrote summary: {args.summary_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
