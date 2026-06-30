#!/usr/bin/env python3
"""Download ASD allosteric-site data and identify X-ray complex rows.

The ASD raw archives are stored locally and intentionally ignored by git.
This script derives a local TSV of ASD allosteric complex rows whose
`allosteric_pdb` entry is confirmed by RCSB to have an X-RAY DIFFRACTION
experimental method.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import pathlib
import sys
import tarfile
import time
import urllib.error
import urllib.request


ASD_ARCHIVE_ROOT = "https://mdl.shsmu.edu.cn/ASD2023Common/static_file/archive_2023"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ASD_SITE_ARCHIVE = "ASD_Release_202309_AS.tar.gz"
ASD_SITE_MEMBER = "ASD_Release_202309_AS.txt"
RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"
X_RAY_METHOD = "X-RAY DIFFRACTION"

DEFAULT_ARCHIVE_PATH = REPO_ROOT / pathlib.Path("data/raw/asd/archives") / ASD_SITE_ARCHIVE
DEFAULT_METHODS_PATH = REPO_ROOT / pathlib.Path("data/interim/asd/rcsb_pdb_methods.tsv")
DEFAULT_XRAY_PATH = REPO_ROOT / pathlib.Path("data/interim/asd/asd_xray_complexes.tsv")
DEFAULT_SUMMARY_PATH = REPO_ROOT / pathlib.Path("outputs/discovery/asd_xray_complexes_summary.json")

RCSB_QUERY = """
query EntryMethods($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    exptl {
      method
    }
    rcsb_entry_info {
      resolution_combined
    }
    struct {
      title
    }
  }
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the local ASD subset with RCSB-confirmed X-ray complex structures."
    )
    parser.add_argument(
        "--archive",
        type=pathlib.Path,
        default=DEFAULT_ARCHIVE_PATH,
        help=f"Local ASD AS archive path. Default: {DEFAULT_ARCHIVE_PATH}",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the ASD AS archive when it is missing.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the ASD AS archive even if it already exists.",
    )
    parser.add_argument(
        "--methods-out",
        type=pathlib.Path,
        default=DEFAULT_METHODS_PATH,
        help=f"RCSB PDB-method cache TSV. Default: {DEFAULT_METHODS_PATH}",
    )
    parser.add_argument(
        "--xray-out",
        type=pathlib.Path,
        default=DEFAULT_XRAY_PATH,
        help=f"Output TSV for ASD rows with X-ray PDBs. Default: {DEFAULT_XRAY_PATH}",
    )
    parser.add_argument(
        "--summary-out",
        type=pathlib.Path,
        default=DEFAULT_SUMMARY_PATH,
        help=f"Summary JSON path. Default: {DEFAULT_SUMMARY_PATH}",
    )
    parser.add_argument(
        "--refresh-rcsb",
        action="store_true",
        help="Ignore an existing methods cache and query RCSB again.",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Allow RCSB queries for missing or refreshed method records.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of PDB IDs per RCSB GraphQL request. Default: 100",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.05,
        help="Delay in seconds between RCSB requests. Default: 0.05",
    )
    return parser.parse_args()


def archive_url() -> str:
    return f"{ASD_ARCHIVE_ROOT}/{ASD_SITE_ARCHIVE}"


def download_archive(destination: pathlib.Path, force: bool) -> None:
    if destination.exists() and not force:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        archive_url(),
        headers={"User-Agent": "allosteric-binding-affinity/0.1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())


def load_asd_site_rows(archive_path: pathlib.Path) -> list[dict[str, str]]:
    if not archive_path.exists():
        raise FileNotFoundError(
            f"{archive_path} does not exist. Run with --download or download {archive_url()}."
        )

    with tarfile.open(archive_path, "r:gz") as archive:
        try:
            member = archive.getmember(ASD_SITE_MEMBER)
        except KeyError as exc:
            member_names = ", ".join(archive.getnames())
            raise RuntimeError(f"{ASD_SITE_MEMBER} not found in archive. Members: {member_names}") from exc

        extracted = archive.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"Could not extract {ASD_SITE_MEMBER} from {archive_path}")

        with extracted:
            text_stream = io.TextIOWrapper(extracted, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(text_stream, delimiter="\t")
            return [dict(row) for row in reader]


def unique_pdb_ids(rows: list[dict[str, str]]) -> list[str]:
    ids = sorted({row["allosteric_pdb"].strip().upper() for row in rows if row.get("allosteric_pdb", "").strip()})
    invalid = [pdb_id for pdb_id in ids if len(pdb_id) != 4 or not pdb_id.isalnum()]
    if invalid:
        raise ValueError(f"Unexpected non-canonical PDB IDs in ASD AS table: {invalid[:20]}")
    return ids


def chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def graphql_post(query: str, variables: dict[str, object]) -> dict[str, object]:
    payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    request = urllib.request.Request(
        RCSB_GRAPHQL_URL,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "allosteric-binding-affinity/0.1",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def query_rcsb_methods(pdb_ids: list[str], batch_size: int, request_delay: float) -> dict[str, dict[str, str]]:
    methods_by_pdb: dict[str, dict[str, str]] = {}
    for batch_number, batch in enumerate(chunked(pdb_ids, batch_size), start=1):
        try:
            payload = graphql_post(RCSB_QUERY, {"ids": batch})
        except urllib.error.URLError as exc:
            raise RuntimeError(f"RCSB GraphQL request failed for batch {batch_number}: {exc}") from exc

        if payload.get("errors"):
            raise RuntimeError(f"RCSB GraphQL returned errors for batch {batch_number}: {payload['errors']}")

        entries = (payload.get("data") or {}).get("entries") or []
        for entry in entries:
            if entry is None:
                continue

            pdb_id = str(entry.get("rcsb_id", "")).upper()
            exptl_methods = [item.get("method", "") for item in entry.get("exptl") or [] if item]
            resolutions = (entry.get("rcsb_entry_info") or {}).get("resolution_combined") or []
            title = (entry.get("struct") or {}).get("title") or ""
            methods_by_pdb[pdb_id] = {
                "pdb_id": pdb_id,
                "methods": ";".join(exptl_methods),
                "resolution_combined": ";".join(str(value) for value in resolutions),
                "title": title,
            }

        if request_delay > 0:
            time.sleep(request_delay)

    missing = sorted(set(pdb_ids) - set(methods_by_pdb))
    for pdb_id in missing:
        methods_by_pdb[pdb_id] = {
            "pdb_id": pdb_id,
            "methods": "",
            "resolution_combined": "",
            "title": "",
        }
    return methods_by_pdb


def read_methods_cache(path: pathlib.Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["pdb_id"]: dict(row) for row in csv.DictReader(handle, delimiter="\t")}


def write_methods_cache(path: pathlib.Path, methods_by_pdb: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["pdb_id", "methods", "resolution_combined", "title"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for pdb_id in sorted(methods_by_pdb):
            writer.writerow({field: methods_by_pdb[pdb_id].get(field, "") for field in fieldnames})


def write_xray_subset(
    path: pathlib.Path,
    rows: list[dict[str, str]],
    methods_by_pdb: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    xray_rows: list[dict[str, str]] = []
    for row in rows:
        pdb_id = row["allosteric_pdb"].strip().upper()
        metadata = methods_by_pdb.get(pdb_id, {})
        method_tokens = [token.strip().upper() for token in metadata.get("methods", "").split(";")]
        if X_RAY_METHOD not in method_tokens:
            continue
        out_row = dict(row)
        out_row["allosteric_pdb"] = pdb_id
        out_row["rcsb_methods"] = metadata.get("methods", "")
        out_row["rcsb_resolution_combined"] = metadata.get("resolution_combined", "")
        out_row["rcsb_title"] = metadata.get("title", "")
        xray_rows.append(out_row)

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) + ["rcsb_methods", "rcsb_resolution_combined", "rcsb_title"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(xray_rows)
    return xray_rows


def method_distribution(methods_by_pdb: dict[str, dict[str, str]]) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for metadata in methods_by_pdb.values():
        methods = metadata.get("methods", "") or "UNRESOLVED"
        distribution[methods] = distribution.get(methods, 0) + 1
    return dict(sorted(distribution.items(), key=lambda item: (-item[1], item[0])))


def write_summary(
    path: pathlib.Path,
    rows: list[dict[str, str]],
    xray_rows: list[dict[str, str]],
    pdb_ids: list[str],
    methods_by_pdb: dict[str, dict[str, str]],
    args: argparse.Namespace,
) -> dict[str, object]:
    xray_pdb_ids = sorted({row["allosteric_pdb"].strip().upper() for row in xray_rows})
    summary = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "sources": {
            "asd_archive_url": archive_url(),
            "asd_download_page": "https://mdl.shsmu.edu.cn/ASD/module/download/download.jsp?tabIndex=1",
            "rcsb_graphql_url": RCSB_GRAPHQL_URL,
        },
        "inputs": {
            "asd_archive": str(args.archive),
            "asd_site_member": ASD_SITE_MEMBER,
            "rcsb_methods_cache": str(args.methods_out),
        },
        "outputs": {
            "xray_subset_tsv": str(args.xray_out),
            "summary_json": str(path),
        },
        "counts": {
            "asd_site_rows": len(rows),
            "unique_asd_target_ids": len({row["target_id"] for row in rows}),
            "unique_allosteric_pdb_ids": len(pdb_ids),
            "rcsb_resolved_pdb_ids": sum(1 for item in methods_by_pdb.values() if item.get("methods")),
            "xray_site_rows": len(xray_rows),
            "xray_unique_asd_target_ids": len({row["target_id"] for row in xray_rows}),
            "xray_unique_pdb_ids": len(xray_pdb_ids),
        },
        "rcsb_method_distribution_by_unique_pdb": method_distribution(methods_by_pdb),
        "filter_rule": f"Keep ASD AS rows when RCSB exptl.method contains {X_RAY_METHOD!r}.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.refresh_rcsb and not args.online:
        raise ValueError("--refresh-rcsb requires --online")

    if args.download or args.force_download:
        download_archive(args.archive, force=args.force_download)

    rows = load_asd_site_rows(args.archive)
    if not rows:
        raise RuntimeError("ASD AS archive contained no rows")

    pdb_ids = unique_pdb_ids(rows)
    if args.methods_out.exists() and not args.refresh_rcsb:
        methods_by_pdb = read_methods_cache(args.methods_out)
        missing = sorted(set(pdb_ids) - set(methods_by_pdb))
        if missing:
            if not args.online:
                raise RuntimeError(
                    f"RCSB methods cache lacks {len(missing)} PDB IDs; use --online to query them."
                )
            queried = query_rcsb_methods(missing, args.batch_size, args.request_delay)
            methods_by_pdb.update(queried)
            write_methods_cache(args.methods_out, methods_by_pdb)
    else:
        if not args.online:
            raise FileNotFoundError(
                f"RCSB methods cache is unavailable: {args.methods_out}. Use --online to build it."
            )
        methods_by_pdb = query_rcsb_methods(pdb_ids, args.batch_size, args.request_delay)
        write_methods_cache(args.methods_out, methods_by_pdb)

    xray_rows = write_xray_subset(args.xray_out, rows, methods_by_pdb)
    summary = write_summary(args.summary_out, rows, xray_rows, pdb_ids, methods_by_pdb, args)

    counts = summary["counts"]
    print(
        "ASD X-ray discovery complete: "
        f"{counts['xray_site_rows']} of {counts['asd_site_rows']} ASD site rows; "
        f"{counts['xray_unique_pdb_ids']} of {counts['unique_allosteric_pdb_ids']} unique PDB IDs."
    )
    print(f"Wrote local subset: {args.xray_out}")
    print(f"Wrote summary: {args.summary_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
