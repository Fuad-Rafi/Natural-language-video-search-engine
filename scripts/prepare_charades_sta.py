#!/usr/bin/env python
"""Prepare a reproducible Charades-STA test subset.

The official Charades-STA videos are inside a 16 GB zip. This script reads the
zip's central directory over HTTP and downloads only the selected videos with
range requests, then writes a query CSV for ``scripts/evaluate.py``.

Example:
    python scripts/prepare_charades_sta.py --download --max-videos 50
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CHARADES_ZIP = "https://ai2-public-datasets.s3.amazonaws.com/charades/Charades_v1_480.zip"
ANNOTATIONS_URL = (
    "https://huggingface.co/datasets/yeliudev/VideoMind-Dataset/"
    "resolve/main/charades_sta/charades_sta_test.txt"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a Charades-STA subset.")
    parser.add_argument("--root", default="data/public/charades_sta")
    parser.add_argument("--max-videos", type=int, default=50)
    parser.add_argument("--min-queries", type=int, default=2)
    parser.add_argument(
        "--out-csv", default="benchmarks/public/charades_sta_subset.csv"
    )
    parser.add_argument(
        "--manifest", default="benchmarks/public/charades_sta_subset.json"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="download selected videos from the official zip via range requests",
    )
    return parser


def read_annotations(path: Path) -> list[tuple[str, float, float, str]]:
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading annotations: {ANNOTATIONS_URL}")
        path.write_bytes(urllib.request.urlopen(ANNOTATIONS_URL, timeout=120).read())
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "##" not in line:
            continue
        left, text = line.split("##", 1)
        parts = left.split()
        if len(parts) != 3:
            continue
        rows.append((parts[0], float(parts[1]), float(parts[2]), text.strip()))
    return rows


def download_videos(video_ids: list[str], out_dir: Path) -> None:
    from remotezip import RemoteZip

    out_dir.mkdir(parents=True, exist_ok=True)
    with RemoteZip(CHARADES_ZIP, timeout=120) as archive:
        for index, video_id in enumerate(video_ids, start=1):
            target = out_dir / f"{video_id}.mp4"
            if target.is_file() and target.stat().st_size > 0:
                continue
            target.write_bytes(
                archive.read(f"Charades_v1_480/{video_id}.mp4")
            )
            print(f"[{index}/{len(video_ids)}] {video_id} {target.stat().st_size}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root)
    rows = read_annotations(root / "charades_sta_test.txt")
    counts = Counter(row[0] for row in rows)
    selected = [
        video_id
        for video_id, count in counts.most_common()
        if count >= args.min_queries
    ][: args.max_videos]
    selected_set = set(selected)

    if args.download:
        download_videos(selected, root / "videos")

    subset = [row for row in rows if row[0] in selected_set]
    subset.sort(key=lambda row: (row[0], row[1]))
    csv_path = Path(args.out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "video_id", "query", "type", "start", "end"])
        for index, (video_id, start, end, text) in enumerate(subset, start=1):
            writer.writerow(
                [f"charades_{index:04d}", video_id, text, "action", start, end]
            )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": CHARADES_ZIP,
        "annotations": ANNOTATIONS_URL,
        "split": "test",
        "videos": len(selected),
        "queries": len(subset),
        "min_queries_per_video": args.min_queries,
        "video_ids": selected,
        "query_counts": {video_id: counts[video_id] for video_id in selected},
        "csv_sha256": sha256_file(csv_path),
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print()
    print(f"  Videos      : {len(selected)}")
    print(f"  Queries     : {len(subset)}")
    print(f"  CSV         : {csv_path}")
    print(f"  Manifest    : {manifest_path}")
    print(f"  CSV SHA-256 : {manifest['csv_sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
