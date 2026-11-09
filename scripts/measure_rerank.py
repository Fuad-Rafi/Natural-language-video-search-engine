#!/usr/bin/env python
"""Measure Qwen2.5-VL top-3 rerank latency on saved search results.

The reranker is optional and intentionally not part of the headline metrics.
This script loads a results JSON, reranks the top 3 segments for a handful of
queries, and reports p50/p95 latency plus the yes-rate.

Example:
    python scripts/measure_rerank.py --limit 5
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ConfigError, load_config  # noqa: E402
from src.search.rerank import QwenReranker  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

_LOG = get_logger("measure_rerank")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure VLM rerank latency.")
    parser.add_argument(
        "--results", default="benchmarks/results/all.json", help="results JSON"
    )
    parser.add_argument("--limit", type=int, default=5, help="queries to rerank")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--out", default="benchmarks/results/rerank_latency.json"
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(round(fraction * (len(ordered) - 1)))),
    )
    return ordered[index]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(level=getattr(logging, args.log_level), console=True)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    results_path = config.resolve(args.results)
    if not results_path.is_file():
        print(f"error: results not found: {results_path}", file=sys.stderr)
        print("Run: python scripts/evaluate.py --mode ALL", file=sys.stderr)
        return 2
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    queries = [
        item
        for item in payload.get("per_query", [])
        if item.get("top3") and item.get("top3")[0].get("path")
    ][: max(1, args.limit)]
    if not queries:
        print("error: no queries with top-3 thumbnails", file=sys.stderr)
        return 1

    reranker = QwenReranker(config)
    # Warm up CUDA kernels so the first measured query is not a 10 s outlier.
    first_path = queries[0]["top3"][0].get("path")
    if first_path:
        reranker.yes_no(queries[0]["query"], first_path)
    latencies: list[float] = []
    records = []
    yes = 0
    total = 0
    started = time.perf_counter()
    for item in queries:
        ordered, elapsed_ms, flags = reranker.rerank(
            item["query"], item["top3"], top_n=args.top_n
        )
        latencies.append(elapsed_ms)
        yes += sum(1 for flag in flags if flag is True)
        total += sum(1 for flag in flags if flag is not None)
        records.append(
            {
                "query_id": item["query_id"],
                "query": item["query"],
                "flags": flags,
                "latency_ms": round(elapsed_ms, 1),
                "top1_before": item["top3"][0].get("timestamp"),
                "top1_after": ordered[0].get("timestamp") if ordered else None,
            }
        )
        _LOG.info(
            "%s %.0f ms flags=%s", item["query_id"], elapsed_ms, flags
        )

    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": config.get("captions.qwen_model"),
        "queries": len(queries),
        "top_n": args.top_n,
        "latency_p50_ms": round(statistics.median(latencies), 1),
        "latency_mean_ms": round(statistics.fmean(latencies), 1),
        "latency_p95_ms": round(_percentile(latencies, 0.95), 1),
        "yes_rate": round(yes / total, 3) if total else 0.0,
        "elapsed_s": round(time.perf_counter() - started, 1),
        "items": records,
    }
    out_path = config.resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"  Reranker        : {summary['model']}")
    print(f"  Queries         : {summary['queries']}")
    print(f"  Top-N per query : {summary['top_n']}")
    print(f"  Latency p50     : {summary['latency_p50_ms']} ms")
    print(f"  Latency mean    : {summary['latency_mean_ms']} ms")
    print(f"  Latency p95     : {summary['latency_p95_ms']} ms")
    print(f"  Yes rate        : {summary['yes_rate']:.3f}")
    print(f"  Results         : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
