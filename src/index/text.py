"""Text embeddings and BM25 helpers for text-based layers (speech, OCR, captions).

CLIP/SigLIP text encoders are aligned to images, not to arbitrary text, so the
text layers use a sentence embedding model plus BM25 over the same chunks.

Dense and sparse rankings are combined with reciprocal rank fusion (RRF) inside
a layer. That keeps the speech layer consistent with the cross-layer fusion in
``src/search/fusion.py``: ranks are comparable, raw scores are not.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Iterable, Sequence

import numpy as np

from src.index.frames import resolve_device
from src.utils.logging import get_logger

_LOG = get_logger("index.text")

DEFAULT_TEXT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_TOKEN = re.compile(r"[a-z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z][a-z])")
_NORMALIZE = re.compile(r"[^a-z0-9]+")


def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation for similarity comparisons."""
    return _NORMALIZE.sub(" ", str(text or "").lower()).strip()


def text_similarity(first: str, second: str) -> float:
    """Return a 0–1 similarity between two short texts."""
    from rapidfuzz import fuzz

    left = normalize_text(first)
    right = normalize_text(second)
    if not left or not right:
        return 0.0
    return float(fuzz.ratio(left, right)) / 100.0


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; punctuation is dropped.

    CamelCase is split so OCR strings such as ``CurrentVersion`` match a query
    for ``current version``. Hex codes like ``0xC0000142`` stay intact.
    """
    value = _CAMEL.sub(" ", str(text or ""))
    return _TOKEN.findall(value.lower())


@lru_cache(maxsize=2)
def load_text_encoder(model_name: str = DEFAULT_TEXT_MODEL, device: str = "auto") -> Any:
    """Load (and cache) a SentenceTransformer for text encoding."""
    from sentence_transformers import SentenceTransformer

    resolved = resolve_device(device)
    _LOG.info("loading text encoder %s on %s", model_name, resolved)
    return SentenceTransformer(model_name, device=resolved)


def encode_texts(
    texts: Iterable[str],
    model: Any,
    *,
    batch_size: int = 32,
    query: bool = False,
    query_prefix: bool = False,
) -> np.ndarray:
    """Return normalized float32 embeddings in input order.

    ``query`` marks a retrieval query so the optional BGE instruction prefix is
    applied. Passages are encoded without the prefix.
    """
    values = [str(text) for text in texts]
    if not values:
        raise ValueError("no texts to encode")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    prefix = DEFAULT_QUERY_PREFIX if (query and query_prefix) else ""
    inputs = [f"{prefix}{value}" for value in values] if prefix else values
    vectors = model.encode(
        inputs,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.ascontiguousarray(vectors, dtype=np.float32)


def build_bm25(corpus: Iterable[str]) -> Any:
    """Build a BM25 index over tokenized documents.

    ``BM25Plus`` is used instead of ``BM25Okapi`` because its IDF stays
    positive on the small corpora produced by short videos.
    """
    from rank_bm25 import BM25Plus

    documents = [tokenize(document) for document in corpus]
    return BM25Plus(documents)


def bm25_rank(
    query: str,
    corpus: Sequence[str],
    *,
    limit: int | None = None,
) -> list[tuple[int, float]]:
    """Return ``(position, score)`` pairs, best first, ignoring zero scores."""
    if not corpus:
        return []
    scores = build_bm25(corpus).get_scores(tokenize(query))
    ranked = [
        (int(position), float(score))
        for position, score in enumerate(scores)
        if float(score) > 0.0
    ]
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    if limit is not None:
        ranked = ranked[: max(0, int(limit))]
    return ranked


def reciprocal_rank_scores(
    rankings: Sequence[Sequence[int]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> dict[int, float]:
    """Fuse several ranked lists of integer positions with RRF."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")

    scores: dict[int, float] = {}
    for weight, ranking in zip(weights, rankings):
        for rank, position in enumerate(ranking, start=1):
            scores[int(position)] = scores.get(int(position), 0.0) + float(weight) / (
                k + rank
            )
    return scores
