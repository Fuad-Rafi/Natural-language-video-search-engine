"""FAISS index and sidecar persistence helpers.

Every retrieval layer gets its own flat inner-product index and JSON sidecar,
so layers can be rebuilt or ablated independently.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class StoreError(RuntimeError):
    """Raised when an index or sidecar is missing or inconsistent."""


def relative_path(path: str | Path, root: str | Path) -> str:
    """Return a POSIX path relative to ``root`` when possible."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Return L2-normalized float32 vectors, safe for zero rows."""
    array = np.ascontiguousarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {array.shape}")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def build_flat_ip(vectors: np.ndarray) -> faiss.Index:
    """Build an exact inner-product index over normalized vectors."""
    normalized = normalize(vectors)
    if normalized.shape[0] == 0:
        raise ValueError("cannot build an index with zero vectors")
    index = faiss.IndexFlatIP(normalized.shape[1])
    index.add(normalized)
    return index


def save_faiss(index: faiss.Index, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(destination))
    return destination


def load_faiss(path: str | Path) -> faiss.Index:
    source = Path(path)
    if not source.is_file():
        raise StoreError(f"index not found: {source}")
    return faiss.read_index(str(source))


def write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def read_json(path: str | Path) -> Any:
    source = Path(path)
    if not source.is_file():
        raise StoreError(f"sidecar not found: {source}")
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StoreError(f"invalid JSON in {source}: {exc}") from exc


def search(index: faiss.Index, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(scores, positions)`` for one query vector."""
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if index.ntotal == 0:
        raise StoreError("index is empty")
    vector = normalize(np.asarray(query).reshape(1, -1))
    if vector.shape[1] != index.d:
        raise StoreError(
            f"query dimension {vector.shape[1]} does not match index dimension {index.d}"
        )
    k = min(int(top_k), int(index.ntotal))
    scores, positions = index.search(vector, k)
    return scores[0], positions[0]
