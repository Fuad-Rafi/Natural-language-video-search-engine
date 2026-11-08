"""Layer B: object detection, tracking, crops, and a second FAISS index.

YOLO11 detects person/vehicle/bag classes on the sampled frames; ByteTrack
assigns track ids. Each track keeps 1-3 representative crops, embedded with
the same SigLIP image encoder used by Layer A so one text encoder serves both
layers. Fusion (P4) will combine them with RRF.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np

from src.config import Config
from src.index import store
from src.index.frames import (
    DEFAULT_MODEL,
    embed_images,
    load_siglip,
    resolve_device,
)
from src.utils.logging import StageTimer, get_logger

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

_LOG = get_logger("index.objects")

DEFAULT_DETECTOR = "yolo11n.pt"
DEFAULT_CLASSES = [0, 1, 2, 3, 5, 7, 24, 26, 28]


# ---------------------------------------------------------------------------
# Geometry helpers (unit-tested)
# ---------------------------------------------------------------------------


def expand_box(
    box: Iterable[float],
    width: int,
    height: int,
    margin: float = 0.1,
) -> tuple[int, int, int, int]:
    """Expand a box by ``margin`` on each side and clamp to the image."""
    x1, y1, x2, y2 = (float(value) for value in box)
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    dx = box_width * max(0.0, margin)
    dy = box_height * max(0.0, margin)
    left = max(0, int(round(x1 - dx)))
    top = max(0, int(round(y1 - dy)))
    right = min(width, int(round(x2 + dx)))
    bottom = min(height, int(round(y2 + dy)))
    if right <= left or bottom <= top:
        return 0, 0, 0, 0
    return left, top, right, bottom


def select_crop_indices(detections: list[dict[str, Any]], max_crops: int = 3) -> list[int]:
    """Pick first, middle, and highest-confidence detections, without repeats."""
    if not detections or max_crops <= 0:
        return []
    if len(detections) <= max_crops:
        return list(range(len(detections)))

    middle = len(detections) // 2
    best = max(range(len(detections)), key=lambda index: detections[index]["confidence"])
    chosen = sorted({0, middle, best})

    if len(chosen) < max_crops:
        step = len(detections) / max_crops
        for slot in range(max_crops):
            candidate = min(len(detections) - 1, int(round(slot * step)))
            if candidate not in chosen:
                chosen.append(candidate)
            if len(chosen) >= max_crops:
                break
    return sorted(chosen)[:max_crops]


def _safe_track_name(track_id: int) -> str:
    return f"t{track_id:06d}".replace("-", "m")


def classify_crop_color(image: np.ndarray) -> str:
    """Return a coarse color label for a crop (red/white/black/blue/...)."""
    if image is None or image.size == 0:
        return "unknown"
    height, width = image.shape[:2]
    y1, y2 = int(height * 0.2), max(int(height * 0.2) + 1, int(height * 0.8))
    x1, x2 = int(width * 0.2), max(int(width * 0.2) + 1, int(width * 0.8))
    roi = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    colored = (saturation > 70) & (value > 50)
    if float(colored.mean()) < 0.15:
        brightness = float(value.mean())
        if brightness > 170:
            return "white"
        if brightness < 60:
            return "black"
        return "grey"

    hue_values = hue[colored]
    bins = {
        "red": ((hue_values < 10) | (hue_values >= 170)),
        "orange": (hue_values >= 10) & (hue_values < 22),
        "yellow": (hue_values >= 22) & (hue_values < 35),
        "green": (hue_values >= 35) & (hue_values < 85),
        "blue": (hue_values >= 85) & (hue_values < 130),
        "purple": (hue_values >= 130) & (hue_values < 170),
    }
    counts = {name: int(mask.sum()) for name, mask in bins.items()}
    dominant, count = max(counts.items(), key=lambda item: item[1])
    if count / max(1, len(hue_values)) < 0.35:
        brightness = float(value.mean())
        if brightness > 170:
            return "white"
        if brightness < 60:
            return "black"
        return "grey"
    return dominant


def split_tracks(
    tracks: list[dict[str, Any]],
    *,
    max_gap_s: float = 2.5,
    max_segment_s: float = 12.0,
    class_names: Optional[dict[int, str]] = None,
) -> list[dict[str, Any]]:
    """Split long-lived ByteTrack ids into short, contiguous segments.

    At 1 fps ByteTrack ids can persist for minutes, so choosing only three
    crops for the whole track would miss most objects. Splitting keeps each
    segment short enough that 1-3 crops cover it.
    """
    segments: list[dict[str, Any]] = []
    next_id = 1
    for track in tracks:
        current: list[dict[str, Any]] = []
        for detection in track["detections"]:
            if current:
                gap = detection["timestamp_s"] - current[-1]["timestamp_s"]
                span = detection["timestamp_s"] - current[0]["timestamp_s"]
                if gap > max_gap_s or span > max_segment_s:
                    segments.append(_segment(track, current, next_id, class_names))
                    next_id += 1
                    current = []
            current.append(detection)
        if current:
            segments.append(_segment(track, current, next_id, class_names))
            next_id += 1
    return segments


def _segment(
    track: dict[str, Any],
    detections: list[dict[str, Any]],
    segment_id: int,
    class_names: Optional[dict[int, str]] = None,
) -> dict[str, Any]:
    class_id = Counter(item["class_id"] for item in detections).most_common(1)[0][0]
    resolved_name = (class_names or {}).get(int(class_id), track["class_name"])
    return {
        "track_id": segment_id,
        "parent_track_id": track["track_id"],
        "class_id": int(class_id),
        "class_name": resolved_name,
        "start": float(detections[0]["timestamp_s"]),
        "end": float(detections[-1]["timestamp_s"]),
        "frame_count": len(detections),
        "detections": detections,
    }


def anchor_tracks(
    frame_detections: list[dict[str, Any]],
    class_names: dict[int, str],
    *,
    interval_s: float = 6.0,
    max_per_frame: int = 3,
) -> list[dict[str, Any]]:
    """Add per-frame object crops independent of ByteTrack ids.

    At 1 fps track ids can be reused across unrelated objects, so this anchor
    grid guarantees that prominent objects at regular intervals are indexed.
    """
    if interval_s <= 0 or max_per_frame <= 0:
        return []
    step = max(1, int(round(interval_s)))
    anchors: list[dict[str, Any]] = []
    next_id = -1_000_000
    for entry in frame_detections:
        timestamp = float(entry["timestamp_s"])
        if int(round(timestamp)) % step != 0:
            continue
        detections = sorted(
            entry["detections"],
            key=lambda item: (
                (item["box"][2] - item["box"][0])
                * (item["box"][3] - item["box"][1])
                * item["confidence"]
            ),
            reverse=True,
        )[:max_per_frame]
        for detection in detections:
            anchors.append(
                {
                    "track_id": next_id,
                    "parent_track_id": None,
                    "class_id": int(detection["class_id"]),
                    "class_name": class_names.get(
                        int(detection["class_id"]), str(detection["class_id"])
                    ),
                    "start": timestamp,
                    "end": timestamp,
                    "frame_count": 1,
                    "detections": [detection],
                    "anchor": True,
                }
            )
            next_id -= 1
    return anchors


def collect_anchor_detections(
    frames: list[dict[str, Any]],
    model: Any,
    config: Config,
    *,
    device: str,
    confidence: float,
    classes: list[int],
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """Run plain detection on anchor frames (the tracker can drop boxes)."""
    interval_s = float(config.get("objects.anchor_interval_s", 2.0))
    if interval_s <= 0:
        return []
    step = max(1, int(round(interval_s)))
    selected = [
        frame
        for frame in frames
        if int(round(float(frame["timestamp_s"]))) % step == 0
    ]

    progress = None
    if show_progress and tqdm is not None:
        progress = tqdm(total=len(selected), unit="frame", desc="anchors", dynamic_ncols=True)

    collected: list[dict[str, Any]] = []
    try:
        for frame in selected:
            image = cv2.imread(str(config.resolve(frame["path"])))
            if image is None:
                continue
            results = model.predict(
                source=image,
                conf=confidence,
                classes=classes,
                verbose=False,
                device=device,
            )
            if not results:
                continue
            boxes = results[0].boxes
            if boxes is None or not len(boxes):
                continue
            timestamp = float(frame["timestamp_s"])
            items = [
                {
                    "timestamp_s": timestamp,
                    "box": [float(value) for value in box],
                    "confidence": float(score),
                    "class_id": int(class_id),
                    "frame_path": str(frame["path"]),
                }
                for box, score, class_id in zip(
                    boxes.xyxy.cpu().numpy(),
                    boxes.conf.cpu().numpy(),
                    boxes.cls.cpu().numpy().astype(int),
                )
            ]
            if items:
                collected.append({"timestamp_s": timestamp, "detections": items})
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()
    return collected


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def object_index_paths(config: Config, video_id: str) -> tuple[Path, Path]:
    base = config.indexes_dir / video_id
    return base / "object.faiss", base / "object.ids.json"


def object_crops_dir(config: Config, video_id: str) -> Path:
    return config.data_dir / "objects" / video_id


# ---------------------------------------------------------------------------
# Detection and tracking
# ---------------------------------------------------------------------------


def _read_metadata(config: Config, video_id: str) -> dict[str, Any]:
    metadata_path = config.frames_dir / video_id / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"frame metadata not found: {metadata_path}. "
            f"Run scripts/index_video.py --video <video> first."
        )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not payload.get("frames"):
        raise ValueError(f"no frames listed in {metadata_path}")
    return payload


def track_frames(
    video_id: str,
    config: Config,
    *,
    device: Optional[str] = None,
    show_progress: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run YOLO11 + ByteTrack over the extracted frames and build track records."""
    from ultralytics import YOLO

    metadata = _read_metadata(config, video_id)
    detector_name = str(config.get("objects.model") or DEFAULT_DETECTOR)
    tracker_name = str(config.get("objects.tracker") or "bytetrack.yaml")
    confidence = float(config.get("objects.conf", 0.35))
    classes = [int(value) for value in (config.get("objects.classes") or DEFAULT_CLASSES)]
    resolved_device = resolve_device(device or config.get("objects.device", "auto"))

    _LOG.info("loading detector %s on %s", detector_name, resolved_device)
    model = YOLO(detector_name)
    model.to(resolved_device)

    detections_by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    class_names: dict[int, str] = {int(key): str(value) for key, value in model.names.items()}
    synthetic_id = 1_000_000
    frames = metadata["frames"]

    progress = None
    if show_progress and tqdm is not None:
        progress = tqdm(total=len(frames), unit="frame", desc="detect", dynamic_ncols=True)

    try:
        for frame in frames:
            frame_path = config.resolve(frame["path"])
            image = cv2.imread(str(frame_path))
            if image is None:
                _LOG.warning("unreadable frame: %s", frame_path)
                continue

            results = model.track(
                source=image,
                persist=True,
                tracker=tracker_name,
                conf=confidence,
                classes=classes,
                verbose=False,
                device=resolved_device,
            )
            if results:
                boxes = results[0].boxes
                if boxes is not None and len(boxes):
                    xyxy = boxes.xyxy.cpu().numpy()
                    confidences = boxes.conf.cpu().numpy()
                    class_ids = boxes.cls.cpu().numpy().astype(int)
                    ids = boxes.id
                    if ids is None:
                        track_ids = [synthetic_id + index for index in range(len(xyxy))]
                    else:
                        track_ids = [int(value) for value in ids.int().cpu().tolist()]

                    timestamp = float(frame["timestamp_s"])
                    for track_id, box, score, class_id in zip(
                        track_ids, xyxy, confidences, class_ids
                    ):
                        detections_by_track[int(track_id)].append(
                            {
                                "timestamp_s": timestamp,
                                "box": [float(value) for value in box],
                                "confidence": float(score),
                                "class_id": int(class_id),
                                "frame_path": str(frame["path"]),
                            }
                        )
                        synthetic_id += 1 if ids is None else 0

            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    tracks: list[dict[str, Any]] = []
    for track_id, detections in detections_by_track.items():
        detections.sort(key=lambda item: item["timestamp_s"])
        class_id = Counter(item["class_id"] for item in detections).most_common(1)[0][0]
        tracks.append(
            {
                "track_id": int(track_id),
                "class_id": int(class_id),
                "class_name": class_names.get(int(class_id), str(class_id)),
                "start": float(detections[0]["timestamp_s"]),
                "end": float(detections[-1]["timestamp_s"]),
                "frame_count": len(detections),
                "detections": detections,
            }
        )

    tracks.sort(key=lambda item: (item["start"], item["track_id"]))
    raw_track_count = len(tracks)
    tracks = split_tracks(
        tracks,
        max_gap_s=float(config.get("objects.max_gap_s", 2.5)),
        max_segment_s=float(config.get("objects.max_segment_s", 12.0)),
        class_names=class_names,
    )
    anchors = anchor_tracks(
        collect_anchor_detections(
            frames,
            model,
            config,
            device=resolved_device,
            confidence=confidence,
            classes=classes,
            show_progress=show_progress,
        ),
        class_names,
        interval_s=float(config.get("objects.anchor_interval_s", 2.0)),
        max_per_frame=int(config.get("objects.anchor_max_per_frame", 3)),
    )
    tracks.extend(anchors)
    tracks.sort(key=lambda item: (item["start"], item["track_id"]))
    stats = {
        "detector": detector_name,
        "tracker": tracker_name,
        "device": resolved_device,
        "frames": len(frames),
        "raw_tracks": raw_track_count,
        "segments": len(tracks) - len(anchors),
        "anchors": len(anchors),
        "classes": class_names,
    }
    return tracks, stats


# ---------------------------------------------------------------------------
# Crop selection and persistence
# ---------------------------------------------------------------------------


def _prepare_crop_specs(
    tracks: list[dict[str, Any]],
    config: Config,
    max_crops: Optional[int] = None,
) -> list[tuple[dict[str, Any], int]]:
    """Return (track, detection_index) pairs, capped across the video."""
    max_per_track = int(config.get("objects.max_crops_per_track", 3))
    limit = int(max_crops if max_crops is not None else config.get("objects.max_crops_per_video", 1500))

    specs = []
    for track in tracks:
        for index in select_crop_indices(track["detections"], max_per_track):
            specs.append((track, index))

    if limit > 0 and len(specs) > limit:
        step = len(specs) / limit
        specs = [specs[min(len(specs) - 1, int(round(slot * step)))] for slot in range(limit)]
    return specs


def save_crops(
    video_id: str,
    config: Config,
    tracks: list[dict[str, Any]],
    *,
    max_crops: Optional[int] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Save representative crops and return flat crop records."""
    crops_dir = object_crops_dir(config, video_id)
    crops_dir.mkdir(parents=True, exist_ok=True)
    min_crop = int(config.get("objects.min_crop_px", 32))
    margin = float(config.get("objects.crop_margin", 0.1))

    specs = _prepare_crop_specs(tracks, config, max_crops)
    crops: list[dict[str, Any]] = []
    image_cache: dict[str, np.ndarray | None] = {}

    for track, detection_index in specs:
        detection = track["detections"][detection_index]
        frame_path = config.resolve(detection["frame_path"])
        cache_key = str(frame_path)
        if cache_key not in image_cache:
            image_cache[cache_key] = cv2.imread(str(frame_path))
        image = image_cache[cache_key]
        if image is None:
            continue

        height, width = image.shape[:2]
        left, top, right, bottom = expand_box(detection["box"], width, height, margin)
        if right - left < min_crop or bottom - top < min_crop:
            continue

        crop = image[top:bottom, left:right]
        if crop.size == 0:
            continue

        crop_index = sum(1 for existing in crops if existing["track_id"] == track["track_id"])
        crop_path = crops_dir / f"{_safe_track_name(track['track_id'])}_{crop_index}.jpg"
        if not cv2.imwrite(str(crop_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            _LOG.warning("could not write crop: %s", crop_path)
            continue

        crops.append(
            {
                "track_id": track["track_id"],
                "class_id": track["class_id"],
                "class_name": track["class_name"],
                "start": track["start"],
                "end": track["end"],
                "frame_count": track["frame_count"],
                "path": store.relative_path(crop_path, config.repo_root),
                "timestamp_s": float(detection["timestamp_s"]),
                "confidence": float(detection["confidence"]),
                "bbox": [round(float(value), 2) for value in detection["box"]],
                "crop_bbox": [left, top, right, bottom],
                "color": classify_crop_color(crop),
                "brightness": round(
                    float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean()), 2
                ),
            }
        )

    image_cache.clear()
    return crops, {"crops_dir": store.relative_path(crops_dir, config.repo_root)}


# ---------------------------------------------------------------------------
# Cache and index build
# ---------------------------------------------------------------------------


def _cached_index(
    ids_path: Path,
    index_path: Path,
    metadata: dict[str, Any],
    detector: str,
    embedding_model: str,
    object_config: dict[str, Any],
) -> Optional[dict[str, Any]]:
    if not index_path.is_file() or not ids_path.is_file():
        return None
    try:
        payload = store.read_json(ids_path)
    except store.StoreError:
        return None
    if payload.get("detector") != detector:
        return None
    if payload.get("embedding_model") != embedding_model:
        return None
    if payload.get("object_config") != object_config:
        return None
    if payload.get("source_hash") != metadata.get("source_hash"):
        return None
    if payload.get("count") != len(payload.get("records") or []):
        return None
    return payload


def index_object_video(
    video_id: str,
    config: Config,
    *,
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
    overwrite: bool = False,
    show_progress: bool = True,
    max_crops: Optional[int] = None,
) -> tuple[dict[str, Any], bool]:
    """Build (or reuse) the Layer B index for one video."""
    metadata = _read_metadata(config, video_id)
    detector_name = str(config.get("objects.model") or DEFAULT_DETECTOR)
    embedding_model = str(config.get("frames.model") or DEFAULT_MODEL)
    batch_size = int(batch_size or config.get("frames.batch_size", 32))
    resolved_device = resolve_device(device or config.get("objects.device", "auto"))

    index_path, ids_path = object_index_paths(config, video_id)
    object_config = {
        "tracker": str(config.get("objects.tracker") or "bytetrack.yaml"),
        "conf": float(config.get("objects.conf", 0.35)),
        "classes": [int(value) for value in (config.get("objects.classes") or DEFAULT_CLASSES)],
        "min_crop_px": int(config.get("objects.min_crop_px", 32)),
        "max_crops_per_track": int(config.get("objects.max_crops_per_track", 3)),
        "max_segment_s": float(config.get("objects.max_segment_s", 12.0)),
        "max_gap_s": float(config.get("objects.max_gap_s", 2.5)),
        "anchor_interval_s": float(config.get("objects.anchor_interval_s", 6.0)),
        "anchor_max_per_frame": int(config.get("objects.anchor_max_per_frame", 3)),
        "crop_margin": float(config.get("objects.crop_margin", 0.1)),
        "max_crops_per_video": int(
            max_crops
            if max_crops is not None
            else config.get("objects.max_crops_per_video", 1500)
        ),
    }
    if not overwrite:
        cached = _cached_index(
            ids_path,
            index_path,
            metadata,
            detector_name,
            embedding_model,
            object_config,
        )
        if cached is not None:
            _LOG.info(
                "cache hit: %s object index (%d crops, %s)",
                video_id,
                int(cached.get("count", 0)),
                detector_name,
            )
            return cached, True
        if index_path.exists() or ids_path.exists():
            _LOG.info("object index stale; rebuilding %s", video_id)

    with StageTimer("track", _LOG):
        tracks, track_stats = track_frames(
            video_id,
            config,
            device=resolved_device,
            show_progress=show_progress,
        )

    with StageTimer("crops", _LOG):
        crops_dir = object_crops_dir(config, video_id)
        if overwrite and crops_dir.exists():
            for stale in crops_dir.glob("*.jpg"):
                stale.unlink()
        crops, crop_stats = save_crops(video_id, config, tracks, max_crops=max_crops)

    if not crops:
        payload = {
            "schema_version": 1,
            "video_id": video_id,
            "layer": "object",
            "detector": detector_name,
            "tracker": track_stats["tracker"],
            "embedding_model": embedding_model,
            "device": resolved_device,
            "dim": None,
            "normalized": True,
            "count": 0,
            "track_count": 0,
            "crop_count": 0,
            "empty": True,
            "source_path": metadata.get("source_path"),
            "source_hash": metadata.get("source_hash"),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "classes": track_stats["classes"],
            "class_counts": {},
            "object_config": object_config,
            "crops_dir": crop_stats["crops_dir"],
            "tracks": [],
            "records": [],
        }
        store.write_json(ids_path, payload)
        _LOG.warning(
            "%s contains no COCO objects after filtering; wrote an empty object sidecar",
            video_id,
        )
        return payload, False

    processor, model = load_siglip(embedding_model, resolved_device)
    crop_paths = [config.resolve(crop["path"]) for crop in crops]
    with StageTimer("embed crops", _LOG):
        vectors = embed_images(
            crop_paths,
            processor,
            model,
            device=resolved_device,
            batch_size=batch_size,
            show_progress=show_progress,
        )
    if vectors.shape[0] != len(crops):
        raise RuntimeError(
            f"crop embedding count {vectors.shape[0]} does not match crops {len(crops)}"
        )

    with StageTimer("faiss index", _LOG):
        index = store.build_flat_ip(vectors)
        store.save_faiss(index, index_path)

    records = []
    for position, crop in enumerate(crops):
        records.append(
            {
                "doc_id": f"{video_id}:object:{crop['track_id']}:{position}",
                "video_id": video_id,
                "layer": "object",
                "start": float(crop["start"]),
                "end": float(crop["end"]),
                "path": crop["path"],
                "text": None,
                "meta": {
                    "track_id": int(crop["track_id"]),
                    "class_id": int(crop["class_id"]),
                    "class_name": crop["class_name"],
                    "confidence": float(crop["confidence"]),
                    "timestamp_s": float(crop["timestamp_s"]),
                    "color": crop.get("color"),
                    "brightness": crop.get("brightness"),
                    "bbox": crop["bbox"],
                    "crop_bbox": crop["crop_bbox"],
                    "position": position,
                },
            }
        )

    track_records = [
        {
            "track_id": int(track["track_id"]),
            "class_id": int(track["class_id"]),
            "class_name": track["class_name"],
            "start": float(track["start"]),
            "end": float(track["end"]),
            "frame_count": int(track["frame_count"]),
            "crops": [
                {
                    "path": crop["path"],
                    "timestamp_s": crop["timestamp_s"],
                    "confidence": crop["confidence"],
                    "color": crop.get("color"),
                    "brightness": crop.get("brightness"),
                    "bbox": crop["bbox"],
                }
                for crop in crops
                if crop["track_id"] == track["track_id"]
            ],
        }
        for track in tracks
    ]
    track_records = [track for track in track_records if track["crops"]]

    payload = {
        "schema_version": 1,
        "video_id": video_id,
        "layer": "object",
        "detector": detector_name,
        "tracker": track_stats["tracker"],
        "embedding_model": embedding_model,
        "device": resolved_device,
        "dim": int(vectors.shape[1]),
        "normalized": True,
        "count": len(records),
        "track_count": len(track_records),
        "crop_count": len(crops),
        "source_path": metadata.get("source_path"),
        "source_hash": metadata.get("source_hash"),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "classes": track_stats["classes"],
        "class_counts": dict(
            Counter(record["meta"]["class_name"] for record in records).most_common()
        ),
        "object_config": object_config,
        "crops_dir": crop_stats["crops_dir"],
        "tracks": track_records,
        "records": records,
    }
    store.write_json(ids_path, payload)
    _LOG.info(
        "wrote %d object vectors from %d tracks to %s",
        len(records),
        len(track_records),
        store.relative_path(index_path, config.repo_root),
    )
    return payload, False
