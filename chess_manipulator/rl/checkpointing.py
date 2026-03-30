"""Checkpoint metadata helpers for offline RL training and promotion."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from shutil import copy2
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def checkpoint_metadata_path(checkpoint_path: Path | str) -> Path:
    path = Path(checkpoint_path)
    return path.with_name(f"{path.name}.metadata.json")


def load_checkpoint_metadata(checkpoint_path: Path | str) -> dict[str, Any]:
    metadata_path = checkpoint_metadata_path(checkpoint_path)
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def save_checkpoint_metadata(checkpoint_path: Path | str, metadata: dict[str, Any]) -> Path:
    metadata_path = checkpoint_metadata_path(checkpoint_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata_path


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
            continue
        merged[key] = value
    return merged


def append_checkpoint_event(
    checkpoint_path: Path | str,
    event_type: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = load_checkpoint_metadata(checkpoint_path)
    event = {
        "type": event_type,
        "timestamp": _utc_now_iso(),
        "payload": payload or {},
    }
    events = list(metadata.get("events", []))
    events.append(event)
    metadata["events"] = events
    metadata["checkpoint"] = str(Path(checkpoint_path))
    save_checkpoint_metadata(checkpoint_path, metadata)
    return metadata


def merge_checkpoint_metadata(
    checkpoint_path: Path | str,
    updates: dict[str, Any],
) -> dict[str, Any]:
    metadata = load_checkpoint_metadata(checkpoint_path)
    metadata = _deep_merge(metadata, updates)
    metadata["checkpoint"] = str(Path(checkpoint_path))
    save_checkpoint_metadata(checkpoint_path, metadata)
    return metadata


def promote_checkpoint(
    source_checkpoint: Path | str,
    destination_checkpoint: Path | str,
    promotion_metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = Path(source_checkpoint)
    destination = Path(destination_checkpoint)
    if not source.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {source}")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination checkpoint already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    copy2(source, destination)

    source_metadata = load_checkpoint_metadata(source)
    merged_metadata = _deep_merge(
        source_metadata,
        {
            "promotion": {
                "source_checkpoint": str(source),
                "destination_checkpoint": str(destination),
                "promoted_at": _utc_now_iso(),
                **(promotion_metadata or {}),
            }
        },
    )
    merged_metadata["checkpoint"] = str(destination)
    save_checkpoint_metadata(destination, merged_metadata)
    return merged_metadata
