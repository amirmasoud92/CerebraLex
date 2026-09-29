"""YAML loading with dotted-key overrides, validation and configuration snapshots."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from common.hashing import sha256_text
from common.io import write_json


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return data


def apply_overrides(data: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``a.b.c=value`` overrides; values are parsed as YAML scalars."""
    result = json.loads(json.dumps(data, default=str))
    for item in overrides:
        key, sep, raw_value = item.partition("=")
        if not sep:
            raise ValueError(f"override '{item}' must have the form key=value")
        cursor = result
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"cannot descend into non-mapping at '{part}' for '{item}'")
        cursor[parts[-1]] = yaml.safe_load(raw_value)
    return result


def load_config[TModel: BaseModel](
    path: Path,
    model: type[TModel],
    overrides: Iterable[str] | None = None,
    section: str | None = None,
) -> TModel:
    """Load ``path``, optionally take ``section``, apply overrides and validate as ``model``."""
    data = load_yaml(path)
    if section is not None:
        data = data.get(section, {})
    if overrides:
        data = apply_overrides(data, overrides)
    return model.model_validate(data)


def config_fingerprint(config: BaseModel) -> str:
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return sha256_text(canonical)


def snapshot_config(config: BaseModel, out_dir: Path, name: str) -> Path:
    """Persist the exact configuration used for a run, keyed by its content hash."""
    fingerprint = config_fingerprint(config)
    target = out_dir / f"{name}_{fingerprint[:12]}.json"
    if target.exists():
        return target
    payload = {"fingerprint": fingerprint, "config": config.model_dump(mode="json")}
    return write_json(payload, target)
