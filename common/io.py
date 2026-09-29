"""Filesystem helpers that never overwrite existing outputs without an explicit request."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

OnExists = Literal["error", "version", "overwrite"]


class OutputExistsError(FileExistsError):
    """Raised when an output path exists and overwriting was not requested."""


def utc_timestamp() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")


def versioned_path(path: Path) -> Path:
    """Return ``path`` if free, else the first free ``{stem}_v{n}{suffix}`` with n >= 2."""
    if not path.exists():
        return path
    n = 2
    while True:
        candidate = path.with_name(f"{path.stem}_v{n}{path.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def resolve_output_path(path: Path, on_exists: OnExists = "error") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or on_exists == "overwrite":
        return path
    if on_exists == "version":
        return versioned_path(path)
    raise OutputExistsError(
        f"{path} exists; pass on_exists='version' or 'overwrite' to proceed deliberately"
    )


def write_json(obj: Any, path: Path, on_exists: OnExists = "error", indent: int = 2) -> Path:
    target = resolve_output_path(path, on_exists)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=indent, default=str, ensure_ascii=False)
        handle.write("\n")
    return target


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_dataframe(frame: pd.DataFrame, path: Path, on_exists: OnExists = "error") -> Path:
    """Write a DataFrame as CSV, TSV or Parquet depending on the suffix."""
    target = resolve_output_path(path, on_exists)
    suffix = target.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(target, index=False, encoding="utf-8")
    elif suffix == ".tsv":
        frame.to_csv(target, index=False, sep="\t", encoding="utf-8")
    elif suffix == ".parquet":
        frame.to_parquet(target, index=False)
    else:
        raise ValueError(f"unsupported table format: {suffix}")
    return target


def read_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", na_values=["n/a"])
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported table format: {suffix}")


def write_text(text: str, path: Path, on_exists: OnExists = "error") -> Path:
    target = resolve_output_path(path, on_exists)
    target.write_text(text, encoding="utf-8")
    return target
