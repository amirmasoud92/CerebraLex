"""Content hashing for provenance and stable subject-to-split assignment."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_subject_hash(subject_id: str, salt: str) -> str:
    """Salted hash of a subject identifier; the salt is fixed per split scheme."""
    return sha256_text(f"{salt}:{subject_id}")


def hash_to_unit_interval(hexdigest: str) -> float:
    """Map the first 64 bits of a hex digest to [0, 1) for reproducible assignment."""
    return int(hexdigest[:16], 16) / float(1 << 64)
