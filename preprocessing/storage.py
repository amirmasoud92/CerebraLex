"""HDF5 storage of cleaned blocks with enough metadata to reconstruct the uncleaned signal."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

FORMAT_VERSION = 1


@dataclass
class CleanedBlock:
    data: np.ndarray                   # channels x times, float32, volts
    sfreq: float
    ch_names: list[str]
    crop_tmin_s: float                 # seconds in the original file at sample 0
    block_key: str
    subject: str
    release: str
    block: int
    config_fingerprint: str
    removed_sources: np.ndarray | None = None   # k x times, ICA sources removed
    mixing_removed: np.ndarray | None = None    # channels x k
    attrs: dict = field(default_factory=dict)

    @property
    def n_times(self) -> int:
        return int(self.data.shape[1])

    def uncleaned(self) -> np.ndarray:
        """Signal before ICA removal, reconstructed from the stored removed components."""
        if self.removed_sources is None or self.mixing_removed is None:
            return self.data
        return self.data + self.mixing_removed.astype(np.float32) @ self.removed_sources

    def file_time_to_sample(self, t_file_s: float) -> int:
        return int(round((t_file_s - self.crop_tmin_s) * self.sfreq))


def write_block(block: CleanedBlock, path: Path, compression: str | None = "lzf") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(tmp, "w") as handle:
        handle.attrs["format_version"] = FORMAT_VERSION
        handle.attrs["sfreq"] = float(block.sfreq)
        handle.attrs["crop_tmin_s"] = float(block.crop_tmin_s)
        handle.attrs["block_key"] = block.block_key
        handle.attrs["subject"] = block.subject
        handle.attrs["release"] = block.release
        handle.attrs["block"] = int(block.block)
        handle.attrs["config_fingerprint"] = block.config_fingerprint
        handle.attrs["ch_names"] = json.dumps(block.ch_names)
        handle.attrs["extra"] = json.dumps(block.attrs, default=str)
        handle.create_dataset("eeg", data=block.data.astype(np.float32),
                              chunks=(block.data.shape[0], min(4096, block.n_times)),
                              compression=compression)
        if block.removed_sources is not None and block.mixing_removed is not None:
            handle.create_dataset("removed_sources", data=block.removed_sources.astype(np.float32),
                                  compression=compression)
            handle.create_dataset("mixing_removed", data=block.mixing_removed.astype(np.float32))
    tmp.replace(path)
    return path


def read_block(path: Path, load_removed: bool = False) -> CleanedBlock:
    with h5py.File(path, "r") as handle:
        removed = handle["removed_sources"][()] if load_removed and "removed_sources" in handle else None
        mixing = handle["mixing_removed"][()] if load_removed and "mixing_removed" in handle else None
        return CleanedBlock(
            data=handle["eeg"][()],
            sfreq=float(handle.attrs["sfreq"]),
            ch_names=json.loads(handle.attrs["ch_names"]),
            crop_tmin_s=float(handle.attrs["crop_tmin_s"]),
            block_key=str(handle.attrs["block_key"]),
            subject=str(handle.attrs["subject"]),
            release=str(handle.attrs["release"]),
            block=int(handle.attrs["block"]),
            config_fingerprint=str(handle.attrs["config_fingerprint"]),
            removed_sources=removed,
            mixing_removed=mixing,
            attrs=json.loads(handle.attrs["extra"]),
        )


def read_block_header(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        return {k: (json.loads(v) if k in ("ch_names", "extra") else v)
                for k, v in handle.attrs.items()} | {"n_times": int(handle["eeg"].shape[1])}
