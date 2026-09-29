"""Device selection, mixed-precision policy and GPU memory accounting."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import psutil
import torch

from common.schema import ComputeConfig

_LOG = logging.getLogger(__name__)


class DeviceManager:
    """Resolve the execution device from configuration with an explicit CPU fallback."""

    def __init__(self, config: ComputeConfig) -> None:
        self.config = config
        self._device = self._resolve_device()
        self._amp_dtype = self._resolve_amp_dtype()
        _LOG.info("device=%s amp_dtype=%s", self._device, self._amp_dtype)

    def _resolve_device(self) -> torch.device:
        cuda_ok = torch.cuda.is_available()
        if self.config.device == "cpu":
            return torch.device("cpu")
        if self.config.device == "cuda":
            if not cuda_ok:
                raise RuntimeError("configuration requires CUDA but no CUDA device is available")
            return torch.device(f"cuda:{self.config.cuda_device_index}")
        if cuda_ok:
            return torch.device(f"cuda:{self.config.cuda_device_index}")
        _LOG.warning("CUDA unavailable; falling back to CPU")
        return torch.device("cpu")

    def _resolve_amp_dtype(self) -> torch.dtype | None:
        precision = self.config.precision
        if precision == "fp32":
            return None
        if precision == "bf16":
            if self.is_cuda and not torch.cuda.is_bf16_supported():
                _LOG.warning("bf16 not supported on this GPU; using fp32")
                return None
            return torch.bfloat16
        if not self.is_cuda:
            _LOG.warning("fp16 autocast is unreliable on CPU; using fp32")
            return None
        return torch.float16

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def is_cuda(self) -> bool:
        return self._device.type == "cuda"

    @property
    def amp_dtype(self) -> torch.dtype | None:
        return self._amp_dtype

    @contextmanager
    def autocast(self) -> Iterator[None]:
        enabled = self._amp_dtype is not None
        dtype = self._amp_dtype or torch.float32
        with torch.autocast(device_type=self._device.type, dtype=dtype, enabled=enabled):
            yield

    def memory_report(self) -> dict[str, float]:
        report = {"process_rss_gb": psutil.Process().memory_info().rss / 1e9}
        if self.is_cuda:
            index = self._device.index or 0
            report.update({
                "cuda_allocated_gb": torch.cuda.memory_allocated(index) / 1e9,
                "cuda_reserved_gb": torch.cuda.memory_reserved(index) / 1e9,
                "cuda_max_allocated_gb": torch.cuda.max_memory_allocated(index) / 1e9,
                "cuda_total_gb": torch.cuda.get_device_properties(index).total_memory / 1e9,
            })
        return report

    def empty_cache(self) -> None:
        if self.is_cuda:
            torch.cuda.empty_cache()

    def describe(self) -> dict[str, object]:
        info: dict[str, object] = {
            "device": str(self._device),
            "precision": self.config.precision,
            "amp_dtype": str(self._amp_dtype),
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory and self.is_cuda,
            "seed": self.config.seed,
            "deterministic": self.config.deterministic,
            "torch_version": torch.__version__,
        }
        if self.is_cuda:
            info["gpu_name"] = torch.cuda.get_device_name(self._device)
        return info
