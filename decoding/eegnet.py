"""EEGNet regressor trained from scratch on raw windows, GPU-resident, bf16 autocast, early stopping.

EEGNet (Lawhern et al. 2018) was designed for 128 Hz input with a 64-sample (0.5 s) temporal
kernel. Windows stored at 250 Hz, already low-passed at 45 Hz, are decimated by two to 125 Hz
(alias-free: the filter's stop band ends below the new 62.5 Hz Nyquist frequency) so the
published kernel and pooling sizes keep their temporal meaning.

Training windows are standardised per channel with statistics from a subsample of the training
fold, converted to float16 and kept on the GPU when they fit in 60% of free memory (otherwise
in pinned host memory), which removes per-epoch disk reads.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from common.device import DeviceManager
from common.schema import ComputeConfig
from decoding.base import Decoder, DecoderInput
from decoding.leakage import LeakageAuditor

_LOG = logging.getLogger(__name__)
_GPU_FRACTION = 0.6


class EinsumSpatialConv(torch.nn.Module):
    """EEGNet's depthwise spatial convolution computed as an einsum over channels.

    Identical to the grouped (channels x 1) convolution it wraps (same constrained weight, same
    output); PyTorch's native depthwise backward kernel for a 129-tall filter dominated the
    training time on 129-channel data, while the einsum runs as batched matrix products.
    """

    def __init__(self, conv: torch.nn.Module) -> None:
        super().__init__()
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: (batch, F1, channels, time)
        weight = self.conv.weight                             # (F1 * D, 1, channels, 1)
        groups = x.shape[1]
        depth = weight.shape[0] // groups
        w = weight[:, 0, :, 0].reshape(groups, depth, -1).to(x.dtype)
        out = torch.einsum("bgct,gdc->bgdt", x, w).reshape(x.shape[0], groups * depth, 1, x.shape[-1])
        if self.conv.bias is not None:
            out = out + self.conv.bias.view(1, -1, 1, 1)
        return out


class EEGNetRegressor(Decoder):
    name = "eegnet"

    def __init__(self, n_channels: int, n_times: int, compute: ComputeConfig | None = None,
                 max_epochs: int = 40, batch_size: int = 256, lr: float = 1e-3,
                 weight_decay: float = 1e-2, patience: int = 6, val_fraction: float = 0.1,
                 seeds: tuple[int, ...] = (0, 1, 2), input_scale: float = 1e6,
                 stats_subsample: int = 2000, decimate: int = 2, kernel_length: int = 64,
                 load_chunk: int = 2048, fast_spatial: bool = True, autocast: bool = False) -> None:
        self.n_channels = n_channels
        self.n_times = n_times
        self.compute = compute or ComputeConfig()
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_fraction = val_fraction
        self.seeds = tuple(seeds)
        self.input_scale = input_scale
        self.stats_subsample = stats_subsample
        self.decimate = decimate
        self.kernel_length = kernel_length
        self.load_chunk = load_chunk
        self.fast_spatial = fast_spatial
        self.autocast = autocast

    @property
    def model_n_times(self) -> int:
        return int(np.ceil(self.n_times / self.decimate))

    def _build(self, seed: int) -> torch.nn.Module:
        from braindecode.models import EEGNet

        torch.manual_seed(seed)
        model = EEGNet(n_chans=self.n_channels, n_outputs=1, n_times=self.model_n_times,
                       kernel_length=self.kernel_length)
        if self.fast_spatial:
            model.conv_spatial = EinsumSpatialConv(model.conv_spatial)
        return model

    def _autocast(self, manager: DeviceManager):
        """bf16 autocast only when requested: on 129-channel EEGNet it measured slower than fp32."""
        import contextlib

        return manager.autocast() if self.autocast else contextlib.nullcontext()

    def _read(self, x, start: int, stop: int) -> np.ndarray:
        block = np.asarray(x[np.arange(start, stop)] if not isinstance(x, np.ndarray) else x[start:stop],
                           dtype=np.float32)
        return block[..., :: self.decimate] * self.input_scale

    def _fit_stats(self, x, seed: int) -> None:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(x), size=min(self.stats_subsample, len(x)), replace=False))
        sample = np.stack([self._read(x, int(i), int(i) + 1)[0] for i in idx])
        self.channel_mean_ = sample.mean(axis=(0, 2))[:, None].astype(np.float32)
        self.channel_std_ = (sample.std(axis=(0, 2))[:, None] + 1e-6).astype(np.float32)

    def _load(self, x, manager: DeviceManager) -> torch.Tensor:
        """Standardised float16 tensor of all rows, on the GPU when it fits."""
        n = len(x)
        shape = (n, self.n_channels, self.model_n_times)
        needed = int(np.prod(shape)) * 2
        on_gpu = False
        if manager.is_cuda:
            free, _ = torch.cuda.mem_get_info(manager.device)
            on_gpu = needed < _GPU_FRACTION * free
        device = manager.device if on_gpu else torch.device("cpu")
        out = torch.empty(shape, dtype=torch.float16, device=device,
                          pin_memory=(manager.is_cuda and not on_gpu))
        mean = torch.from_numpy(self.channel_mean_)
        std = torch.from_numpy(self.channel_std_)
        for start in range(0, n, self.load_chunk):
            stop = min(start + self.load_chunk, n)
            chunk = (torch.from_numpy(self._read(x, start, stop)) - mean) / std
            out[start:stop] = chunk.to(dtype=torch.float16, device=device)
        self.storage_ = "gpu" if on_gpu else "host"
        return out

    def _split_validation(self, subjects: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        unique = np.unique(subjects)
        n_val = max(1, int(round(self.val_fraction * len(unique))))
        val_subjects = set(rng.choice(unique, size=n_val, replace=False))
        val_mask = np.array([s in val_subjects for s in subjects])
        return ~val_mask, val_mask

    def _batches(self, data: torch.Tensor, index: torch.Tensor, device: torch.device):
        for start in range(0, len(index), self.batch_size):
            idx = index[start:start + self.batch_size]
            if data.device.type != "cpu":
                yield data[idx.to(data.device)].float(), idx
            else:
                yield data[idx.cpu()].to(device, non_blocking=True).float(), idx

    def fit(self, data: DecoderInput, y: np.ndarray) -> EEGNetRegressor:
        manager = DeviceManager(self.compute)
        device = manager.device
        self._fit_stats(data.x, self.seeds[0])
        x_all = self._load(data.x, manager)
        y = np.asarray(y, dtype=np.float32)
        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std() + 1e-6)
        y_all = torch.from_numpy((y - self.y_mean_) / self.y_std_).to(device)
        self.models_: list[torch.nn.Module] = []
        self.history_: list[dict] = []
        for seed in self.seeds:
            train_mask, val_mask = self._split_validation(data.subjects, seed)
            LeakageAuditor.assert_disjoint(data.subjects[train_mask], data.subjects[val_mask])
            train_idx = torch.from_numpy(np.flatnonzero(train_mask)).to(device)
            val_idx = torch.from_numpy(np.flatnonzero(val_mask)).to(device)
            generator = torch.Generator(device=device).manual_seed(seed)
            model = self._build(seed).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr,
                                          weight_decay=self.weight_decay)
            best_state, best_val, bad_epochs, epoch_used = None, np.inf, 0, 0
            for epoch in range(self.max_epochs):
                model.train()
                order = train_idx[torch.randperm(len(train_idx), device=device, generator=generator)]
                for xb, idx in self._batches(x_all, order, device):
                    optimizer.zero_grad(set_to_none=True)
                    with self._autocast(manager):
                        out = model(xb).squeeze(-1).float()
                    loss = torch.nn.functional.mse_loss(out, y_all[idx.to(device)])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                val_loss = self._evaluate(model, x_all, y_all, val_idx, manager)
                if val_loss < best_val - 1e-4:
                    best_val, bad_epochs = val_loss, 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    epoch_used = epoch + 1
                else:
                    bad_epochs += 1
                    if bad_epochs >= self.patience:
                        break
            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            self.models_.append(model)
            self.history_.append({"seed": seed, "best_val_mse_z": best_val, "epochs": epoch_used,
                                  "n_train": int(train_mask.sum()), "n_val": int(val_mask.sum())})
        self.fitted_subjects_ = np.unique(data.subjects)
        self.device_info_ = manager.describe()
        self.memory_ = manager.memory_report()
        del x_all, y_all
        manager.empty_cache()
        return self

    def _evaluate(self, model: torch.nn.Module, x_all: torch.Tensor, y_all: torch.Tensor,
                  index: torch.Tensor, manager: DeviceManager) -> float:
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad(), self._autocast(manager):
            for xb, idx in self._batches(x_all, index, manager.device):
                pred = model(xb).squeeze(-1).float()
                target = y_all[idx.to(manager.device)]
                total += float(((pred - target) ** 2).sum())
                n += len(idx)
        return total / max(n, 1)

    def predict(self, data: DecoderInput) -> np.ndarray:
        LeakageAuditor.assert_fit_subjects(self.fitted_subjects_, data.subjects, self.name)
        manager = DeviceManager(self.compute)
        x_all = self._load(data.x, manager)
        index = torch.arange(len(data), device=manager.device)
        preds = np.zeros(len(data), dtype=np.float64)
        with torch.no_grad(), self._autocast(manager):
            for model in self.models_:
                model.to(manager.device).eval()
                out = [model(xb).squeeze(-1).float().cpu().numpy()
                       for xb, _ in self._batches(x_all, index, manager.device)]
                preds += np.concatenate(out)
        del x_all
        manager.empty_cache()
        preds /= len(self.models_)
        return preds * self.y_std_ + self.y_mean_

    def describe(self) -> dict:
        return {"name": self.name, "max_epochs": self.max_epochs, "batch_size": self.batch_size,
                "lr": self.lr, "weight_decay": self.weight_decay, "seeds": list(self.seeds),
                "decimate": self.decimate, "kernel_length": self.kernel_length,
                "fast_spatial": self.fast_spatial, "autocast": self.autocast,
                "storage": getattr(self, "storage_", None),
                "history": getattr(self, "history_", None),
                "device": getattr(self, "device_info_", None),
                "memory": getattr(self, "memory_", None)}
