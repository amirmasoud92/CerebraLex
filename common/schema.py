"""Pydantic configuration models. Every tunable parameter enters the pipeline here."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_RAW_ROOT_ENV = "CEREBRALEX_RAW_ROOT"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BIDSRootSpec(StrictModel):
    """One folder of the local raw store.

    ``release_subdirs`` folders contain one BIDS dataset per release (``R1`` ... ``NC``);
    ``single_release`` folders are themselves one BIDS dataset for ``release``.
    """

    path: Path
    layout: Literal["release_subdirs", "single_release"]
    release: str | None = None

    @model_validator(mode="after")
    def _check_release(self) -> BIDSRootSpec:
        if self.layout == "single_release" and not self.release:
            raise ValueError("single_release layout requires 'release'")
        if self.layout == "release_subdirs" and self.release:
            raise ValueError("release_subdirs layout must not set 'release'")
        return self


class RawDataConfig(StrictModel):
    """Location and expected properties of the immutable raw HBN-EEG store."""

    raw_root: Path
    bids_roots: list[BIDSRootSpec]
    releases: list[str] = Field(default_factory=lambda: [f"R{i}" for i in range(1, 12)] + ["NC"])
    expected_sampling_rate_hz: float = 500.0
    expected_eeg_channel_count: int = 129
    expected_reference: str = "Cz"
    expected_power_line_hz: float = 60.0

    @model_validator(mode="after")
    def _apply_env_override(self) -> RawDataConfig:
        override = os.environ.get(_RAW_ROOT_ENV)
        if override:
            object.__setattr__(self, "raw_root", Path(override))
        return self

    def resolve(self, spec: BIDSRootSpec) -> Path:
        return spec.path if spec.path.is_absolute() else self.raw_root / spec.path


class DerivativesConfig(StrictModel):
    """Root of generated intermediate data, kept on fast local storage."""

    root: Path
    preprocessing_version: str = "preproc-v1"
    manifest_dirname: str = "manifest"

    @model_validator(mode="after")
    def _apply_env_override(self) -> DerivativesConfig:
        override = os.environ.get("CEREBRALEX_DERIVATIVES_ROOT")
        if override:
            object.__setattr__(self, "root", Path(override))
        return self

    @property
    def manifest_dir(self) -> Path:
        return self.root / self.manifest_dirname

    @property
    def preprocessing_dir(self) -> Path:
        return self.root / self.preprocessing_version


class ComputeConfig(StrictModel):
    """Execution backend, precision, workers and seeds."""

    device: Literal["auto", "cuda", "cpu"] = "auto"
    cuda_device_index: int = 0
    precision: Literal["fp32", "bf16", "fp16"] = "bf16"
    num_workers: int = 8
    pin_memory: bool = True
    seed: int = 12345
    deterministic: bool = True


class FilterConfig(StrictModel):
    l_freq: float = 0.5
    h_freq: float = 45.0
    ica_l_freq: float = 1.0
    ica_h_freq: float | None = 100.0
    ica_notch: bool = True
    notch_freqs: list[float] = Field(default_factory=lambda: [60.0, 120.0, 180.0])
    fir_design: Literal["firwin"] = "firwin"

    @model_validator(mode="after")
    def _check_bands(self) -> FilterConfig:
        if self.h_freq <= self.l_freq:
            raise ValueError("h_freq must exceed l_freq")
        if self.ica_h_freq is not None and self.ica_h_freq <= self.ica_l_freq:
            raise ValueError("ica_h_freq must exceed ica_l_freq")
        return self


class BadChannelConfig(StrictModel):
    ransac: bool = True
    max_bad_fraction: float = 0.2
    random_state: int = 12345


class ICAConfig(StrictModel):
    enabled: bool = True
    method: Literal["infomax", "fastica", "picard"] = "infomax"
    extended: bool = True
    max_components: int = 60
    reject_labels: list[str] = Field(default_factory=lambda: [
        "eye blink", "muscle artifact", "heart beat", "line noise", "channel noise",
    ])
    reject_threshold: float = 0.8
    random_state: int = 12345


class PreprocessingConfig(StrictModel):
    """Preprocessing parameters (preprocessing/preprocessing.yaml)."""

    version: str = "preproc-v1"
    montage: str = "GSN-HydroCel-129"
    resample_hz: float = 250.0
    crop_pre_margin_s: float = 1.0
    crop_post_margin_s: float = 3.0
    filter: FilterConfig = Field(default_factory=FilterConfig)
    bad_channels: BadChannelConfig = Field(default_factory=BadChannelConfig)
    ica: ICAConfig = Field(default_factory=ICAConfig)
    n_jobs: int = 12
    threads_per_job: int = 2
    hdf5_compression: Literal["lzf", "gzip"] | None = "lzf"


class QCConfig(StrictModel):
    """Analysis-exclusion rules applied after preprocessing."""

    common_mode_sd_cv_threshold: float = 0.01
    amplitude_outlier_n_mad: float = 3.0
    exclude_common_mode: bool = True


class WindowSpec(StrictModel):
    """Analysis window relative to the contrast-change onset (seconds)."""

    name: str
    tmin_s: float
    tmax_s: float
    description: str = ""

    @model_validator(mode="after")
    def _check_order(self) -> WindowSpec:
        if self.tmax_s <= self.tmin_s:
            raise ValueError("tmax_s must exceed tmin_s")
        return self

    @property
    def length_s(self) -> float:
        return self.tmax_s - self.tmin_s


class TrialConfig(StrictModel):
    """Trial-table and analysis-window parameters."""

    lapse_rt_factor: float = 1.5
    min_valid_responses_per_subject: int = 10
    windows: list[WindowSpec] = Field(default_factory=lambda: [
        WindowSpec(name="prestim", tmin_s=-2.0, tmax_s=0.0,
                   description="pre-stimulus vigilance window (primary)"),
        WindowSpec(name="challenge", tmin_s=0.5, tmax_s=2.5,
                   description="NeurIPS 2025 Challenge 1 window (external reference)"),
    ])


class EEGNetConfig(StrictModel):
    max_epochs: int = 40
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-2
    patience: int = 6
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    decimate: int = 2
    kernel_length: int = 64
    fast_spatial: bool = True       # einsum form of the depthwise spatial convolution (same math)
    autocast_bf16: bool = False     # measured slower than fp32 for 129-channel EEGNet


class BenchmarkConfig(StrictModel):
    """Behavioural decoding benchmark."""

    name: str = "benchmark-v1"
    windows: list[str] = Field(default_factory=lambda: ["prestim", "challenge"])
    targets: list[Literal["rt_s", "log_rt_centred", "lapse"]] = Field(
        default_factory=lambda: ["rt_s", "lapse"])
    models: list[Literal["mean", "demographics", "spectral", "spectral_aperiodic", "riemann",
                         "timecourse", "eegnet", "reve_attnpool", "reve_meanfinal"]] = Field(default_factory=lambda: [
        "mean", "demographics", "spectral", "spectral_aperiodic", "riemann", "eegnet"])
    split_file: str = "cerebralex-split-v1.json"
    min_valid_responses_per_subject: int = 10
    exclude_artifact_windows: bool = True
    artifact_flag: Literal["artifact_window", "high_amplitude_window", "flat_window"] = "artifact_window"
    # index column that defines an excluded window; the S-art sensitivity analysis keeps high-amplitude windows and
    # excludes only flat (dead-signal) windows, whose covariances are singular
    n_bootstrap: int = 2000
    ridge_alphas: list[float] = Field(default_factory=lambda: [
        1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5])
    inner_folds: int = 3
    riemann_pca_components: int | None = 500
    riemann_mean_subsample: int = 4000
    riemann_projection_rank: int | None = 100
    aperiodic_regions: int = 10
    aperiodic_include_fit_r2: bool = False
    timecourse_bin_ms: float = Field(100.0, gt=0)    # timing-preserving baseline
    timecourse_pca_components: int | None = 500
    feature_n_jobs: int = 8
    log_fold_scores: bool = False
    continue_on_model_error: bool = True
    cohort_store: str | None = None          # store whose artifact flags define the cohort
    min_rt_s: float | None = None            # S-antic: drop valid responses faster than this
    exclude_amplitude_outlier_blocks: bool = False   # S-amp: drop amplitude-outlier blocks
    exclude_duplicate_affected_subjects: bool = False  # sensitivity: drop children whose block 3 was
                                                       # distributed twice (deduplicated in the trial table)
    reve_config: str = "foundation_model/reve.yaml"    # frozen REVE probe
    eegnet: EEGNetConfig = Field(default_factory=EEGNetConfig)
    evaluate_holdout_release: bool = True
    subsample_subjects: int | None = None
    seed: int = 12345


class AuditConfig(StrictModel):
    """Parameters of the dataset audit."""

    header_verification_sample_per_task: int = 40
    duration_outlier_iqr_factor: float = 5.0
    trial_count_tasks: list[str] = Field(default_factory=lambda: ["contrastChangeDetection"])
    seed: int = 12345


class FastGuessConfig(StrictModel):
    """Accuracy-by-latency search for the fast-guess cutoff (Ratcliff & Tuerlinckx 2002)."""

    step_s: float = Field(0.05, gt=0)
    max_rt_s: float = Field(1.0, gt=0)
    ci_level: float = Field(0.95, gt=0, lt=1)
    chance: float = Field(0.5, gt=0, lt=1)


class SensitivityConfig(StrictModel):
    """Parameters of the sensitivity analyses."""

    fast_guess: FastGuessConfig = Field(default_factory=FastGuessConfig)


class WindowQCConfig(StrictModel):
    """Window-level artifact rules applied to every extracted window."""

    p2p_threshold_uv: float = Field(300.0, gt=0)
    max_bad_channel_fraction: float = Field(0.10, ge=0, lt=1)
    flat_threshold_uv: float = Field(1.0, ge=0)


class HFSnapshotSpec(StrictModel):
    """A Hugging Face snapshot pinned by revision and verified file checksums."""

    repo: str
    revision: str
    local_dir: str
    sha256: dict[str, str]


class REVEConfig(StrictModel):
    """Frozen REVE probe (foundation_model/reve.yaml)."""

    model: HFSnapshotSpec
    positions: HFSnapshotSpec
    target_sfreq: float = 200.0
    amplitude: Literal["downstream", "zscore"] = "downstream"
    downstream_scale: float = 1.0e4
    zscore_clip_sd: float = Field(15.0, gt=0)
    batch_size: int = Field(64, gt=0)
    unseen_releases: list[str] = Field(default_factory=lambda: ["R4", "NC"])


class StateModelConfig(StrictModel):
    """Latent state models over trial sequences (state_models_and_trait_moderation/state_models.yaml)."""

    name: str = "states-v1"
    windows: list[str] = Field(default_factory=lambda: ["prestim", "challenge"])
    features: Literal["spectral", "reve_attnpool"] = "spectral"
    n_components: int = Field(10, gt=0)
    pca_subsample: int = Field(50000, gt=0)
    hmm_states: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5])
    hmm_restarts: int = Field(5, gt=0)
    lgssm_latents: list[int] = Field(default_factory=lambda: [1, 2, 3, 4])
    inner_validation_fraction: float = Field(0.2, gt=0, lt=1)
    ewma_alpha: float = Field(0.3, gt=0, le=1)
    max_iter: int = Field(100, gt=0)
    targets: list[Literal["rt_s", "lapse"]] = Field(default_factory=lambda: ["rt_s", "lapse"])
    n_bootstrap: int = 2000
    seed: int = 12345
