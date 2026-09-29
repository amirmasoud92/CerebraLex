"""Registry of HBN-EEG releases with licensing and provenance metadata.

OpenNeuro accession numbers for R1 to R9 are taken from the reference list of the
HBN-EEG preprint (Shirazi et al. 2024, refs 26 to 34). R10 and R11 were not part of
that dissemination and their accession numbers are recorded as unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

S3_PREFIX = "s3://fcp-indi/data/Projects/HBN/BIDS_EEG"
CHALLENGE_VALIDATION_RELEASE = "R5"
CHALLENGE_TEST_RELEASE = "R12"


class DataLicense(StrEnum):
    CC_BY_SA_4 = "CC-BY-SA-4.0"
    CC_BY_NC_SA_4 = "CC-BY-NC-SA-4.0"


@dataclass(frozen=True)
class HBNRelease:
    release_id: str
    license: DataLicense
    s3_uri: str
    openneuro_id: str | None

    @property
    def commercial_use_allowed(self) -> bool:
        return self.license is DataLicense.CC_BY_SA_4

    @property
    def is_public_openneuro(self) -> bool:
        return self.openneuro_id is not None


_OPENNEURO_IDS: dict[str, str | None] = {
    "R1": "ds005505", "R2": "ds005506", "R3": "ds005507", "R4": "ds005508",
    "R5": "ds005509", "R6": "ds005510", "R7": "ds005511", "R8": "ds005512",
    "R9": "ds005514", "R10": None, "R11": None,
}


def _build_registry() -> dict[str, HBNRelease]:
    registry = {
        rid: HBNRelease(rid, DataLicense.CC_BY_SA_4, f"{S3_PREFIX}/cmi_bids_{rid}", accession)
        for rid, accession in _OPENNEURO_IDS.items()
    }
    registry["NC"] = HBNRelease("NC", DataLicense.CC_BY_NC_SA_4, f"{S3_PREFIX}/cmi_bids_NC", None)
    return registry


RELEASE_REGISTRY: dict[str, HBNRelease] = _build_registry()


def get_release(release_id: str) -> HBNRelease:
    try:
        return RELEASE_REGISTRY[release_id]
    except KeyError as exc:
        raise KeyError(
            f"unknown HBN release '{release_id}'; known: {sorted(RELEASE_REGISTRY)}"
        ) from exc
