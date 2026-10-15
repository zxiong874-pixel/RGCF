"""Timestamp-alignment quality for asynchronous multimodal fusion."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyncQualityConfig:
    clean_offset_sec: float = 0.02
    rejected_offset_sec: float = 0.08
    minimum_quality: float = 0.0

    def validate(self):
        if self.clean_offset_sec < 0.0:
            raise ValueError("clean_offset_sec must be non-negative")
        if self.rejected_offset_sec <= self.clean_offset_sec:
            raise ValueError(
                "rejected_offset_sec must be greater than clean_offset_sec"
            )
        if self.minimum_quality < 0.0 or self.minimum_quality > 1.0:
            raise ValueError("minimum_quality must be in [0, 1]")


def timestamp_alignment_quality(offset_sec, config=None):
    """Map an absolute timestamp offset to a bounded synchronization quality."""
    config = config or SyncQualityConfig()
    config.validate()
    if offset_sec is None:
        return 0.0
    offset = abs(float(offset_sec))
    if not np.isfinite(offset):
        return 0.0
    if offset <= config.clean_offset_sec:
        return 1.0
    if offset >= config.rejected_offset_sec:
        return float(config.minimum_quality)
    fraction = (offset - config.clean_offset_sec) / (
        config.rejected_offset_sec - config.clean_offset_sec
    )
    return float(
        np.clip(
            1.0 - fraction * (1.0 - config.minimum_quality),
            config.minimum_quality,
            1.0,
        )
    )
