"""Motion-consistency quality for VINS poses used by temporal BEV fusion."""

import math

import numpy as np


def _linear_quality(value, clean_limit, rejected_limit):
    value = float(value)
    clean_limit = float(clean_limit)
    rejected_limit = float(rejected_limit)
    if clean_limit < 0.0 or rejected_limit <= clean_limit:
        raise ValueError("limits must satisfy 0 <= clean_limit < rejected_limit")
    if not math.isfinite(value) or value >= rejected_limit:
        return 0.0
    if value <= clean_limit:
        return 1.0
    return (rejected_limit - value) / (rejected_limit - clean_limit)


def relative_rotation_angle(previous_rotation, current_rotation):
    previous = np.asarray(previous_rotation, dtype=float)
    current = np.asarray(current_rotation, dtype=float)
    if previous.shape != (3, 3) or current.shape != (3, 3):
        raise ValueError("rotations must be 3x3")
    relative = previous.T @ current
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(math.acos(cosine))


def vins_motion_quality(
    previous_transform,
    current_transform,
    delta_time_sec,
    clean_speed_mps=10.0,
    rejected_speed_mps=20.0,
    clean_angular_rate_rps=0.8,
    rejected_angular_rate_rps=1.5,
):
    """Return quality in [0, 1] from inter-pose speed and angular rate."""
    previous = np.asarray(previous_transform, dtype=float)
    current = np.asarray(current_transform, dtype=float)
    if previous.shape != (4, 4) or current.shape != (4, 4):
        raise ValueError("transforms must be 4x4")
    delta_time_sec = float(delta_time_sec)
    if not math.isfinite(delta_time_sec) or delta_time_sec <= 0.0:
        return 0.0

    speed = np.linalg.norm(current[:3, 3] - previous[:3, 3]) / delta_time_sec
    angular_rate = (
        relative_rotation_angle(previous[:3, :3], current[:3, :3])
        / delta_time_sec
    )
    speed_quality = _linear_quality(
        speed, clean_speed_mps, rejected_speed_mps
    )
    angular_quality = _linear_quality(
        angular_rate, clean_angular_rate_rps, rejected_angular_rate_rps
    )
    return float(min(speed_quality, angular_quality))
