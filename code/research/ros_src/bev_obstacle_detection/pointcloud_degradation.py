"""Deterministic point cloud degradation helpers for robustness tests."""

import numpy as np


def validate_degradation_params(keep_ratio=1.0, frame_drop_rate=0.0, xyz_noise_std=0.0, z_noise_std=0.0):
    if keep_ratio < 0.0 or keep_ratio > 1.0:
        raise ValueError("keep_ratio must be in [0, 1]")
    if frame_drop_rate < 0.0 or frame_drop_rate > 1.0:
        raise ValueError("frame_drop_rate must be in [0, 1]")
    if xyz_noise_std < 0.0:
        raise ValueError("xyz_noise_std must be non-negative")
    if z_noise_std < 0.0:
        raise ValueError("z_noise_std must be non-negative")


def degrade_xyz_points(
    points_xyz,
    rng,
    keep_ratio=1.0,
    xyz_noise_std=0.0,
    z_noise_std=0.0,
    frame_drop_rate=0.0,
):
    validate_degradation_params(
        keep_ratio=keep_ratio,
        frame_drop_rate=frame_drop_rate,
        xyz_noise_std=xyz_noise_std,
        z_noise_std=z_noise_std,
    )
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape Nx3")
    if points.size == 0:
        return points.copy()

    if frame_drop_rate > 0.0 and rng.random_sample() < frame_drop_rate:
        return points[:0].copy()

    degraded = points.copy()
    if keep_ratio < 1.0:
        if keep_ratio <= 0.0:
            degraded = degraded[:0]
        else:
            keep_mask = rng.random_sample(degraded.shape[0]) < keep_ratio
            degraded = degraded[keep_mask]

    if degraded.size == 0:
        return degraded.reshape((0, 3)).astype(np.float32, copy=False)

    if xyz_noise_std > 0.0:
        degraded += rng.normal(0.0, xyz_noise_std, size=degraded.shape).astype(np.float32)
    if z_noise_std > 0.0:
        degraded[:, 2] += rng.normal(0.0, z_noise_std, size=degraded.shape[0]).astype(np.float32)
    return degraded.astype(np.float32, copy=False)


def degradation_label(keep_ratio=1.0, xyz_noise_std=0.0, z_noise_std=0.0, frame_drop_rate=0.0):
    parts = []
    if keep_ratio < 1.0:
        parts.append("keep_{:.2f}".format(keep_ratio).replace(".", "p"))
    if xyz_noise_std > 0.0:
        parts.append("xyz_noise_{:.3f}".format(xyz_noise_std).replace(".", "p"))
    if z_noise_std > 0.0:
        parts.append("z_noise_{:.3f}".format(z_noise_std).replace(".", "p"))
    if frame_drop_rate > 0.0:
        parts.append("frame_drop_{:.2f}".format(frame_drop_rate).replace(".", "p"))
    return "_".join(parts) if parts else "clean"
