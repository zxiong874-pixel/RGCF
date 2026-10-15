"""Point cloud filters used before BEV feature generation."""

import numpy as np


def adaptive_ground_height(points_xyz, quantile=0.05, min_points=50):
    points = np.asarray(points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points_xyz must have shape Nx3")
    if points.shape[0] < int(min_points):
        return None
    if quantile < 0.0 or quantile > 1.0:
        raise ValueError("quantile must be in [0, 1]")
    z_values = points[:, 2]
    z_values = z_values[np.isfinite(z_values)]
    if z_values.shape[0] < int(min_points):
        return None
    return float(np.quantile(z_values, quantile))


def filter_points_by_height(
    points_xyz,
    min_point_z=None,
    max_point_z=None,
    use_adaptive_ground_filter=False,
    adaptive_ground_quantile=0.05,
    adaptive_ground_clearance=0.12,
    adaptive_ground_min_points=50,
):
    points = np.asarray(points_xyz)
    if points.size == 0:
        return points
    effective_min_z = min_point_z
    if use_adaptive_ground_filter:
        ground_z = adaptive_ground_height(
            points,
            quantile=adaptive_ground_quantile,
            min_points=adaptive_ground_min_points,
        )
        if ground_z is not None:
            adaptive_min_z = ground_z + float(adaptive_ground_clearance)
            effective_min_z = adaptive_min_z if effective_min_z is None else max(float(effective_min_z), adaptive_min_z)

    if effective_min_z is None and max_point_z is None:
        return points

    mask = np.ones((points.shape[0],), dtype=bool)
    if effective_min_z is not None:
        mask &= points[:, 2] >= float(effective_min_z)
    if max_point_z is not None:
        mask &= points[:, 2] <= float(max_point_z)
    return points[mask]


def filter_points_by_lateral_range_median(
    points_xyz,
    lateral_resolution=0.2,
    max_range_deviation=0.25,
    min_points_per_bin=3,
):
    points = np.asarray(points_xyz)
    if points.size == 0:
        return points
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape Nx3")
    if lateral_resolution <= 0.0:
        raise ValueError("lateral_resolution must be positive")
    if max_range_deviation < 0.0:
        raise ValueError("max_range_deviation must be non-negative")
    if min_points_per_bin <= 1:
        return points

    y_bins = np.floor(points[:, 1] / float(lateral_resolution)).astype(np.int64)
    z_bins = np.floor(points[:, 2] / float(lateral_resolution)).astype(np.int64)
    bin_keys = np.stack((y_bins, z_bins), axis=1)
    unique_keys, inverse, counts = np.unique(bin_keys, axis=0, return_inverse=True, return_counts=True)
    del unique_keys

    keep = np.ones((points.shape[0],), dtype=bool)
    for bin_index, count in enumerate(counts):
        if count < int(min_points_per_bin):
            continue
        mask = inverse == bin_index
        median_x = float(np.median(points[mask, 0]))
        keep[mask] &= np.abs(points[mask, 0] - median_x) <= float(max_range_deviation)
    return points[keep]


def lateral_range_deviation_score(
    points_xyz,
    lateral_resolution=0.2,
    min_points_per_bin=5,
    score_quantile=0.5,
):
    points = np.asarray(points_xyz)
    if points.size == 0:
        return None
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape Nx3")
    if lateral_resolution <= 0.0:
        raise ValueError("lateral_resolution must be positive")
    if min_points_per_bin <= 1:
        raise ValueError("min_points_per_bin must be greater than 1")
    if score_quantile < 0.0 or score_quantile > 1.0:
        raise ValueError("score_quantile must be in [0, 1]")

    y_bins = np.floor(points[:, 1] / float(lateral_resolution)).astype(np.int64)
    z_bins = np.floor(points[:, 2] / float(lateral_resolution)).astype(np.int64)
    bin_keys = np.stack((y_bins, z_bins), axis=1)
    unique_keys, inverse, counts = np.unique(bin_keys, axis=0, return_inverse=True, return_counts=True)
    del unique_keys

    order = np.argsort(inverse, kind="stable")
    sorted_x = points[order, 0]
    offsets = np.concatenate(([0], np.cumsum(counts)))
    deviations = []
    for bin_index, count in enumerate(counts):
        if count < int(min_points_per_bin):
            continue
        start = int(offsets[bin_index])
        end = int(offsets[bin_index + 1])
        values = sorted_x[start:end]
        median_x = float(np.median(values))
        deviations.append(np.abs(values - median_x))
    if not deviations:
        return None
    return float(np.quantile(np.concatenate(deviations), float(score_quantile)))
