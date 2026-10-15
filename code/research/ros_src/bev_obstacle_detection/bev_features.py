"""Feature map generation for BEV obstacle experiments."""

import numpy as np

from bev_obstacle_detection.bev_grid import points_to_grid_indices


DEFAULT_LAYER_ORDER = ("occupancy", "max_height", "density", "mean_intensity")


def point_cloud_to_bev_feature_maps(points_xyz, spec, intensities=None, weights=None, min_occupancy_weight=0.0):
    points = np.asarray(points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape Nx3")

    if intensities is not None:
        intensities = np.asarray(intensities, dtype=float)
        if intensities.shape != (points.shape[0],):
            raise ValueError("intensities must have shape N")

    if weights is None:
        weights = np.ones((points.shape[0],), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (points.shape[0],):
            raise ValueError("weights must have shape N")
        if np.any(weights < 0.0):
            raise ValueError("weights must be non-negative")
    if min_occupancy_weight < 0.0:
        raise ValueError("min_occupancy_weight must be non-negative")

    rows, cols, mask = points_to_grid_indices(points, spec)
    shape = (spec.height, spec.width)
    counts = np.zeros(shape, dtype=float)
    weighted_counts = np.zeros(shape, dtype=float)
    max_z = np.full(shape, spec.z_min, dtype=float)
    intensity_sum = np.zeros(shape, dtype=float)

    valid_points = points[mask]
    if valid_points.size:
        np.add.at(counts, (rows, cols), 1.0)
        np.add.at(weighted_counts, (rows, cols), weights[mask])
        np.maximum.at(max_z, (rows, cols), valid_points[:, 2])
        if intensities is not None:
            np.add.at(intensity_sum, (rows, cols), intensities[mask] * weights[mask])

    occupancy = (weighted_counts > float(min_occupancy_weight)).astype(np.float32)
    density = np.minimum(1.0, np.log1p(weighted_counts) / np.log(16.0)).astype(np.float32)
    max_height = np.zeros(shape, dtype=np.float32)
    if np.any(counts > 0.0):
        normalized_height = (max_z - spec.z_min) / (spec.z_max - spec.z_min)
        max_height[counts > 0.0] = np.clip(normalized_height[counts > 0.0], 0.0, 1.0)

    mean_intensity = np.zeros(shape, dtype=np.float32)
    if intensities is not None and np.any(weighted_counts > 0.0):
        mean_intensity[weighted_counts > 0.0] = (
            intensity_sum[weighted_counts > 0.0] / weighted_counts[weighted_counts > 0.0]
        ).astype(np.float32)

    return {
        "occupancy": occupancy,
        "max_height": max_height,
        "density": density,
        "mean_intensity": mean_intensity,
        "point_count": counts.astype(np.float32),
        "weighted_point_count": weighted_counts.astype(np.float32),
    }


def stack_feature_maps(feature_maps, layer_order=DEFAULT_LAYER_ORDER):
    layers = []
    for name in layer_order:
        if name not in feature_maps:
            raise ValueError("missing BEV feature layer: {}".format(name))
        layers.append(np.asarray(feature_maps[name], dtype=np.float32))

    if not layers:
        raise ValueError("layer_order must be non-empty")

    shape = layers[0].shape
    for layer in layers:
        if layer.shape != shape:
            raise ValueError("all BEV feature layers must have the same shape")

    return np.stack(layers, axis=0)
