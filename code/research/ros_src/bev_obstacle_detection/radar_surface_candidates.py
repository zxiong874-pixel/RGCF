"""Class-agnostic obstacle candidates from sparse Radar surface returns."""

import numpy as np

from bev_obstacle_detection.bev_grid import metric_xy_to_grid_rc
from bev_obstacle_detection.obstacle_detection import BevObstacle


def _euclidean_components(points_xy, radius):
    count = points_xy.shape[0]
    if count == 0:
        return ()
    distance_sq = np.sum(
        (points_xy[:, None, :] - points_xy[None, :, :]) ** 2,
        axis=2,
    )
    visited = np.zeros(count, dtype=bool)
    components = []
    for start in range(count):
        if visited[start]:
            continue
        queue = [start]
        visited[start] = True
        component = []
        while queue:
            index = queue.pop()
            component.append(index)
            neighbors = np.flatnonzero(
                (distance_sq[index] <= radius * radius) & ~visited
            )
            visited[neighbors] = True
            queue.extend(neighbors.tolist())
        components.append(np.asarray(component, dtype=int))
    return tuple(components)


def _grid_bounds(center_x, center_y, size_x, size_y, spec):
    corners = np.asarray(
        [
            [center_x - size_x * 0.5, center_y - size_y * 0.5],
            [center_x + size_x * 0.5, center_y + size_y * 0.5],
        ],
        dtype=float,
    )
    rows, cols = metric_xy_to_grid_rc(corners, spec)
    return int(min(rows)), int(max(rows)), int(min(cols)), int(max(cols))


def radar_surface_candidates(
    points_xyz,
    spec,
    cluster_radius=0.65,
    min_points=2,
    radial_extent=0.8,
    padding=0.15,
    max_surface_span=1.5,
    max_range=10.0,
    score_saturation_points=6,
):
    """Complete front-surface Radar clusters into conservative BEV boxes."""
    points = np.asarray(points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape Nx3")
    if cluster_radius <= 0.0 or min_points <= 0:
        raise ValueError("cluster radius and min_points must be positive")
    if radial_extent < 0.0 or padding < 0.0:
        raise ValueError("radial extent and padding must be non-negative")
    if max_surface_span <= 0.0 or max_range <= 0.0:
        raise ValueError("surface span and max range must be positive")
    if score_saturation_points <= 0:
        raise ValueError("score_saturation_points must be positive")

    finite = np.all(np.isfinite(points[:, :2]), axis=1)
    points = points[finite]
    candidates = []
    for component_indices in _euclidean_components(points[:, :2], cluster_radius):
        component = points[component_indices]
        if component.shape[0] < int(min_points):
            continue
        surface_span = np.ptp(component[:, :2], axis=0)
        if float(np.max(surface_span)) > float(max_surface_span):
            continue
        ranges = np.linalg.norm(component[:, :2], axis=1)
        if float(np.mean(ranges)) > float(max_range):
            continue

        radial_directions = component[:, :2] / np.maximum(ranges[:, None], 1.0e-6)
        completed_xy = np.vstack(
            (
                component[:, :2],
                component[:, :2] + radial_directions * float(radial_extent),
            )
        )
        lower = completed_xy.min(axis=0) - float(padding)
        upper = completed_xy.max(axis=0) + float(padding)
        center = 0.5 * (lower + upper)
        size = np.maximum(upper - lower, spec.resolution)
        row_min, row_max, col_min, col_max = _grid_bounds(
            center[0], center[1], size[0], size[1], spec
        )
        candidates.append(
            BevObstacle(
                obstacle_id=len(candidates) + 1,
                center_x=float(center[0]),
                center_y=float(center[1]),
                size_x=float(size[0]),
                size_y=float(size[1]),
                cell_count=int(component.shape[0]),
                score=float(
                    np.clip(
                        component.shape[0] / float(score_saturation_points),
                        0.0,
                        1.0,
                    )
                ),
                row_min=row_min,
                row_max=row_max,
                col_min=col_min,
                col_max=col_max,
                source="radar_surface",
            )
        )
    return tuple(candidates)
