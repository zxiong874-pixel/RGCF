"""Monocular BEV candidates from semantic-mask component geometry."""

import math

import numpy as np

from bev_obstacle_detection.bev_grid import metric_xy_to_grid_rc
from bev_obstacle_detection.obstacle_detection import BevObstacle


def _grid_bounds(center_x, center_y, size_x, size_y, spec):
    corners = np.asarray(
        [
            [center_x - 0.5 * size_x, center_y - 0.5 * size_y],
            [center_x + 0.5 * size_x, center_y + 0.5 * size_y],
        ],
        dtype=float,
    )
    rows, cols = metric_xy_to_grid_rc(corners, spec)
    return int(min(rows)), int(max(rows)), int(min(cols)), int(max(cols))


def camera_semantic_candidates(
    mask,
    spec,
    horizontal_fov_deg=90.0,
    vertical_fov_deg=65.0,
    assumed_height_m=1.5,
    min_component_area_px=32,
    max_component_fraction=0.20,
    reject_border_components=True,
    min_pixel_height=8,
    min_range=1.0,
    max_range=12.0,
    radial_size=1.2,
    min_lateral_size=0.4,
    max_lateral_size=1.8,
    lateral_padding=0.15,
    score_saturation_area_px=4096,
):
    """Estimate class-agnostic BEV boxes from mask apparent size and bearing."""
    import cv2

    image = np.asarray(mask)
    if image.ndim != 2:
        raise ValueError("mask must be 2D")
    if not 0.0 < horizontal_fov_deg < 180.0:
        raise ValueError("horizontal_fov_deg must be in (0, 180)")
    if not 0.0 < vertical_fov_deg < 180.0:
        raise ValueError("vertical_fov_deg must be in (0, 180)")
    if assumed_height_m <= 0.0 or min_component_area_px <= 0:
        raise ValueError("height and component area must be positive")
    if not 0.0 < max_component_fraction <= 1.0:
        raise ValueError("max_component_fraction must be in (0, 1]")
    if min_pixel_height <= 0 or min_range < 0.0 or max_range <= min_range:
        raise ValueError("invalid pixel-height or range limits")
    if radial_size <= 0.0 or min_lateral_size <= 0.0:
        raise ValueError("candidate sizes must be positive")
    if max_lateral_size < min_lateral_size or lateral_padding < 0.0:
        raise ValueError("invalid lateral size limits")
    if score_saturation_area_px <= 0:
        raise ValueError("score_saturation_area_px must be positive")
    if image.size == 0:
        return ()

    binary = (image > 0).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    height, width = binary.shape
    fx = width / (
        2.0 * math.tan(math.radians(float(horizontal_fov_deg)) * 0.5)
    )
    fy = height / (
        2.0 * math.tan(math.radians(float(vertical_fov_deg)) * 0.5)
    )
    cx = 0.5 * (width - 1.0)
    max_area = float(max_component_fraction) * float(binary.size)
    candidates = []
    for label in range(1, count):
        x, y, component_width, component_height, area = stats[label]
        touches_border = (
            x <= 0
            or y <= 0
            or x + component_width >= width
            or y + component_height >= height
        )
        if (
            area < int(min_component_area_px)
            or area > max_area
            or component_height < int(min_pixel_height)
            or (reject_border_components and touches_border)
        ):
            continue

        distance = fy * float(assumed_height_m) / float(component_height)
        if not min_range <= distance <= max_range:
            continue
        center_u = float(centroids[label, 0])
        lateral = -(center_u - cx) * distance / fx
        lateral_size = (
            float(component_width) * distance / fx + 2.0 * float(lateral_padding)
        )
        lateral_size = float(
            np.clip(lateral_size, min_lateral_size, max_lateral_size)
        )
        if not (
            spec.x_min <= distance < spec.x_max
            and spec.y_min <= lateral < spec.y_max
        ):
            continue
        row_min, row_max, col_min, col_max = _grid_bounds(
            distance,
            lateral,
            radial_size,
            lateral_size,
            spec,
        )
        candidates.append(
            BevObstacle(
                obstacle_id=len(candidates) + 1,
                center_x=float(distance),
                center_y=float(lateral),
                size_x=float(radial_size),
                size_y=lateral_size,
                cell_count=int(area),
                score=float(np.clip(area / float(score_saturation_area_px), 0.0, 1.0)),
                row_min=row_min,
                row_max=row_max,
                col_min=col_min,
                col_max=col_max,
                size_z=float(assumed_height_m),
                source="camera_semantic",
            )
        )
    return tuple(candidates)

