"""Adapters for synchronized external 3D detector candidates."""

import math

import numpy as np

from bev_obstacle_detection.bev_grid import metric_xy_to_grid_rc
from bev_obstacle_detection.obstacle_detection import BevObstacle
from bev_obstacle_detection.online_candidate_gate import oriented_iou


def _grid_bounds(center_x, center_y, size_x, size_y, yaw, spec):
    half_x = max(float(size_x), spec.resolution) * 0.5
    half_y = max(float(size_y), spec.resolution) * 0.5
    cos_yaw = abs(math.cos(float(yaw)))
    sin_yaw = abs(math.sin(float(yaw)))
    extent_x = cos_yaw * half_x + sin_yaw * half_y
    extent_y = sin_yaw * half_x + cos_yaw * half_y
    corners = np.asarray(
        [
            [float(center_x) - extent_x, float(center_y) - extent_y],
            [float(center_x) + extent_x, float(center_y) + extent_y],
        ],
        dtype=float,
    )
    rows, cols = metric_xy_to_grid_rc(corners, spec)
    return (
        int(min(rows)),
        int(max(rows)),
        int(min(cols)),
        int(max(cols)),
    )


def detection_message_to_obstacles(message, spec, minimum_score=0.0):
    """Convert a Detection3DArray-like object while preserving oriented boxes."""
    obstacles = []
    threshold = float(minimum_score)
    for index, detection in enumerate(message.detections):
        score = float(detection.score)
        if score < threshold:
            continue
        row_min, row_max, col_min, col_max = _grid_bounds(
            detection.center_x,
            detection.center_y,
            detection.size_x,
            detection.size_y,
            detection.yaw,
            spec,
        )
        clipped_rows = max(
            0,
            min(spec.height - 1, row_max)
            - max(0, row_min)
            + 1,
        )
        clipped_cols = max(
            0,
            min(spec.width - 1, col_max)
            - max(0, col_min)
            + 1,
        )
        obstacles.append(
            BevObstacle(
                obstacle_id=int(detection.obstacle_id or index + 1),
                center_x=float(detection.center_x),
                center_y=float(detection.center_y),
                size_x=float(detection.size_x),
                size_y=float(detection.size_y),
                cell_count=int(clipped_rows * clipped_cols),
                score=score,
                row_min=row_min,
                row_max=row_max,
                col_min=col_min,
                col_max=col_max,
                center_z=float(detection.center_z),
                size_z=float(detection.size_z),
                yaw=float(detection.yaw),
                class_id=int(detection.class_id),
                class_name=str(detection.class_name),
                source=str(detection.source),
                sample_token=str(detection.sample_token),
            )
        )
    return tuple(obstacles)


def independent_candidate_kind(obstacle):
    source = str(getattr(obstacle, "source", "") or "").strip().lower()
    if "only" not in source:
        return ""
    if "camera" in source:
        return "camera"
    if "radar" in source:
        return "radar"
    return ""


def prefilter_independent_candidates(
    obstacles,
    topk_per_modality=0,
    nms_iou=0.30,
):
    """Limit independent experts before expensive exact evidence extraction."""
    items = tuple(obstacles)
    limit = int(topk_per_modality)
    threshold = float(nms_iou)
    if limit < 0:
        raise ValueError("independent candidate top-k must be non-negative")
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("independent candidate NMS IoU must be in [0, 1]")

    grouped = {"camera": [], "radar": []}
    passthrough = []
    for index, obstacle in enumerate(items):
        kind = independent_candidate_kind(obstacle)
        if kind:
            grouped[kind].append((index, obstacle))
        else:
            passthrough.append((index, obstacle))

    selected = list(passthrough)
    independent_input = sum(len(group) for group in grouped.values())
    if limit <= 0:
        selected.extend(item for group in grouped.values() for item in group)
    else:
        for group in grouped.values():
            ranked = sorted(
                group,
                key=lambda item: (
                    -float(item[1].score),
                    int(item[1].obstacle_id),
                    item[0],
                ),
            )
            kept = []
            for item in ranked:
                if any(
                    oriented_iou(item[1], previous[1]) >= threshold
                    for previous in kept
                ):
                    continue
                kept.append(item)
                if len(kept) >= limit:
                    break
            selected.extend(kept)

    selected.sort(key=lambda item: item[0])
    kept_obstacles = tuple(item[1] for item in selected)
    independent_kept = len(kept_obstacles) - len(passthrough)
    return (
        kept_obstacles,
        {
            "input_count": len(items),
            "kept_count": len(kept_obstacles),
            "dropped_count": len(items) - len(kept_obstacles),
            "independent_input_count": independent_input,
            "independent_kept_count": independent_kept,
            "independent_dropped_count": (
                independent_input - independent_kept
            ),
        },
    )


def candidate_occupancy_from_obstacles(obstacles, spec):
    """Rasterize candidate bounds for local support measurement."""
    occupancy = np.zeros((spec.height, spec.width), dtype=np.float32)
    for obstacle in obstacles:
        row_min = max(0, int(obstacle.row_min))
        row_max = min(spec.height - 1, int(obstacle.row_max))
        col_min = max(0, int(obstacle.col_min))
        col_max = min(spec.width - 1, int(obstacle.col_max))
        if row_max < row_min or col_max < col_min:
            continue
        occupancy[row_min : row_max + 1, col_min : col_max + 1] = 1.0
    return occupancy


def points_in_oriented_obstacle(
    points_xyz,
    obstacle,
    use_height=False,
    vertical_margin=0.5,
):
    """Return a mask for points inside one oriented 3D/BEV candidate."""
    points = np.asarray(points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("points_xyz must have shape Nx2 or Nx3")
    if points.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    dx = points[:, 0] - float(obstacle.center_x)
    dy = points[:, 1] - float(obstacle.center_y)
    cos_yaw = math.cos(float(obstacle.yaw))
    sin_yaw = math.sin(float(obstacle.yaw))
    local_x = cos_yaw * dx + sin_yaw * dy
    local_y = -sin_yaw * dx + cos_yaw * dy
    inside = (
        (np.abs(local_x) <= max(float(obstacle.size_x), 0.0) * 0.5)
        & (np.abs(local_y) <= max(float(obstacle.size_y), 0.0) * 0.5)
    )
    if use_height and points.shape[1] >= 3 and obstacle.size_z > 0.0:
        half_z = float(obstacle.size_z) * 0.5 + max(
            0.0,
            float(vertical_margin),
        )
        inside &= np.abs(points[:, 2] - float(obstacle.center_z)) <= half_z
    return inside


def oriented_footprint_support_ratio(obstacle, support_map, spec):
    """Measure support only in cells whose centers fall inside the yawed box."""
    support = np.asarray(support_map) > 0
    if support.shape != (spec.height, spec.width):
        raise ValueError("support_map shape does not match BEV grid")
    row_min = max(0, int(obstacle.row_min))
    row_max = min(spec.height - 1, int(obstacle.row_max))
    col_min = max(0, int(obstacle.col_min))
    col_max = min(spec.width - 1, int(obstacle.col_max))
    if row_max < row_min or col_max < col_min:
        return 0.0
    rows, cols = np.mgrid[
        row_min : row_max + 1,
        col_min : col_max + 1,
    ]
    x = spec.x_min + (cols.astype(float) + 0.5) * spec.resolution
    y = spec.y_max - (rows.astype(float) + 0.5) * spec.resolution
    dx = x - float(obstacle.center_x)
    dy = y - float(obstacle.center_y)
    cos_yaw = math.cos(float(obstacle.yaw))
    sin_yaw = math.sin(float(obstacle.yaw))
    local_x = cos_yaw * dx + sin_yaw * dy
    local_y = -sin_yaw * dx + cos_yaw * dy
    footprint = (
        (np.abs(local_x) <= max(float(obstacle.size_x), 0.0) * 0.5)
        & (np.abs(local_y) <= max(float(obstacle.size_y), 0.0) * 0.5)
    )
    footprint_cells = int(np.count_nonzero(footprint))
    if footprint_cells <= 0:
        return 0.0
    crop = support[row_min : row_max + 1, col_min : col_max + 1]
    return float(np.count_nonzero(crop & footprint)) / float(
        footprint_cells
    )


def exact_candidate_evidence(
    obstacles,
    lidar_points,
    radar_points,
    camera_support_map,
    spec,
    radar_intensities=None,
    lidar_point_saturation=8,
    radar_point_saturation=2,
    camera_coverage_reference=0.05,
    lidar_vertical_margin=0.5,
):
    """Build normalized local evidence and raw diagnostics per candidate."""
    lidar = np.asarray(lidar_points, dtype=float)
    radar = np.asarray(radar_points, dtype=float)
    if lidar.ndim != 2 or lidar.shape[1] < 2:
        raise ValueError("lidar_points must have shape Nx2 or Nx3")
    if radar.ndim != 2 or radar.shape[1] < 2:
        raise ValueError("radar_points must have shape Nx2 or Nx3")
    if radar_intensities is None:
        radar_strength = np.zeros(radar.shape[0], dtype=float)
    else:
        radar_strength = np.asarray(radar_intensities, dtype=float).reshape(-1)
        if radar_strength.shape[0] != radar.shape[0]:
            raise ValueError(
                "radar_intensities must align with radar_points"
            )
        radar_strength = np.nan_to_num(
            radar_strength,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
    radar_frame_median = (
        float(np.median(radar_strength))
        if radar_strength.size
        else 0.0
    )
    lidar_scale = max(1, int(lidar_point_saturation))
    radar_scale = max(1, int(radar_point_saturation))
    camera_scale = max(float(camera_coverage_reference), 1.0e-6)
    camera_map = (
        np.asarray(camera_support_map)
        if camera_support_map is not None
        else np.zeros((spec.height, spec.width), dtype=np.float32)
    )
    ratios = {}
    diagnostics = {}
    for obstacle in obstacles:
        lidar_count = int(
            np.count_nonzero(
                points_in_oriented_obstacle(
                    lidar,
                    obstacle,
                    use_height=True,
                    vertical_margin=lidar_vertical_margin,
                )
            )
        )
        radar_mask = points_in_oriented_obstacle(
            radar,
            obstacle,
            use_height=False,
        )
        radar_count = int(np.count_nonzero(radar_mask))
        local_radar_strength = radar_strength[radar_mask]
        radar_intensity_mean = (
            float(np.mean(local_radar_strength))
            if local_radar_strength.size
            else 0.0
        )
        radar_intensity_max = (
            float(np.max(local_radar_strength))
            if local_radar_strength.size
            else 0.0
        )
        radar_intensity_contrast = (
            radar_intensity_mean - radar_frame_median
            if local_radar_strength.size
            else 0.0
        )
        camera_coverage = oriented_footprint_support_ratio(
            obstacle,
            camera_map,
            spec,
        )
        lidar_support = min(1.0, lidar_count / float(lidar_scale))
        radar_support = min(1.0, radar_count / float(radar_scale))
        radar_intensity_support = radar_support * float(
            np.clip(
                (radar_intensity_mean - 0.30) / 0.70,
                0.0,
                1.0,
            )
        )
        camera_support = min(1.0, camera_coverage / camera_scale)
        obstacle_id = int(obstacle.obstacle_id)
        ratios[obstacle_id] = {
            "lidar": lidar_support,
            "camera": camera_support,
            "radar": radar_support,
        }
        diagnostics[obstacle_id] = {
            "lidar_point_count": lidar_count,
            "lidar_point_support": lidar_support,
            "radar_point_count": radar_count,
            "radar_point_support": radar_support,
            "radar_intensity_mean": radar_intensity_mean,
            "radar_intensity_max": radar_intensity_max,
            "radar_intensity_contrast": radar_intensity_contrast,
            "radar_intensity_support": radar_intensity_support,
            "camera_oriented_coverage": camera_coverage,
            "camera_oriented_support": camera_support,
        }
    return ratios, diagnostics
