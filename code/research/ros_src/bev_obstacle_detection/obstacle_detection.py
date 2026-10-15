"""Simple BEV occupancy based obstacle extraction."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from bev_obstacle_detection.bev_grid import grid_rc_to_metric_xy_centers


@dataclass(frozen=True)
class BevObstacle:
    obstacle_id: int
    center_x: float
    center_y: float
    size_x: float
    size_y: float
    cell_count: int
    score: float
    row_min: int
    row_max: int
    col_min: int
    col_max: int
    center_z: float = 0.0
    size_z: float = 0.5
    yaw: float = 0.0
    class_id: int = -1
    class_name: str = ""
    source: str = ""
    sample_token: str = ""


def _renumber_obstacle(obstacle, obstacle_id):
    return BevObstacle(
        obstacle_id=int(obstacle_id),
        center_x=obstacle.center_x,
        center_y=obstacle.center_y,
        size_x=obstacle.size_x,
        size_y=obstacle.size_y,
        cell_count=obstacle.cell_count,
        score=obstacle.score,
        row_min=obstacle.row_min,
        row_max=obstacle.row_max,
        col_min=obstacle.col_min,
        col_max=obstacle.col_max,
        center_z=obstacle.center_z,
        size_z=obstacle.size_z,
        yaw=obstacle.yaw,
        class_id=obstacle.class_id,
        class_name=obstacle.class_name,
        source=obstacle.source,
        sample_token=obstacle.sample_token,
    )


def obstacle_iou(a, b):
    inter_row_min = max(a.row_min, b.row_min)
    inter_row_max = min(a.row_max, b.row_max)
    inter_col_min = max(a.col_min, b.col_min)
    inter_col_max = min(a.col_max, b.col_max)
    inter_rows = max(0, inter_row_max - inter_row_min + 1)
    inter_cols = max(0, inter_col_max - inter_col_min + 1)
    intersection = inter_rows * inter_cols
    if intersection <= 0:
        return 0.0
    area_a = max(0, a.row_max - a.row_min + 1) * max(0, a.col_max - a.col_min + 1)
    area_b = max(0, b.row_max - b.row_min + 1) * max(0, b.col_max - b.col_min + 1)
    union = area_a + area_b - intersection
    return float(intersection) / float(union) if union > 0 else 0.0


def merge_obstacle_candidates(primary_obstacles, supplemental_obstacles, iou_threshold=0.3):
    if iou_threshold < 0.0 or iou_threshold > 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    merged = []
    for obstacle in primary_obstacles:
        merged.append(_renumber_obstacle(obstacle, len(merged) + 1))
    for candidate in supplemental_obstacles:
        if any(obstacle_iou(candidate, existing) >= iou_threshold for existing in merged):
            continue
        merged.append(_renumber_obstacle(candidate, len(merged) + 1))
    return tuple(merged)


def parse_distance_value_rules(text, value_name="value", value_type=float):
    rules = []
    if text in ("", None):
        return tuple()
    for item in str(text).split(","):
        stripped = item.strip()
        if not stripped:
            continue
        if ":" not in stripped or "-" not in stripped.split(":", 1)[0]:
            raise ValueError("distance rule must look like '0-3:3'")
        distance_part, value_part = stripped.split(":", 1)
        min_text, max_text = distance_part.split("-", 1)
        distance_min = float(min_text)
        distance_max = float("inf") if max_text.strip().lower() in ("inf", "infinity") else float(max_text)
        value = value_type(float(value_part))
        if distance_min < 0.0 or distance_max <= distance_min:
            raise ValueError("distance rule must have 0 <= min < max")
        if value <= 0:
            raise ValueError("distance rule {} must be positive".format(value_name))
        rules.append((distance_min, distance_max, value))
    return tuple(rules)


def parse_distance_min_cells_rules(text):
    return parse_distance_value_rules(text, value_name="cells", value_type=int)


def min_cells_for_distance(distance_m, default_min_cells, rules=()):
    return int(value_for_distance(distance_m, default_min_cells, rules))


def value_for_distance(distance_m, default_value, rules=()):
    for distance_min, distance_max, cells in rules or ():
        if distance_min <= distance_m < distance_max:
            return cells
    return default_value


def _validate_optional_positive(name, value):
    if value is not None and float(value) <= 0.0:
        raise ValueError("{} must be positive".format(name))


def _neighbors(row, col, height, width, connectivity):
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    for d_row, d_col in offsets:
        n_row = row + d_row
        n_col = col + d_col
        if 0 <= n_row < height and 0 <= n_col < width:
            yield n_row, n_col


def connected_occupied_components(occupancy, min_cells=1, connectivity=8):
    occupied = np.asarray(occupancy) > 0
    if occupied.ndim != 2:
        raise ValueError("occupancy must be a 2D array")
    if min_cells <= 0:
        raise ValueError("min_cells must be positive")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    height, width = occupied.shape
    try:
        import cv2

        count, labels = cv2.connectedComponents(
            occupied.astype(np.uint8),
            connectivity=connectivity,
        )
        if count <= 1:
            return []
        flat_labels = labels.ravel()
        flat_indices = np.flatnonzero(flat_labels)
        component_labels = flat_labels[flat_indices]
        order = np.argsort(component_labels, kind="stable")
        flat_indices = flat_indices[order]
        counts = np.bincount(component_labels, minlength=count)
        offsets = np.concatenate(([0], np.cumsum(counts[1:])))
        components = []
        for label in range(1, count):
            size = int(counts[label])
            if size < int(min_cells):
                continue
            start = int(offsets[label - 1])
            indices = flat_indices[start : start + size]
            rows, cols = np.divmod(indices, width)
            components.append(np.column_stack((rows, cols)))
        return components
    except ImportError:
        pass

    visited = np.zeros_like(occupied, dtype=bool)
    components = []
    for row in range(height):
        for col in range(width):
            if visited[row, col] or not occupied[row, col]:
                continue
            queue = deque([(row, col)])
            visited[row, col] = True
            cells = []
            while queue:
                current_row, current_col = queue.popleft()
                cells.append((current_row, current_col))
                for n_row, n_col in _neighbors(current_row, current_col, height, width, connectivity):
                    if not visited[n_row, n_col] and occupied[n_row, n_col]:
                        visited[n_row, n_col] = True
                        queue.append((n_row, n_col))
            if len(cells) >= min_cells:
                components.append(np.asarray(cells, dtype=int))
    return components


def dilate_binary_map(binary_map, radius_cells=0):
    mask = np.asarray(binary_map) > 0
    if mask.ndim != 2:
        raise ValueError("binary_map must be a 2D array")
    radius = int(radius_cells)
    if radius < 0:
        raise ValueError("radius_cells must be non-negative")
    if radius == 0 or not np.any(mask):
        return mask

    try:
        import cv2

        kernel_size = 2 * radius + 1
        kernel = np.ones(
            (kernel_size, kernel_size),
            dtype=np.uint8,
        )
        return cv2.dilate(
            mask.astype(np.uint8),
            kernel,
            iterations=1,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
    except ImportError:
        pass

    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    dilated = np.zeros_like(mask, dtype=bool)
    for d_row in range(-radius, radius + 1):
        for d_col in range(-radius, radius + 1):
            row_start = radius + d_row
            col_start = radius + d_col
            dilated |= padded[row_start : row_start + mask.shape[0], col_start : col_start + mask.shape[1]]
    return dilated


def gate_occupancy_by_support(occupancy, support_map, dilation_cells=0):
    occupied = np.asarray(occupancy) > 0
    support = np.asarray(support_map) > 0
    if occupied.ndim != 2:
        raise ValueError("occupancy must be a 2D array")
    if support.shape != occupied.shape:
        raise ValueError("support_map must have the same shape as occupancy")
    support = dilate_binary_map(support, dilation_cells)
    return (occupied & support).astype(np.float32)


def split_component_by_support(component, support_map, min_cells=1, connectivity=8):
    support = np.asarray(support_map) > 0
    if support.ndim != 2:
        raise ValueError("support_map must be a 2D array")
    rows = component[:, 0]
    cols = component[:, 1]
    if np.any(rows < 0) or np.any(rows >= support.shape[0]) or np.any(cols < 0) or np.any(cols >= support.shape[1]):
        raise ValueError("component contains cells outside support_map")
    component_support = np.zeros_like(support, dtype=bool)
    supported = support[rows, cols]
    if not np.any(supported):
        return ()
    component_support[rows[supported], cols[supported]] = True
    return tuple(connected_occupied_components(component_support, min_cells=min_cells, connectivity=connectivity))


def component_to_obstacle(component, spec, obstacle_id, score_map=None):
    rows = component[:, 0]
    cols = component[:, 1]
    x_centers, y_centers = grid_rc_to_metric_xy_centers(rows, cols, spec)
    row_min = int(rows.min())
    row_max = int(rows.max())
    col_min = int(cols.min())
    col_max = int(cols.max())
    score = 1.0
    if score_map is not None:
        scores = np.asarray(score_map, dtype=float)[rows, cols]
        score = float(np.clip(scores.mean(), 0.0, 1.0))

    return BevObstacle(
        obstacle_id=int(obstacle_id),
        center_x=float(x_centers.mean()),
        center_y=float(y_centers.mean()),
        size_x=float((col_max - col_min + 1) * spec.resolution),
        size_y=float((row_max - row_min + 1) * spec.resolution),
        cell_count=int(component.shape[0]),
        score=score,
        row_min=row_min,
        row_max=row_max,
        col_min=col_min,
        col_max=col_max,
    )


def detect_occupied_obstacles(
    occupancy,
    spec,
    min_cells=3,
    connectivity=8,
    score_map=None,
    support_map=None,
    min_support_ratio=0.0,
    support_dilation_cells=0,
    split_components_by_support=False,
    max_size_x=None,
    max_size_y=None,
    min_cells_by_distance=None,
    max_size_x_by_distance=None,
    max_size_y_by_distance=None,
    max_aspect_ratio=None,
    max_aspect_ratio_by_distance=None,
):
    if min_support_ratio < 0.0 or min_support_ratio > 1.0:
        raise ValueError("min_support_ratio must be in [0, 1]")
    _validate_optional_positive("max_size_x", max_size_x)
    _validate_optional_positive("max_size_y", max_size_y)
    _validate_optional_positive("max_aspect_ratio", max_aspect_ratio)
    support = None
    if support_map is not None:
        support = np.asarray(support_map) > 0
        if support.shape != np.asarray(occupancy).shape:
            raise ValueError("support_map must have the same shape as occupancy")
        support = dilate_binary_map(support, support_dilation_cells)

    min_cells_rules = tuple(min_cells_by_distance or ())
    max_size_x_rules = tuple(max_size_x_by_distance or ())
    max_size_y_rules = tuple(max_size_y_by_distance or ())
    max_aspect_ratio_rules = tuple(max_aspect_ratio_by_distance or ())
    component_min_cells = int(min_cells)
    if min_cells_rules:
        component_min_cells = min(component_min_cells, min(rule[2] for rule in min_cells_rules))

    components = connected_occupied_components(occupancy, min_cells=component_min_cells, connectivity=connectivity)
    obstacles = []
    for component in components:
        candidate_components = (component,)
        if split_components_by_support and support is not None:
            split_components = split_component_by_support(
                component,
                support,
                min_cells=component_min_cells,
                connectivity=connectivity,
            )
            if split_components:
                candidate_components = split_components
        for candidate_component in candidate_components:
            if support is not None and min_support_ratio > 0.0:
                rows = candidate_component[:, 0]
                cols = candidate_component[:, 1]
                support_ratio = float(np.count_nonzero(support[rows, cols])) / float(candidate_component.shape[0])
                if support_ratio < min_support_ratio:
                    continue
            obstacle = component_to_obstacle(candidate_component, spec, obstacle_id=len(obstacles) + 1, score_map=score_map)
            distance_m = float(np.hypot(obstacle.center_x, obstacle.center_y))
            required_cells = min_cells_for_distance(distance_m, min_cells, min_cells_rules)
            if obstacle.cell_count < required_cells:
                continue
            distance_max_size_x = value_for_distance(distance_m, max_size_x, max_size_x_rules)
            distance_max_size_y = value_for_distance(distance_m, max_size_y, max_size_y_rules)
            distance_max_aspect_ratio = value_for_distance(distance_m, max_aspect_ratio, max_aspect_ratio_rules)
            if distance_max_size_x is not None and obstacle.size_x > float(distance_max_size_x):
                continue
            if distance_max_size_y is not None and obstacle.size_y > float(distance_max_size_y):
                continue
            if distance_max_aspect_ratio is not None:
                shorter = max(min(obstacle.size_x, obstacle.size_y), 1.0e-9)
                aspect_ratio = max(obstacle.size_x, obstacle.size_y) / shorter
                if aspect_ratio > float(distance_max_aspect_ratio):
                    continue
            obstacles.append(obstacle)
    return tuple(obstacles)
