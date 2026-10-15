"""Lightweight camera/LiDAR/radar BEV support fusion primitives."""

import math

import numpy as np

from bev_obstacle_detection.bev_grid import points_to_grid_indices
from bev_obstacle_detection.obstacle_detection import dilate_binary_map


def normalize_support_map(support):
    array = np.asarray(support, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("support map must be 2D")
    if array.size == 0:
        return array
    max_value = float(array.max())
    if max_value <= 0.0:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip(array / max_value, 0.0, 1.0).astype(np.float32)


def calibrated_fusion_score(
    base_score_map,
    fused_score_map,
    lidar_quality,
    modality_qualities,
):
    """Keep relative fusion geometry while preserving absolute uncertainty."""
    base = np.asarray(base_score_map, dtype=np.float32)
    fused = np.asarray(fused_score_map, dtype=np.float32)
    if base.shape != fused.shape:
        raise ValueError("base and fused score maps must have the same shape")
    lidar_scale = float(np.clip(lidar_quality, 0.0, 1.0))
    values = [
        float(np.clip(value, 0.0, 1.0))
        for value in dict(modality_qualities or {}).values()
    ]
    fusion_scale = max(values) if values else 0.0
    return np.maximum(base * lidar_scale, fused * fusion_scale).astype(np.float32)


def image_quality_score(image, exposure_target=0.45, contrast_reference=64.0, gradient_reference=32.0):
    source = np.asarray(image)
    if source.ndim != 2:
        raise ValueError("image must be a 2D grayscale array")
    if source.size == 0:
        return 0.0
    try:
        import cv2

        if source.dtype == np.uint8:
            clipped = source
        else:
            gray = np.asarray(source, dtype=np.float32)
            clipped = (
                gray
                if float(gray.min()) >= 0.0 and float(gray.max()) <= 255.0
                else np.clip(gray, 0.0, 255.0)
            )
        mean, stddev = cv2.meanStdDev(clipped)
        brightness = float(mean[0, 0]) / 255.0
        gray_std = float(stddev[0, 0])
    except ImportError:
        gray = np.asarray(source, dtype=np.float32)
        clipped = np.clip(gray, 0.0, 255.0)
        brightness = float(clipped.mean()) / 255.0
        gray_std = float(gray.std())
    target = max(float(exposure_target), 1.0e-6)
    exposure = 1.0 - min(abs(brightness - target) / target, 1.0)
    contrast = min(gray_std / float(contrast_reference), 1.0) if contrast_reference > 0.0 else 1.0
    if source.shape[0] > 1 or source.shape[1] > 1:
        diffs = []
        if source.shape[1] > 1:
            try:
                diffs.append(
                    cv2.mean(
                        cv2.absdiff(source[:, 1:], source[:, :-1])
                    )[0]
                )
            except (ImportError, NameError):
                gray = np.asarray(source, dtype=np.float32)
                diffs.append(np.abs(np.diff(gray, axis=1)).mean())
        if source.shape[0] > 1:
            try:
                diffs.append(
                    cv2.mean(
                        cv2.absdiff(source[1:, :], source[:-1, :])
                    )[0]
                )
            except (ImportError, NameError):
                gray = np.asarray(source, dtype=np.float32)
                diffs.append(np.abs(np.diff(gray, axis=0)).mean())
        gradient = min(float(np.mean(diffs)) / float(gradient_reference), 1.0) if gradient_reference > 0.0 else 1.0
    else:
        gradient = 0.0
    return float(np.clip(0.4 * exposure + 0.4 * contrast + 0.2 * gradient, 0.0, 1.0))


def robust_image_quality_score(
    image,
    sharpness_degraded_laplacian=0.20,
    sharpness_clean_laplacian=0.80,
    noise_clean_laplacian=4.0,
    noise_degraded_laplacian=20.0,
    min_degradation_quality=0.05,
):
    source = np.asarray(image)
    base_quality = image_quality_score(source)
    if source.shape[0] < 3 or source.shape[1] < 3:
        return base_quality

    try:
        import cv2

        laplacian = cv2.Laplacian(
            source,
            cv2.CV_32F,
            ksize=1,
        )[1:-1, 1:-1]
    except ImportError:
        gray = np.asarray(source, dtype=np.float32)
        center = gray[1:-1, 1:-1]
        laplacian = (
            gray[1:-1, :-2]
            + gray[1:-1, 2:]
            + gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            - 4.0 * center
        )
    laplacian_response = float(np.mean(np.abs(laplacian)))

    sharpness_span = max(
        float(sharpness_clean_laplacian) - float(sharpness_degraded_laplacian),
        1.0e-6,
    )
    sharpness_quality = np.clip(
        (laplacian_response - float(sharpness_degraded_laplacian)) / sharpness_span,
        0.0,
        1.0,
    )
    noise_span = max(
        float(noise_degraded_laplacian) - float(noise_clean_laplacian),
        1.0e-6,
    )
    noise_degradation = np.clip(
        (laplacian_response - float(noise_clean_laplacian)) / noise_span,
        0.0,
        1.0,
    )
    minimum = float(np.clip(min_degradation_quality, 0.0, 1.0))
    degradation_quality = max(
        minimum,
        min(float(sharpness_quality), 1.0 - (1.0 - minimum) * float(noise_degradation)),
    )
    return float(np.clip(base_quality * degradation_quality, 0.0, 1.0))


def image_edge_score(image, percentile=95.0):
    gray = np.asarray(image, dtype=np.float32)
    if gray.ndim != 2:
        raise ValueError("image must be a 2D grayscale array")
    if gray.size == 0:
        return gray.astype(np.float32)

    dx = np.zeros_like(gray, dtype=np.float32)
    dy = np.zeros_like(gray, dtype=np.float32)
    if gray.shape[1] > 1:
        dx[:, 1:-1] = 0.5 * (gray[:, 2:] - gray[:, :-2])
        dx[:, 0] = gray[:, 1] - gray[:, 0]
        dx[:, -1] = gray[:, -1] - gray[:, -2]
    if gray.shape[0] > 1:
        dy[1:-1, :] = 0.5 * (gray[2:, :] - gray[:-2, :])
        dy[0, :] = gray[1, :] - gray[0, :]
        dy[-1, :] = gray[-1, :] - gray[-2, :]

    gradient = np.sqrt(dx * dx + dy * dy)
    positive = gradient[gradient > 0.0]
    if positive.size == 0:
        return np.zeros_like(gray, dtype=np.float32)
    scale = float(np.percentile(positive, np.clip(float(percentile), 1.0, 100.0)))
    if scale <= 0.0:
        return np.zeros_like(gray, dtype=np.float32)
    return np.clip(gradient / scale, 0.0, 1.0).astype(np.float32)


def camera_support_score_image(
    image_gray,
    mode="intensity",
    edge_percentile=95.0,
    edge_weight=0.6,
    min_score=0.0,
):
    source = np.asarray(image_gray)
    if source.ndim != 2:
        raise ValueError("image_gray must be 2D")
    mode = str(mode or "intensity").strip().lower()
    min_value = float(np.clip(min_score, 0.0, 1.0))
    if mode in ("semantic_mask", "mask") and source.dtype == np.uint8:
        if min_value <= 0.0:
            return source
        threshold = int(np.ceil(min_value * 255.0))
        return np.where(source >= threshold, source, 0).astype(
            np.uint8,
            copy=False,
        )

    gray = np.asarray(image_gray, dtype=np.float32)
    intensity = np.clip(gray, 0.0, 255.0) / 255.0
    if mode == "intensity":
        score = intensity
    elif mode == "edge":
        score = image_edge_score(gray, percentile=edge_percentile)
    elif mode == "hybrid":
        edge = image_edge_score(gray, percentile=edge_percentile)
        weight = float(np.clip(edge_weight, 0.0, 1.0))
        score = (1.0 - weight) * intensity + weight * edge
    elif mode in ("semantic_mask", "mask"):
        score = intensity
    else:
        raise ValueError("unsupported camera support mode: {}".format(mode))

    if min_value > 0.0:
        score = np.where(score >= min_value, score, 0.0)
    return (np.clip(score, 0.0, 1.0) * 255.0).astype(np.float32)


def semantic_mask_quality_score(
    mask,
    min_coverage=0.001,
    saturation_coverage=0.12,
    overcoverage_start=0.5,
    max_coverage=0.9,
    overcoverage_min_quality=0.05,
):
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("mask must be 2D")
    if array.size == 0:
        return 0.0
    coverage = float(np.count_nonzero(array > 0.0)) / float(array.size)
    if coverage < float(min_coverage):
        return 0.0
    saturation = max(float(saturation_coverage), float(min_coverage))
    quality = float(np.clip(coverage / saturation, 0.0, 1.0))
    start = float(overcoverage_start)
    max_cov = float(max_coverage)
    if max_cov > start and coverage > start:
        min_quality = float(np.clip(overcoverage_min_quality, 0.0, 1.0))
        excess_ratio = np.clip((coverage - start) / (max_cov - start), 0.0, 1.0)
        penalty = 1.0 - (1.0 - min_quality) * excess_ratio
        quality *= float(np.clip(penalty, min_quality, 1.0))
    return float(np.clip(quality, 0.0, 1.0))


def camera_semantic_reliability_score(image_quality, semantic_quality, image_quality_floor=0.25):
    """Combine image observability and mask quality without trusting either alone."""
    image_value = float(np.clip(image_quality, 0.0, 1.0))
    semantic_value = float(np.clip(semantic_quality, 0.0, 1.0))
    floor = float(np.clip(image_quality_floor, 0.0, 1.0))
    return semantic_value * (floor + (1.0 - floor) * image_value)


def lidar_quality_score(
    points_xyz,
    saturation_points=5000,
    range_deviation=None,
    clean_range_deviation=0.07,
    degraded_range_deviation=0.16,
    min_noise_quality=0.10,
):
    """Estimate LiDAR reliability from usable density and local range consistency."""
    points = np.asarray(points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape Nx3")
    density_quality = float(
        np.clip(points.shape[0] / float(max(1, int(saturation_points))), 0.0, 1.0)
    )
    if range_deviation is None:
        return density_quality
    deviation = max(0.0, float(range_deviation))
    clean = max(0.0, float(clean_range_deviation))
    degraded = max(clean + 1.0e-9, float(degraded_range_deviation))
    minimum = float(np.clip(min_noise_quality, 0.0, 1.0))
    ratio = float(np.clip((deviation - clean) / (degraded - clean), 0.0, 1.0))
    noise_quality = 1.0 - (1.0 - minimum) * ratio
    return density_quality * noise_quality


class CausalRelativeDensityTracker:
    """Detect sudden point-density loss against a causal healthy envelope."""

    def __init__(self, envelope_decay=0.999, warmup_frames=10):
        self.envelope_decay = float(np.clip(envelope_decay, 0.9, 1.0))
        self.warmup_frames = max(1, int(warmup_frames))
        self.expected_count = None
        self.frame_count = 0

    def reset(self):
        self.expected_count = None
        self.frame_count = 0

    def update(self, point_count):
        count = max(0.0, float(point_count))
        self.frame_count += 1
        if self.expected_count is None:
            self.expected_count = max(count, 1.0)
        else:
            self.expected_count = max(
                count,
                self.expected_count * self.envelope_decay,
                1.0,
            )
        if self.frame_count < self.warmup_frames:
            return 1.0
        return float(np.clip(count / self.expected_count, 0.0, 1.0))


class TemporalModalityReliabilityController:
    """Smooth modality reliability with fast degradation and slower recovery."""

    def __init__(
        self,
        rise_alpha=0.20,
        fall_alpha=0.70,
        failure_threshold=0.05,
        recovery_threshold=0.12,
        use_smoothed_weights=False,
    ):
        self.rise_alpha = float(np.clip(rise_alpha, 0.0, 1.0))
        self.fall_alpha = float(np.clip(fall_alpha, 0.0, 1.0))
        self.failure_threshold = float(np.clip(failure_threshold, 0.0, 1.0))
        self.recovery_threshold = float(
            np.clip(max(recovery_threshold, self.failure_threshold), 0.0, 1.0)
        )
        self.use_smoothed_weights = bool(use_smoothed_weights)
        self.smoothed_qualities = {}
        self.latest_qualities = {}
        self.active = {}

    def reset(self):
        self.smoothed_qualities.clear()
        self.latest_qualities.clear()
        self.active.clear()

    def update(self, qualities, priors=None, min_lidar_weight=0.0):
        priors = priors or {}
        names = sorted(set(priors) | set(qualities) | set(self.smoothed_qualities))
        for name in names:
            target = float(np.clip(qualities.get(name, 0.0), 0.0, 1.0))
            self.latest_qualities[name] = target
            previous = self.smoothed_qualities.get(name, target)
            alpha = self.fall_alpha if target < previous else self.rise_alpha
            smoothed = previous + alpha * (target - previous)
            self.smoothed_qualities[name] = float(np.clip(smoothed, 0.0, 1.0))

            was_active = self.active.get(name, target >= self.recovery_threshold)
            if was_active and self.smoothed_qualities[name] <= self.failure_threshold:
                was_active = False
            elif not was_active and self.smoothed_qualities[name] >= self.recovery_threshold:
                was_active = True
            self.active[name] = was_active

        scores = {}
        for name in names:
            if not self.active.get(name, False):
                scores[name] = (
                    float(min_lidar_weight)
                    if name == "lidar" and name in qualities
                    else 0.0
                )
                continue
            quality = (
                self.smoothed_qualities[name]
                if self.use_smoothed_weights
                else self.latest_qualities[name]
            )
            score = quality * max(0.0, float(priors.get(name, 1.0)))
            if name == "lidar":
                score = max(score, float(min_lidar_weight))
            scores[name] = score
        total = sum(scores.values())
        if total <= 0.0:
            return {name: 0.0 for name in names}
        return {name: score / total for name, score in scores.items()}

    def state_label(self):
        failed = sorted(name for name, active in self.active.items() if not active)
        degraded = sorted(
            name
            for name, quality in self.smoothed_qualities.items()
            if self.active.get(name, False) and quality < 0.35
        )
        if failed:
            return "fallback:" + "+".join(failed)
        if degraded:
            return "degraded:" + "+".join(degraded)
        return "normal"


def support_from_points(points_xyz, spec, weights=None, dilation_cells=0):
    points = np.asarray(points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape Nx3")
    if weights is None:
        weights = np.ones((points.shape[0],), dtype=np.float32)
    else:
        weights = np.asarray(weights, dtype=np.float32)
        if weights.shape != (points.shape[0],):
            raise ValueError("weights must have shape N")
        if np.any(weights < 0.0):
            raise ValueError("weights must be non-negative")

    support = np.zeros((spec.height, spec.width), dtype=np.float32)
    if points.size == 0:
        return support
    rows, cols, mask = points_to_grid_indices(points, spec)
    if np.any(mask):
        np.add.at(support, (rows, cols), weights[mask])
    support = normalize_support_map(support)
    if int(dilation_cells) > 0 and np.any(support > 0.0):
        support = dilate_binary_map(support > 0.0, radius_cells=int(dilation_cells)).astype(np.float32)
    return support


def camera_support_from_body_points(
    points_xyz,
    image_gray,
    spec,
    horizontal_fov_deg=69.0,
    vertical_fov_deg=42.0,
    min_depth_m=0.2,
    dilation_cells=0,
):
    points = np.asarray(points_xyz, dtype=float)
    gray = np.asarray(image_gray, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape Nx3")
    if gray.ndim != 2:
        raise ValueError("image_gray must be 2D")
    if gray.size == 0 or points.size == 0:
        return np.zeros((spec.height, spec.width), dtype=np.float32)

    height, width = gray.shape
    fx = width / (2.0 * math.tan(math.radians(float(horizontal_fov_deg)) / 2.0))
    fy = height / (2.0 * math.tan(math.radians(float(vertical_fov_deg)) / 2.0))
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0

    depth = points[:, 0]
    valid = depth > float(min_depth_m)
    if not np.any(valid):
        return np.zeros((spec.height, spec.width), dtype=np.float32)

    # Body frame convention: x forward, y left, z up. Realsense depth-camera
    # frame used elsewhere maps body x=z_cam, body y=-x_cam, body z=-y_cam.
    u = fx * (-points[:, 1] / np.maximum(depth, 1.0e-6)) + cx
    v = fy * (-points[:, 2] / np.maximum(depth, 1.0e-6)) + cy
    u_int = np.rint(u).astype(int)
    v_int = np.rint(v).astype(int)
    valid &= (u_int >= 0) & (u_int < width) & (v_int >= 0) & (v_int < height)
    if not np.any(valid):
        return np.zeros((spec.height, spec.width), dtype=np.float32)

    pixel_scores = np.clip(gray[v_int[valid], u_int[valid]] / 255.0, 0.0, 1.0).astype(np.float32)
    return support_from_points(points[valid], spec, weights=pixel_scores, dilation_cells=dilation_cells)


def radar_spatial_inlier_mask(points_xyz, neighbor_radius=0.75, min_neighbors=2):
    """Return Radar points supported by a small local XY neighborhood."""
    points = np.asarray(points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape Nx3")
    count = points.shape[0]
    if count == 0:
        return np.zeros((0,), dtype=bool)
    radius = float(neighbor_radius)
    required = max(0, int(min_neighbors))
    if radius <= 0.0 or required == 0:
        return np.all(np.isfinite(points[:, :2]), axis=1)

    valid = np.all(np.isfinite(points[:, :2]), axis=1)
    try:
        from scipy.spatial import cKDTree

        valid_indices = np.flatnonzero(valid)
        if valid_indices.size == 0:
            return np.zeros((count,), dtype=bool)
        valid_points = points[valid_indices, :2]
        neighbor_distances = cKDTree(valid_points).query(
            valid_points,
            k=required + 1,
            distance_upper_bound=np.nextafter(radius, np.inf),
        )
        inliers = np.zeros((count,), dtype=bool)
        distances = np.asarray(neighbor_distances[0], dtype=float)
        inliers[valid_indices] = np.isfinite(distances[:, -1])
        return inliers
    except ImportError:
        pass

    cells = np.floor(points[:, :2] / radius).astype(np.int64, copy=False)
    buckets = {}
    for index in np.flatnonzero(valid):
        key = (int(cells[index, 0]), int(cells[index, 1]))
        buckets.setdefault(key, []).append(int(index))

    radius_sq = radius * radius
    inliers = np.zeros((count,), dtype=bool)
    for index in np.flatnonzero(valid):
        cell_x, cell_y = int(cells[index, 0]), int(cells[index, 1])
        neighbors = 0
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                for candidate in buckets.get((cell_x + offset_x, cell_y + offset_y), ()):
                    if candidate == index:
                        continue
                    delta = points[candidate, :2] - points[index, :2]
                    if float(np.dot(delta, delta)) <= radius_sq:
                        neighbors += 1
                        if neighbors >= required:
                            inliers[index] = True
                            break
                if inliers[index]:
                    break
            if inliers[index]:
                break
    return inliers


def radar_temporal_inlier_mask(
    points_xyz,
    reference_points_xyz,
    neighbor_radius=0.75,
    min_neighbors=1,
):
    """Return current Radar points supported by VINS-aligned history points."""
    points = np.asarray(points_xyz, dtype=float)
    reference = np.asarray(reference_points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape Nx3")
    if reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("reference_points_xyz must have shape Nx3")
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    radius = float(neighbor_radius)
    required = max(0, int(min_neighbors))
    if required == 0:
        return np.all(np.isfinite(points[:, :2]), axis=1)
    if radius <= 0.0 or reference.shape[0] == 0:
        return np.zeros((points.shape[0],), dtype=bool)

    reference_valid = np.all(np.isfinite(reference[:, :2]), axis=1)
    current_valid = np.all(np.isfinite(points[:, :2]), axis=1)
    try:
        from scipy.spatial import cKDTree

        reference_xy = reference[reference_valid, :2]
        current_indices = np.flatnonzero(current_valid)
        if reference_xy.shape[0] == 0 or current_indices.size == 0:
            return np.zeros((points.shape[0],), dtype=bool)
        neighbor_distances = cKDTree(reference_xy).query(
            points[current_indices, :2],
            k=required,
            distance_upper_bound=np.nextafter(radius, np.inf),
        )
        inliers = np.zeros((points.shape[0],), dtype=bool)
        distances = np.asarray(neighbor_distances[0], dtype=float)
        if required == 1:
            inliers[current_indices] = np.isfinite(distances)
        else:
            inliers[current_indices] = np.isfinite(distances[:, -1])
        return inliers
    except ImportError:
        pass

    reference_cells = np.floor(reference[:, :2] / radius).astype(np.int64, copy=False)
    buckets = {}
    for index in np.flatnonzero(reference_valid):
        key = (int(reference_cells[index, 0]), int(reference_cells[index, 1]))
        buckets.setdefault(key, []).append(int(index))

    current_cells = np.floor(points[:, :2] / radius).astype(np.int64, copy=False)
    radius_sq = radius * radius
    inliers = np.zeros((points.shape[0],), dtype=bool)
    for index in np.flatnonzero(current_valid):
        cell_x, cell_y = int(current_cells[index, 0]), int(current_cells[index, 1])
        neighbors = 0
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                for candidate in buckets.get((cell_x + offset_x, cell_y + offset_y), ()):
                    delta = reference[candidate, :2] - points[index, :2]
                    if float(np.dot(delta, delta)) <= radius_sq:
                        neighbors += 1
                        if neighbors >= required:
                            inliers[index] = True
                            break
                if inliers[index]:
                    break
            if inliers[index]:
                break
    return inliers


def radar_intensity_inlier_mask(intensities, min_intensity=0.0):
    values = np.asarray(intensities, dtype=float)
    if values.ndim != 1:
        raise ValueError("intensities must be a 1D array")
    return np.isfinite(values) & (values >= float(min_intensity))


def adaptive_radar_consistency_mask(
    spatial_mask,
    temporal_mask,
    min_temporal_inliers=3,
    min_temporal_ratio=0.01,
    fallback_min_spatial_ratio=0.80,
):
    """Use temporal filtering only when it has enough supporting evidence."""
    spatial = np.asarray(spatial_mask, dtype=bool)
    temporal = np.asarray(temporal_mask, dtype=bool)
    if spatial.ndim != 1 or temporal.ndim != 1 or spatial.shape != temporal.shape:
        raise ValueError("spatial_mask and temporal_mask must be matching 1D arrays")
    if spatial.size == 0:
        return spatial.copy(), False

    strict = spatial & temporal
    temporal_ratio = float(np.mean(temporal))
    spatial_ratio = float(np.mean(spatial))
    temporal_is_usable = (
        int(np.count_nonzero(strict)) >= max(1, int(min_temporal_inliers))
        and temporal_ratio >= max(0.0, float(min_temporal_ratio))
    )
    if temporal_is_usable:
        return strict, False
    if spatial_ratio >= float(fallback_min_spatial_ratio):
        return spatial.copy(), True
    return strict, False


def radar_quality_score(
    points_xyz,
    min_points=3,
    saturation_points=64,
    spatial_inlier_ratio=None,
    min_spatial_quality=0.25,
):
    points = np.asarray(points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape Nx3")
    if points.shape[0] < int(min_points):
        return 0.0
    saturation = max(int(saturation_points), int(min_points))
    density_quality = float(np.clip(points.shape[0] / float(saturation), 0.0, 1.0))
    if spatial_inlier_ratio is None:
        return density_quality
    ratio = float(np.clip(spatial_inlier_ratio, 0.0, 1.0))
    minimum = float(np.clip(min_spatial_quality, 0.0, 1.0))
    return density_quality * (minimum + (1.0 - minimum) * ratio)


def adaptive_support_weights(qualities, priors=None, min_lidar_weight=0.0):
    priors = priors or {}
    scores = {}
    for name, quality in qualities.items():
        score = max(0.0, float(quality)) * max(0.0, float(priors.get(name, 1.0)))
        scores[name] = score
    if "lidar" in scores:
        scores["lidar"] = max(scores["lidar"], float(min_lidar_weight))
    total = sum(scores.values())
    if total <= 0.0:
        return {name: 0.0 for name in scores}
    return {name: value / total for name, value in scores.items()}


def fuse_bev_support_maps(lidar_map, support_maps=None, weights=None, threshold=0.5):
    lidar = np.asarray(lidar_map, dtype=np.float32)
    if lidar.ndim != 2:
        raise ValueError("lidar_map must be 2D")
    support_maps = support_maps or {}
    weights = weights or {"lidar": 1.0}
    score = max(0.0, float(weights.get("lidar", 1.0))) * normalize_support_map(lidar)
    for name, support in support_maps.items():
        array = np.asarray(support, dtype=np.float32)
        if array.shape != lidar.shape:
            raise ValueError("support map shape must match lidar_map")
        score = score + max(0.0, float(weights.get(name, 0.0))) * normalize_support_map(array)
    score = np.clip(score, 0.0, 1.0).astype(np.float32)
    occupancy = (score >= float(threshold)).astype(np.float32)
    return occupancy, score


def reliability_gated_temporal_support(
    current_support,
    temporal_support,
    lidar_quality,
    max_lidar_quality=0.12,
    temporal_scale=0.6,
):
    """Add decayed temporal support only while LiDAR is observably degraded."""
    current = np.asarray(current_support, dtype=np.float32)
    temporal = np.asarray(temporal_support, dtype=np.float32)
    if current.ndim != 2 or temporal.shape != current.shape:
        raise ValueError("current and temporal support maps must be matching 2D arrays")
    if max_lidar_quality < 0.0 or max_lidar_quality > 1.0:
        raise ValueError("max_lidar_quality must be in [0, 1]")
    if temporal_scale < 0.0 or temporal_scale > 1.0:
        raise ValueError("temporal_scale must be in [0, 1]")
    active = float(lidar_quality) <= float(max_lidar_quality) and np.any(temporal > 0.0)
    if not active:
        return np.clip(current, 0.0, 1.0), False
    fused = np.maximum(current, np.clip(temporal, 0.0, 1.0) * float(temporal_scale))
    return np.clip(fused, 0.0, 1.0).astype(np.float32), True
