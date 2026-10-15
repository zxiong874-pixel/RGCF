"""Calibration-driven Camera semantic projection into a LiDAR-frame BEV."""

import numpy as np

from bev_obstacle_detection.multimodal_support_fusion import support_from_points


def load_camera_calibration(projection_path, lidar_to_camera_path):
    projection = np.loadtxt(str(projection_path), dtype=float)
    lidar_to_camera = np.loadtxt(str(lidar_to_camera_path), dtype=float)
    if projection.shape == (4, 4):
        projection = projection[:3, :4]
    if projection.shape != (3, 4):
        raise ValueError("camera projection matrix must be 3x4 or 4x4")
    if lidar_to_camera.shape != (4, 4):
        raise ValueError("LiDAR-to-Camera transform must be 4x4")
    return projection, lidar_to_camera


def calibrated_camera_support_from_lidar_points(
    points_lidar,
    score_image,
    spec,
    projection,
    lidar_to_camera,
    min_camera_depth=0.2,
    dilation_cells=0,
):
    points = np.asarray(points_lidar, dtype=float)
    image = np.asarray(score_image)
    projection = np.asarray(projection, dtype=float)
    lidar_to_camera = np.asarray(lidar_to_camera, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_lidar must have shape Nx3")
    if image.ndim != 2:
        raise ValueError("score_image must be 2D")
    if projection.shape != (3, 4) or lidar_to_camera.shape != (4, 4):
        raise ValueError("invalid camera calibration shapes")
    if points.size == 0 or image.size == 0:
        return np.zeros((spec.height, spec.width), dtype=np.float32)

    homogeneous = np.column_stack(
        (points, np.ones((points.shape[0],), dtype=float))
    )
    camera_points = homogeneous @ lidar_to_camera.T
    projected = camera_points @ projection.T
    depth = camera_points[:, 2]
    valid = depth > float(min_camera_depth)
    denominator = np.maximum(projected[:, 2], 1.0e-9)
    u = np.rint(projected[:, 0] / denominator).astype(int)
    v = np.rint(projected[:, 1] / denominator).astype(int)
    height, width = image.shape
    valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(valid):
        return np.zeros((spec.height, spec.width), dtype=np.float32)
    weights = np.clip(
        image[v[valid], u[valid]].astype(np.float32) / 255.0,
        0.0,
        1.0,
    )
    return support_from_points(
        points[valid],
        spec,
        weights=weights.astype(np.float32),
        dilation_cells=dilation_cells,
    )


def semantic_mask_ground_contact_points(
    mask,
    projection,
    lidar_to_camera,
    ground_z=-1.5,
    min_range_m=1.0,
    max_range_m=30.0,
    min_component_area_px=128,
    max_component_fraction=0.15,
    reject_border_components=True,
    column_stride_px=8,
    bottom_band_px=3,
):
    """Project filtered semantic-component bottom envelopes onto a ground plane."""
    import cv2

    image = np.asarray(mask)
    projection = np.asarray(projection, dtype=float)
    lidar_to_camera = np.asarray(lidar_to_camera, dtype=float)
    if image.ndim != 2:
        raise ValueError("mask must be 2D")
    if projection.shape != (3, 4) or lidar_to_camera.shape != (4, 4):
        raise ValueError("invalid camera calibration shapes")
    if min_range_m < 0.0 or max_range_m <= min_range_m:
        raise ValueError("invalid ground projection range")
    if column_stride_px <= 0 or bottom_band_px <= 0:
        raise ValueError("pixel stride and bottom band must be positive")

    binary = (image > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    height, width = binary.shape
    max_area = float(max_component_fraction) * float(binary.size)
    intrinsic = projection[:3, :3]
    intrinsic_inverse = np.linalg.inv(intrinsic)
    rotation = lidar_to_camera[:3, :3]
    camera_origin_lidar = np.linalg.inv(lidar_to_camera)[:3, 3]
    points = []
    accepted_components = 0
    rejected_components = 0

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
            or (reject_border_components and touches_border)
        ):
            rejected_components += 1
            continue

        component = labels[
            y : y + component_height, x : x + component_width
        ] == label
        accepted_components += 1
        for local_u in range(0, component_width, int(column_stride_px)):
            rows = np.flatnonzero(component[:, local_u])
            if rows.size == 0:
                continue
            bottom = int(rows[-1])
            for offset in range(int(bottom_band_px)):
                local_v = bottom - offset
                if local_v < 0 or not component[local_v, local_u]:
                    continue
                u = float(x + local_u)
                v = float(y + local_v)
                ray_camera = intrinsic_inverse @ np.array([u, v, 1.0])
                ray_lidar = rotation.T @ ray_camera
                if ray_lidar[2] >= -1.0e-9:
                    continue
                distance_along_ray = (
                    float(ground_z) - camera_origin_lidar[2]
                ) / ray_lidar[2]
                if distance_along_ray <= 0.0:
                    continue
                point = camera_origin_lidar + distance_along_ray * ray_lidar
                horizontal_range = float(np.linalg.norm(point[:2]))
                if min_range_m <= horizontal_range <= max_range_m:
                    points.append(point)

    return (
        np.asarray(points, dtype=float).reshape((-1, 3)),
        {
            "accepted_components": accepted_components,
            "rejected_components": rejected_components,
            "ground_points": len(points),
        },
    )


def semantic_mask_ground_support(
    mask,
    spec,
    projection,
    lidar_to_camera,
    dilation_cells=2,
    **kwargs
):
    points, stats = semantic_mask_ground_contact_points(
        mask,
        projection,
        lidar_to_camera,
        **kwargs
    )
    support = support_from_points(
        points,
        spec,
        dilation_cells=dilation_cells,
    )
    return support, stats
