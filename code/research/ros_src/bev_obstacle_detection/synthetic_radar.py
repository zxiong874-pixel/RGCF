"""Synthetic sparse radar detections derived from Gazebo model states."""

import math
from dataclasses import dataclass

import numpy as np

from bev_obstacle_detection.detection_metrics import quaternion_to_yaw, world_delta_to_body_xy


@dataclass(frozen=True)
class SyntheticRadarConfig:
    points_per_obstacle: int = 6
    min_range_m: float = 0.5
    max_range_m: float = 30.0
    horizontal_fov_deg: float = 120.0
    obstacle_dropout_rate: float = 0.05
    xy_noise_std_m: float = 0.18
    z_noise_std_m: float = 0.05
    false_positive_points: int = 0
    false_positive_clusters: int = 0
    false_positive_points_per_cluster: int = 3
    false_positive_cluster_std_m: float = 0.12
    false_positive_min_range_m: float = 1.0
    false_positive_max_range_m: float = 20.0


def obstacle_centers_from_model_states(message, obstacle_specs, ego_model):
    """Return obstacle centers in the ego body frame from a Gazebo ModelStates message."""
    name_to_index = {name: index for index, name in enumerate(message.name)}
    ego_index = name_to_index.get(ego_model)
    if ego_index is None:
        return []

    ego_pose = message.pose[ego_index]
    ego_yaw = quaternion_to_yaw(ego_pose.orientation)
    ego_x = float(ego_pose.position.x)
    ego_y = float(ego_pose.position.y)
    centers = []
    for spec in obstacle_specs:
        obstacle_index = name_to_index.get(spec.model_name)
        if obstacle_index is None:
            continue
        obstacle_pose = message.pose[obstacle_index]
        center_x, center_y = world_delta_to_body_xy(
            float(obstacle_pose.position.x) - ego_x,
            float(obstacle_pose.position.y) - ego_y,
            ego_yaw,
        )
        centers.append(
            {
                "model_name": spec.model_name,
                "center_x": center_x,
                "center_y": center_y,
                "size_x": float(spec.size_x),
                "size_y": float(spec.size_y),
            }
        )
    return centers


def _in_radar_view(center_x, center_y, config):
    distance = math.hypot(center_x, center_y)
    if distance < config.min_range_m or distance > config.max_range_m:
        return False
    half_fov = math.radians(config.horizontal_fov_deg) / 2.0
    return abs(math.atan2(center_y, center_x)) <= half_fov


def generate_synthetic_radar_points(
    obstacle_centers,
    config=None,
    rng=None,
    false_positive_rng=None,
):
    """Generate sparse body-frame radar points as Nx4 x/y/z/intensity rows."""
    config = config or SyntheticRadarConfig()
    rng = rng or np.random.RandomState(7)
    false_positive_rng = false_positive_rng or rng
    rows = []

    for center in obstacle_centers:
        center_x = float(center["center_x"])
        center_y = float(center["center_y"])
        if not _in_radar_view(center_x, center_y, config):
            continue
        if config.obstacle_dropout_rate > 0.0 and rng.rand() < config.obstacle_dropout_rate:
            continue

        size_x = max(float(center.get("size_x", 0.0)), 0.1)
        size_y = max(float(center.get("size_y", 0.0)), 0.1)
        count = max(1, int(config.points_per_obstacle))
        distance = max(math.hypot(center_x, center_y), 1e-3)
        for _ in range(count):
            offset_x = rng.normal(0.0, min(size_x * 0.25, config.xy_noise_std_m))
            offset_y = rng.normal(0.0, min(size_y * 0.25, config.xy_noise_std_m))
            noise_x = rng.normal(0.0, config.xy_noise_std_m)
            noise_y = rng.normal(0.0, config.xy_noise_std_m)
            x = center_x + offset_x + noise_x
            y = center_y + offset_y + noise_y
            z = rng.normal(0.35, config.z_noise_std_m)
            intensity = min(1.0, max(0.05, 8.0 / distance + rng.normal(0.0, 0.03)))
            rows.append((x, y, z, intensity))

    false_count = max(0, int(config.false_positive_points))
    if false_count:
        half_fov = math.radians(config.horizontal_fov_deg) / 2.0
        for _ in range(false_count):
            distance = false_positive_rng.uniform(
                config.false_positive_min_range_m,
                config.false_positive_max_range_m,
            )
            angle = false_positive_rng.uniform(-half_fov, half_fov)
            rows.append(
                (
                    distance * math.cos(angle),
                    distance * math.sin(angle),
                    false_positive_rng.normal(0.35, config.z_noise_std_m),
                    false_positive_rng.uniform(0.03, 0.25),
                )
            )

    cluster_count = max(0, int(config.false_positive_clusters))
    points_per_cluster = max(1, int(config.false_positive_points_per_cluster))
    if cluster_count:
        half_fov = math.radians(config.horizontal_fov_deg) / 2.0
        cluster_std = max(0.0, float(config.false_positive_cluster_std_m))
        for _ in range(cluster_count):
            distance = false_positive_rng.uniform(
                config.false_positive_min_range_m,
                config.false_positive_max_range_m,
            )
            angle = false_positive_rng.uniform(-half_fov, half_fov)
            center_x = distance * math.cos(angle)
            center_y = distance * math.sin(angle)
            for _ in range(points_per_cluster):
                rows.append(
                    (
                        center_x + false_positive_rng.normal(0.0, cluster_std),
                        center_y + false_positive_rng.normal(0.0, cluster_std),
                        false_positive_rng.normal(0.35, config.z_noise_std_m),
                        false_positive_rng.uniform(0.03, 0.25),
                    )
                )

    return np.asarray(rows, dtype=np.float32).reshape((-1, 4))
