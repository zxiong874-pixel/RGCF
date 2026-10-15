"""Direct BEV detection helpers for public dataset lidar files."""

import csv
import json
from pathlib import Path

import numpy as np

from bev_obstacle_detection.bev_features import point_cloud_to_bev_feature_maps
from bev_obstacle_detection.bev_grid import BevGridSpec
from bev_obstacle_detection.detection_metrics import (
    Box2D,
    evaluate_detection_distance_bins,
    evaluate_detection_frames,
    parse_distance_bins,
    write_distance_eval_summary_csv,
    write_eval_summary_csv,
    write_gt_csv,
)
from bev_obstacle_detection.obstacle_detection import (
    detect_occupied_obstacles,
    parse_distance_min_cells_rules,
    parse_distance_value_rules,
)
from bev_obstacle_detection.point_filters import filter_points_by_height, filter_points_by_lateral_range_median
from bev_obstacle_detection.public_gt_adapter import load_public_gt_frames
from bev_obstacle_detection.smoke_evaluation import format_number


DETECTION_COLUMNS = [
    "timestamp",
    "method",
    "frame_id",
    "obstacle_id",
    "center_x",
    "center_y",
    "size_x",
    "size_y",
    "cell_count",
    "score",
]


def load_lidar_bin(path, columns=6):
    values = np.fromfile(str(path), dtype=np.float32)
    if values.size % int(columns) != 0:
        raise RuntimeError("lidar file {} does not divide into {} float32 columns".format(path, columns))
    return values.reshape((-1, int(columns)))


def detect_lidar_obstacles(
    points,
    grid_spec=None,
    min_obstacle_cells=3,
    connectivity=8,
    min_point_z=None,
    max_point_z=None,
    use_adaptive_ground_filter=True,
    adaptive_ground_quantile=0.05,
    adaptive_ground_clearance=0.15,
    adaptive_ground_min_points=50,
    use_range_median_filter=False,
    range_filter_lateral_resolution=0.2,
    range_filter_max_deviation=0.25,
    range_filter_min_points=3,
    min_obstacle_cells_by_distance="",
    max_obstacle_size_x=None,
    max_obstacle_size_y=None,
    max_obstacle_size_x_by_distance="",
    max_obstacle_size_y_by_distance="",
    max_obstacle_aspect_ratio=None,
    max_obstacle_aspect_ratio_by_distance="",
):
    spec = grid_spec or BevGridSpec(x_min=-60.0, x_max=80.0, y_min=-60.0, y_max=60.0, z_min=-3.0, z_max=5.0)
    points_xyz = np.asarray(points[:, :3], dtype=float)
    if use_range_median_filter:
        points_xyz = filter_points_by_lateral_range_median(
            points_xyz,
            lateral_resolution=range_filter_lateral_resolution,
            max_range_deviation=range_filter_max_deviation,
            min_points_per_bin=range_filter_min_points,
        )
    points_xyz = filter_points_by_height(
        points_xyz,
        min_point_z=min_point_z,
        max_point_z=max_point_z,
        use_adaptive_ground_filter=use_adaptive_ground_filter,
        adaptive_ground_quantile=adaptive_ground_quantile,
        adaptive_ground_clearance=adaptive_ground_clearance,
        adaptive_ground_min_points=adaptive_ground_min_points,
    )
    features = point_cloud_to_bev_feature_maps(points_xyz, spec)
    return detect_occupied_obstacles(
        features["occupancy"],
        spec,
        min_cells=min_obstacle_cells,
        connectivity=connectivity,
        score_map=features["density"],
        min_cells_by_distance=parse_distance_min_cells_rules(min_obstacle_cells_by_distance),
        max_size_x=max_obstacle_size_x,
        max_size_y=max_obstacle_size_y,
        max_size_x_by_distance=parse_distance_value_rules(max_obstacle_size_x_by_distance, "max_size_x"),
        max_size_y_by_distance=parse_distance_value_rules(max_obstacle_size_y_by_distance, "max_size_y"),
        max_aspect_ratio=max_obstacle_aspect_ratio,
        max_aspect_ratio_by_distance=parse_distance_value_rules(
            max_obstacle_aspect_ratio_by_distance, "max_aspect_ratio"
        ),
    )


def write_detection_csv(path, frame_obstacles, method="bev_lidar_direct", frame_id="boreas_lidar"):
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=DETECTION_COLUMNS)
        writer.writeheader()
        for timestamp, obstacles in sorted(frame_obstacles.items()):
            for obstacle in obstacles:
                writer.writerow(
                    {
                        "timestamp": format_number(timestamp, digits=9),
                        "method": method,
                        "frame_id": frame_id,
                        "obstacle_id": obstacle.obstacle_id,
                        "center_x": format_number(obstacle.center_x),
                        "center_y": format_number(obstacle.center_y),
                        "size_x": format_number(obstacle.size_x),
                        "size_y": format_number(obstacle.size_y),
                        "cell_count": obstacle.cell_count,
                        "score": format_number(obstacle.score),
                    }
                )
    return output


def obstacle_frames_to_box_frames(frame_obstacles):
    box_frames = {}
    for timestamp, obstacles in frame_obstacles.items():
        box_frames[timestamp] = [
            Box2D(
                timestamp=timestamp,
                center_x=obstacle.center_x,
                center_y=obstacle.center_y,
                size_x=obstacle.size_x,
                size_y=obstacle.size_y,
                label=str(obstacle.obstacle_id),
                score=obstacle.score,
            )
            for obstacle in obstacles
        ]
    return box_frames


def expected_sensor_sizes_from_manifest(path, sensor="lidar"):
    manifest_path = Path(path).expanduser() if path else None
    if not manifest_path or not manifest_path.exists():
        return {}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    sizes = {}
    for item in data.get("selected_sensor_files", []):
        if item.get("sensor") == sensor or item.get("relative_path", "").startswith(sensor + "/"):
            sizes[int(item["timestamp_us"])] = int(item["size"])
    for item in data.get("selected", []):
        if item.get("sensor") == sensor or item.get("relative_path", "").startswith(sensor + "/"):
            sizes[int(item["timestamp_us"])] = int(item["size"])
    return sizes


def evaluate_public_lidar(
    gt_csv,
    lidar_root,
    output_dir,
    manifest=None,
    method="bev_lidar_direct",
    iou_threshold=0.10,
    distance_bins="0,3,6,10,15,20,30,50,inf",
    lidar_columns=6,
    allow_partial=False,
    labels=None,
    detector_kwargs=None,
):
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    gt_frames = load_public_gt_frames(gt_csv, fmt="csv", allowed_labels=labels)
    expected_sizes = expected_sensor_sizes_from_manifest(manifest)
    lidar_dir = Path(lidar_root).expanduser()
    frame_obstacles = {}
    skipped = []
    detector_kwargs = detector_kwargs or {}
    for timestamp in sorted(gt_frames):
        timestamp_us = int(round(timestamp * 1.0e6))
        lidar_path = lidar_dir / "{}.bin".format(timestamp_us)
        if not lidar_path.exists():
            skipped.append({"timestamp": timestamp, "reason": "missing_lidar", "path": str(lidar_path)})
            continue
        expected_size = expected_sizes.get(timestamp_us)
        local_size = lidar_path.stat().st_size
        if expected_size and local_size != expected_size and not allow_partial:
            skipped.append(
                {
                    "timestamp": timestamp,
                    "reason": "incomplete_lidar",
                    "path": str(lidar_path),
                    "bytes": local_size,
                    "expected_bytes": expected_size,
                }
            )
            continue
        try:
            points = load_lidar_bin(lidar_path, columns=lidar_columns)
        except RuntimeError as exc:
            skipped.append({"timestamp": timestamp, "reason": "invalid_lidar", "path": str(lidar_path), "error": str(exc)})
            continue
        frame_obstacles[timestamp] = detect_lidar_obstacles(points, **detector_kwargs)

    detection_csv = write_detection_csv(output / "public_lidar_detections.csv", frame_obstacles, method=method)
    detection_box_frames = obstacle_frames_to_box_frames(frame_obstacles)
    evaluated_gt_frames = {timestamp: gt_frames.get(timestamp, []) for timestamp in frame_obstacles}
    metrics = evaluate_detection_frames(detection_box_frames, evaluated_gt_frames, iou_threshold)
    status = "ok" if frame_obstacles else "no_complete_lidar_frames"
    summary_row = {
        "experiment_id": "public_lidar_direct_eval",
        "dataset": "boreas_objects",
        "environment": "real",
        "method": method,
        "bag_path": "",
        "detection_csv_path": str(detection_csv),
        "status": status,
        **metrics,
    }
    summary_csv = write_eval_summary_csv(output / "public_lidar_eval_summary.csv", [summary_row])
    distance_rows = []
    for row in evaluate_detection_distance_bins(
        detection_box_frames,
        evaluated_gt_frames,
        iou_threshold,
        parse_distance_bins(distance_bins),
    ):
        distance_rows.append(
            {
                "experiment_id": "public_lidar_direct_eval",
                "dataset": "boreas_objects",
                "environment": "real",
                "method": method,
                "status": status,
                **row,
            }
        )
    distance_csv = write_distance_eval_summary_csv(output / "public_lidar_eval_by_distance.csv", distance_rows)
    gt_out = write_gt_csv(output / "public_lidar_gt_obstacles.csv", {method: gt_frames}, frame_id="boreas_lidar")
    diagnostics = {
        "status": status,
        "gt_frames_total": len(gt_frames),
        "gt_frames_evaluated": len(evaluated_gt_frames),
        "detection_frames": len(frame_obstacles),
        "skipped": skipped,
        "detection_csv": str(detection_csv),
        "summary_csv": str(summary_csv),
        "distance_csv": str(distance_csv),
        "gt_csv": str(gt_out),
        "metrics": metrics,
    }
    diagnostics_path = output / "public_lidar_eval_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    diagnostics["diagnostics_json"] = str(diagnostics_path)
    return diagnostics
