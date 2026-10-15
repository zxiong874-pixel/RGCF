"""Utilities for reproducible BEV+VINS fusion parameter sweeps."""

import csv
import hashlib
import itertools
import re
from pathlib import Path

from bev_obstacle_detection.detection_metrics import EVAL_COLUMNS, evaluate_method_run
from bev_obstacle_detection.smoke_evaluation import format_number


SWEEP_COLUMNS = [
    "run_id",
    "status_rate",
    "process_on_lidar_callback",
    "lidar_processing_delay_sec",
    "sensor_queue_size",
    "playback_rate",
    "post_replay_wait_sec",
    "temporal_decay_tau",
    "min_current_support_ratio",
    "current_support_dilation_cells",
    "use_current_support_gate",
    "use_dual_obstacle_fusion",
    "use_support_component_split",
    "dual_fusion_iou_threshold",
    "min_obstacle_cells",
    "min_obstacle_cells_by_distance",
    "min_occupancy_weight",
    "min_point_z",
    "max_point_z",
    "use_range_median_filter",
    "use_auto_range_median_filter",
    "range_filter_lateral_resolution",
    "range_filter_max_deviation",
    "range_filter_min_points",
    "auto_range_filter_score_quantile",
    "auto_range_filter_threshold",
    "auto_range_filter_min_points",
    "auto_range_filter_window_frames",
    "auto_range_filter_min_noisy_frames",
    "use_adaptive_ground_filter",
    "adaptive_ground_quantile",
    "adaptive_ground_clearance",
    "adaptive_ground_min_points",
    "max_obstacle_size_x",
    "max_obstacle_size_y",
    "use_adaptive_multimodal_fusion",
    "use_multimodal_sync_quality",
    "multimodal_sync_clean_offset_sec",
    "multimodal_sync_rejected_offset_sec",
    "multimodal_sync_minimum_quality",
    "use_candidate_adaptive_fusion",
    "candidate_reliability_threshold",
    "candidate_evidence_mix",
    "candidate_quality_mix",
    "candidate_local_support_floor",
    "candidate_local_support_exponent",
    "candidate_participation_min_quality",
    "candidate_participation_min_support",
    "use_candidate_temporal_confirmation",
    "candidate_temporal_match_distance",
    "candidate_temporal_grace_frames",
    "candidate_temporal_max_unconfirmed_frames",
    "candidate_temporal_unconfirmed_decay",
    "candidate_temporal_max_track_age",
    "candidate_temporal_min_active_modalities",
    "candidate_temporal_require_initial_confirmation",
    "candidate_temporal_failsafe_quality_threshold",
    "radar_in_base_occupancy",
    "use_radar_spatial_consistency",
    "radar_spatial_neighbor_radius",
    "radar_spatial_min_neighbors",
    "radar_min_spatial_quality",
    "filter_radar_spatial_outliers",
    "use_radar_temporal_consistency",
    "radar_temporal_neighbor_radius",
    "radar_temporal_min_neighbors",
    "radar_temporal_max_frames",
    "radar_temporal_max_age",
    "use_radar_intensity_filter",
    "radar_intensity_field",
    "radar_min_intensity",
    "use_temporal_modality_reliability",
    "allow_modality_dropout",
    "camera_semantic_image_quality_floor",
    "camera_sharpness_degraded_laplacian",
    "camera_sharpness_clean_laplacian",
    "camera_noise_clean_laplacian",
    "camera_noise_degraded_laplacian",
    "camera_image_min_degradation_quality",
    "lidar_quality_clean_range_deviation",
    "lidar_quality_degraded_range_deviation",
    "lidar_quality_min_noise_quality",
    "modality_reliability_rise_alpha",
    "modality_reliability_fall_alpha",
    "modality_use_smoothed_weights",
    "modality_failure_threshold",
    "modality_recovery_threshold",
    "multimodal_fusion_threshold",
    "lidar_fusion_prior",
    "camera_fusion_prior",
    "radar_fusion_prior",
    "min_lidar_fusion_weight",
    "camera_support_dilation_cells",
    "use_camera_semantic_gate",
    "camera_semantic_gate_dilation_cells",
    "radar_support_dilation_cells",
    "reliability_csv_path",
    "candidate_reliability_csv_path",
    "result_dir",
] + EVAL_COLUMNS


def parse_float_list(text):
    values = []
    for item in str(text).split(","):
        stripped = item.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    if not values:
        raise ValueError("expected at least one numeric value")
    return values


def parse_bool_list(text):
    values = []
    true_values = {"1", "true", "yes", "on"}
    false_values = {"0", "false", "no", "off"}
    for item in str(text).split(","):
        stripped = item.strip().lower()
        if not stripped:
            continue
        if stripped in true_values:
            values.append(True)
        elif stripped in false_values:
            values.append(False)
        else:
            raise ValueError("invalid boolean value: {}".format(item.strip()))
    if not values:
        raise ValueError("expected at least one boolean value")
    return values


def parse_string_list(text):
    if text in ("", None):
        return [""]
    values = []
    for item in str(text).split(";"):
        stripped = item.strip()
        if stripped.lower() in ("", "none", "off"):
            values.append("")
        else:
            values.append(stripped)
    return values or [""]


def safe_run_id(params):
    parts = []
    for key in sorted(params):
        value = params[key]
        if isinstance(value, float):
            value_text = ("{:.4g}".format(value)).replace(".", "p").replace("-", "m")
        else:
            value_text = str(value)
        key_text = re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_")
        value_text = re.sub(r"[^A-Za-z0-9pm]+", "_", value_text).strip("_")
        parts.append("{}_{}".format(key_text, value_text))
    run_id = "__".join(parts)
    if len(run_id) <= 120:
        return run_id
    digest = hashlib.sha1(run_id.encode("utf-8")).hexdigest()[:10]
    important = []
    for key in (
        "min_point_z",
        "use_range_median_filter",
        "use_auto_range_median_filter",
        "range_filter_max_deviation",
        "auto_range_filter_threshold",
        "use_adaptive_ground_filter",
        "adaptive_ground_clearance",
        "min_obstacle_cells",
        "min_obstacle_cells_by_distance",
        "use_current_support_gate",
        "use_support_component_split",
        "use_adaptive_multimodal_fusion",
    ):
        if key in params:
            value = str(params[key]).replace(".", "p").replace("-", "m")
            value = re.sub(r"[^A-Za-z0-9pm]+", "_", value).strip("_")
            important.append("{}_{}".format(key, value))
    prefix = "__".join(important)[:90] or "sweep"
    return "{}__{}".format(prefix, digest)


def parameter_grid(
    temporal_decay_tau,
    min_current_support_ratio,
    current_support_dilation_cells,
    use_current_support_gate,
    use_dual_obstacle_fusion,
    use_support_component_split,
    dual_fusion_iou_threshold,
    min_obstacle_cells,
    min_obstacle_cells_by_distance,
    min_occupancy_weight,
    min_point_z,
    max_point_z,
    use_range_median_filter,
    use_auto_range_median_filter,
    range_filter_lateral_resolution,
    range_filter_max_deviation,
    range_filter_min_points,
    auto_range_filter_score_quantile,
    auto_range_filter_threshold,
    auto_range_filter_min_points,
    auto_range_filter_window_frames,
    auto_range_filter_min_noisy_frames,
    use_adaptive_ground_filter,
    adaptive_ground_quantile,
    adaptive_ground_clearance,
    adaptive_ground_min_points,
    max_obstacle_size_x,
    max_obstacle_size_y,
    use_adaptive_multimodal_fusion,
    multimodal_fusion_threshold,
    lidar_fusion_prior,
    camera_fusion_prior,
    radar_fusion_prior,
    min_lidar_fusion_weight,
    camera_support_dilation_cells,
    use_camera_semantic_gate,
    camera_semantic_gate_dilation_cells,
    radar_support_dilation_cells,
):
    keys = [
        "temporal_decay_tau",
        "min_current_support_ratio",
        "current_support_dilation_cells",
        "use_current_support_gate",
        "use_dual_obstacle_fusion",
        "use_support_component_split",
        "dual_fusion_iou_threshold",
        "min_obstacle_cells",
        "min_obstacle_cells_by_distance",
        "min_occupancy_weight",
        "min_point_z",
        "max_point_z",
        "use_range_median_filter",
        "use_auto_range_median_filter",
        "range_filter_lateral_resolution",
        "range_filter_max_deviation",
        "range_filter_min_points",
        "auto_range_filter_score_quantile",
        "auto_range_filter_threshold",
        "auto_range_filter_min_points",
        "auto_range_filter_window_frames",
        "auto_range_filter_min_noisy_frames",
        "use_adaptive_ground_filter",
        "adaptive_ground_quantile",
        "adaptive_ground_clearance",
        "adaptive_ground_min_points",
        "max_obstacle_size_x",
        "max_obstacle_size_y",
        "use_adaptive_multimodal_fusion",
        "multimodal_fusion_threshold",
        "lidar_fusion_prior",
        "camera_fusion_prior",
        "radar_fusion_prior",
        "min_lidar_fusion_weight",
        "camera_support_dilation_cells",
        "use_camera_semantic_gate",
        "camera_semantic_gate_dilation_cells",
        "radar_support_dilation_cells",
    ]
    for values in itertools.product(
        temporal_decay_tau,
        min_current_support_ratio,
        current_support_dilation_cells,
        use_current_support_gate,
        use_dual_obstacle_fusion,
        use_support_component_split,
        dual_fusion_iou_threshold,
        min_obstacle_cells,
        min_obstacle_cells_by_distance,
        min_occupancy_weight,
        min_point_z,
        max_point_z,
        use_range_median_filter,
        use_auto_range_median_filter,
        range_filter_lateral_resolution,
        range_filter_max_deviation,
        range_filter_min_points,
        auto_range_filter_score_quantile,
        auto_range_filter_threshold,
        auto_range_filter_min_points,
        auto_range_filter_window_frames,
        auto_range_filter_min_noisy_frames,
        use_adaptive_ground_filter,
        adaptive_ground_quantile,
        adaptive_ground_clearance,
        adaptive_ground_min_points,
        max_obstacle_size_x,
        max_obstacle_size_y,
        use_adaptive_multimodal_fusion,
        multimodal_fusion_threshold,
        lidar_fusion_prior,
        camera_fusion_prior,
        radar_fusion_prior,
        min_lidar_fusion_weight,
        camera_support_dilation_cells,
        use_camera_semantic_gate,
        camera_semantic_gate_dilation_cells,
        radar_support_dilation_cells,
    ):
        yield dict(zip(keys, values))


def evaluate_fusion_run(config, gt_config, result_dir, bag_root=None):
    if not config.datasets:
        raise ValueError("experiment config has no datasets")
    if not config.environments:
        raise ValueError("experiment config has no environments")
    row, _ = evaluate_method_run(
        config,
        config.datasets[0],
        config.environments[0],
        "bev_vins_fusion",
        result_dir,
        gt_config,
        bag_root=bag_root,
    )
    return row


def sweep_score(row):
    def value(name, fallback):
        raw = row.get(name)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return fallback

    return (
        value("f1", -1.0),
        value("mean_iou", -1.0),
        -value("false_positive_rate", 1.0),
        value("recall", -1.0),
    )


def write_sweep_summary(path, rows):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SWEEP_COLUMNS)
        writer.writeheader()
        for row in rows:
            output = {}
            for column in SWEEP_COLUMNS:
                value = row.get(column, "")
                if isinstance(value, float):
                    value = format_number(value)
                output[column] = value
            writer.writerow(output)
    return output_path
