#!/usr/bin/env python3

import csv
import json
import os
import sys
import threading
import time
from collections import deque

import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from bev_obstacle_detection.msg import Detection3DArray
from bev_obstacle_detection.bev_features import point_cloud_to_bev_feature_maps
from bev_obstacle_detection.bev_grid import BevGridSpec
from bev_obstacle_detection.adaptive_sync_wait import (
    AdaptiveSyncWaitController,
    snapshot_ready_for_early_processing,
)
from bev_obstacle_detection.candidate_adaptive_fusion import (
    adapt_obstacle_candidates,
    occupancy_for_candidates,
)
from bev_obstacle_detection.candidate_temporal_confirmation import (
    CandidateTemporalConfirmationTracker,
)
from bev_obstacle_detection.camera_bev_projection import (
    calibrated_camera_support_from_lidar_points,
    load_camera_calibration,
    semantic_mask_ground_support,
)
from bev_obstacle_detection.camera_semantic_candidates import (
    camera_semantic_candidates,
)
from bev_obstacle_detection.fusion_core import health_score
from bev_obstacle_detection.fusion_latency import compute_fusion_latency
from bev_obstacle_detection.hierarchical_candidate_gate import (
    HierarchicalMultisourceCandidateGate,
)
from bev_obstacle_detection.external_detector_candidates import (
    candidate_occupancy_from_obstacles,
    detection_message_to_obstacles,
    exact_candidate_evidence,
    prefilter_independent_candidates,
)
from bev_obstacle_detection.multimodal_support_fusion import (
    CausalRelativeDensityTracker,
    TemporalModalityReliabilityController,
    adaptive_support_weights,
    calibrated_fusion_score,
    camera_semantic_reliability_score,
    camera_support_score_image,
    camera_support_from_body_points,
    adaptive_radar_consistency_mask,
    fuse_bev_support_maps,
    image_quality_score,
    lidar_quality_score,
    radar_quality_score,
    radar_intensity_inlier_mask,
    radar_spatial_inlier_mask,
    radar_temporal_inlier_mask,
    reliability_gated_temporal_support,
    robust_image_quality_score,
    semantic_mask_quality_score,
    support_from_points,
)
from bev_obstacle_detection.online_candidate_gate import (
    OnlineCandidateReliabilityGate,
)
from bev_obstacle_detection.multimodal_sync_quality import (
    SyncQualityConfig,
    timestamp_alignment_quality,
)
from bev_obstacle_detection.obstacle_detection import (
    detect_occupied_obstacles,
    gate_occupancy_by_support,
    merge_obstacle_candidates,
    parse_distance_min_cells_rules,
    parse_distance_value_rules,
)
from bev_obstacle_detection.point_filters import (
    filter_points_by_height,
    filter_points_by_lateral_range_median,
    lateral_range_deviation_score,
)
from bev_obstacle_detection.ros_adapters import (
    image_to_gray_array,
    odometry_to_transform,
    pointcloud2_to_xyz_array,
    pointcloud2_to_xyz_intensity_arrays,
    transform_point_cloud_xyz,
)
from bev_obstacle_detection.radar_surface_candidates import radar_surface_candidates
from bev_obstacle_detection.sync_buffer import MessageCache, MonotonicStampGuard
from bev_obstacle_detection.temporal_fusion import TemporalPointCloudBuffer
from bev_obstacle_detection.vins_reliability import vins_motion_quality


class MultimodalBevNode:
    def __init__(self):
        self.cache = MessageCache(max_age_sec=rospy.get_param("~max_sensor_age", 0.25))
        required_modalities = rospy.get_param(
            "~required_modalities",
            ["camera", "lidar", "radar", "odom"],
        )
        point_cloud_modalities = rospy.get_param("~point_cloud_modalities", ["lidar", "radar"])
        self.frame_id = rospy.get_param("~bev_frame_id", "iris_0/base_link")
        self.method = rospy.get_param("~method", "bev_vins_fusion")
        self.candidate_source = str(
            rospy.get_param("~candidate_source", "bev_connected_components")
        ).strip().lower()
        if self.candidate_source not in (
            "bev_connected_components",
            "external_detector",
        ):
            raise ValueError(
                "candidate_source must be bev_connected_components or external_detector"
            )
        self.external_detector_min_score = float(
            rospy.get_param("~external_detector_min_score", 0.0)
        )
        if not 0.0 <= self.external_detector_min_score <= 1.0:
            raise ValueError("external detector minimum score must be in [0, 1]")
        self.external_lidar_point_saturation = int(
            rospy.get_param("~external_lidar_point_saturation", 8)
        )
        self.external_radar_point_saturation = int(
            rospy.get_param("~external_radar_point_saturation", 2)
        )
        self.external_camera_coverage_reference = float(
            rospy.get_param("~external_camera_coverage_reference", 0.05)
        )
        self.external_lidar_vertical_margin = float(
            rospy.get_param("~external_lidar_vertical_margin", 0.5)
        )
        self.external_independent_candidate_topk = int(
            rospy.get_param(
                "~external_independent_candidate_topk_per_modality",
                0,
            )
        )
        self.external_independent_candidate_nms_iou = float(
            rospy.get_param(
                "~external_independent_candidate_nms_iou",
                0.30,
            )
        )
        if (
            self.external_lidar_point_saturation <= 0
            or self.external_radar_point_saturation <= 0
            or self.external_camera_coverage_reference <= 0.0
            or self.external_lidar_vertical_margin < 0.0
            or self.external_independent_candidate_topk < 0
            or not 0.0
            <= self.external_independent_candidate_nms_iou
            <= 1.0
        ):
            raise ValueError("external candidate evidence scales are invalid")
        self.point_cloud_transform = rospy.get_param("~point_cloud_transform", "none")
        self.point_cloud_transform_by_modality = {
            "lidar": rospy.get_param("~lidar_point_cloud_transform", self.point_cloud_transform),
            "radar": rospy.get_param("~radar_point_cloud_transform", self.point_cloud_transform),
        }
        self.grid_spec = BevGridSpec(
            x_min=rospy.get_param("~x_min", -10.0),
            x_max=rospy.get_param("~x_max", 30.0),
            y_min=rospy.get_param("~y_min", -20.0),
            y_max=rospy.get_param("~y_max", 20.0),
            z_min=rospy.get_param("~z_min", -2.0),
            z_max=rospy.get_param("~z_max", 5.0),
            resolution=rospy.get_param("~resolution", 0.2),
        )
        self.min_obstacle_cells = int(rospy.get_param("~min_obstacle_cells", 3))
        self.min_obstacle_cells_by_distance = parse_distance_min_cells_rules(
            rospy.get_param("~min_obstacle_cells_by_distance", "")
        )
        self.min_occupancy_weight = float(rospy.get_param("~min_occupancy_weight", 0.0))
        self.min_current_support_ratio = float(rospy.get_param("~min_current_support_ratio", 0.0))
        self.current_support_dilation_cells = int(float(rospy.get_param("~current_support_dilation_cells", 0)))
        self.use_current_support_gate = bool(rospy.get_param("~use_current_support_gate", False))
        self.use_dual_obstacle_fusion = bool(rospy.get_param("~use_dual_obstacle_fusion", False))
        self.use_support_component_split = bool(rospy.get_param("~use_support_component_split", False))
        self.dual_fusion_iou_threshold = float(rospy.get_param("~dual_fusion_iou_threshold", 0.3))
        self.min_point_z = self._optional_float_param("~min_point_z")
        self.max_point_z = self._optional_float_param("~max_point_z")
        self.use_range_median_filter = bool(rospy.get_param("~use_range_median_filter", False))
        self.use_auto_range_median_filter = bool(rospy.get_param("~use_auto_range_median_filter", False))
        self.range_filter_lateral_resolution = float(rospy.get_param("~range_filter_lateral_resolution", 0.2))
        self.range_filter_max_deviation = float(rospy.get_param("~range_filter_max_deviation", 0.25))
        self.range_filter_min_points = int(rospy.get_param("~range_filter_min_points", 3))
        self.auto_range_filter_score_quantile = float(rospy.get_param("~auto_range_filter_score_quantile", 0.5))
        self.auto_range_filter_threshold = float(rospy.get_param("~auto_range_filter_threshold", 0.095))
        self.auto_range_filter_min_points = int(rospy.get_param("~auto_range_filter_min_points", 5))
        self.auto_range_filter_window_frames = int(rospy.get_param("~auto_range_filter_window_frames", 5))
        self.auto_range_filter_min_noisy_frames = int(rospy.get_param("~auto_range_filter_min_noisy_frames", 4))
        self._auto_range_filter_history = deque(maxlen=max(1, self.auto_range_filter_window_frames))
        self.use_adaptive_ground_filter = bool(rospy.get_param("~use_adaptive_ground_filter", False))
        self.adaptive_ground_quantile = float(rospy.get_param("~adaptive_ground_quantile", 0.05))
        self.adaptive_ground_clearance = float(rospy.get_param("~adaptive_ground_clearance", 0.12))
        self.adaptive_ground_min_points = int(rospy.get_param("~adaptive_ground_min_points", 50))
        self.max_points_per_frame = int(rospy.get_param("~max_points_per_frame", 0))
        self.max_obstacle_size_x = self._optional_positive_param("~max_obstacle_size_x")
        self.max_obstacle_size_y = self._optional_positive_param("~max_obstacle_size_y")
        self.max_obstacle_size_x_by_distance = parse_distance_value_rules(
            rospy.get_param("~max_obstacle_size_x_by_distance", ""), "max_size_x"
        )
        self.max_obstacle_size_y_by_distance = parse_distance_value_rules(
            rospy.get_param("~max_obstacle_size_y_by_distance", ""), "max_size_y"
        )
        self.max_obstacle_aspect_ratio = self._optional_positive_param("~max_obstacle_aspect_ratio")
        self.max_obstacle_aspect_ratio_by_distance = parse_distance_value_rules(
            rospy.get_param("~max_obstacle_aspect_ratio_by_distance", ""), "max_aspect_ratio"
        )
        self.use_adaptive_multimodal_fusion = bool(rospy.get_param("~use_adaptive_multimodal_fusion", False))
        self.use_multimodal_sync_quality = bool(
            rospy.get_param("~use_multimodal_sync_quality", True)
        )
        self.multimodal_sync_config = SyncQualityConfig(
            clean_offset_sec=float(
                rospy.get_param("~multimodal_sync_clean_offset_sec", 0.02)
            ),
            rejected_offset_sec=float(
                rospy.get_param(
                    "~multimodal_sync_rejected_offset_sec",
                    0.08,
                )
            ),
            minimum_quality=float(
                rospy.get_param("~multimodal_sync_minimum_quality", 0.0)
            ),
        )
        self.multimodal_sync_config.validate()
        self.use_candidate_adaptive_fusion = bool(
            rospy.get_param("~use_candidate_adaptive_fusion", False)
        )
        self.candidate_reliability_threshold = float(
            rospy.get_param("~candidate_reliability_threshold", 0.0)
        )
        self.candidate_evidence_mix = float(
            rospy.get_param("~candidate_evidence_mix", 0.35)
        )
        self.candidate_quality_mix = float(
            rospy.get_param("~candidate_quality_mix", 0.0)
        )
        self.candidate_local_support_floor = float(
            rospy.get_param("~candidate_local_support_floor", 0.05)
        )
        self.candidate_local_support_exponent = float(
            rospy.get_param("~candidate_local_support_exponent", 1.0)
        )
        self.candidate_participation_min_quality = float(
            rospy.get_param("~candidate_participation_min_quality", 0.05)
        )
        self.candidate_participation_min_support = float(
            rospy.get_param("~candidate_participation_min_support", 0.05)
        )
        self.use_candidate_temporal_confirmation = bool(
            rospy.get_param("~use_candidate_temporal_confirmation", False)
        )
        self.candidate_temporal_match_distance = float(
            rospy.get_param("~candidate_temporal_match_distance", 1.0)
        )
        self.candidate_temporal_grace_frames = int(
            rospy.get_param("~candidate_temporal_grace_frames", 1)
        )
        self.candidate_temporal_max_unconfirmed_frames = int(
            rospy.get_param(
                "~candidate_temporal_max_unconfirmed_frames",
                3,
            )
        )
        self.candidate_temporal_unconfirmed_decay = float(
            rospy.get_param("~candidate_temporal_unconfirmed_decay", 0.75)
        )
        self.candidate_temporal_max_track_age = float(
            rospy.get_param("~candidate_temporal_max_track_age", 1.0)
        )
        self.candidate_temporal_min_active_modalities = int(
            rospy.get_param(
                "~candidate_temporal_min_active_modalities",
                1,
            )
        )
        self.candidate_temporal_require_initial_confirmation = bool(
            rospy.get_param(
                "~candidate_temporal_require_initial_confirmation",
                True,
            )
        )
        self.candidate_temporal_failsafe_quality_threshold = float(
            rospy.get_param(
                "~candidate_temporal_failsafe_quality_threshold",
                0.20,
            )
        )
        if not (
            0.0
            <= self.candidate_temporal_failsafe_quality_threshold
            <= 1.0
        ):
            raise ValueError(
                "candidate temporal fail-safe quality threshold must be in [0, 1]"
            )
        if self.use_candidate_adaptive_fusion and not self.use_adaptive_multimodal_fusion:
            raise ValueError(
                "candidate adaptive fusion requires adaptive multimodal fusion"
            )
        if (
            self.use_candidate_temporal_confirmation
            and not self.use_candidate_adaptive_fusion
        ):
            raise ValueError(
                "candidate temporal confirmation requires candidate adaptive fusion"
            )
        self.use_learned_candidate_gate = bool(
            rospy.get_param("~use_learned_candidate_gate", False)
        )
        self.learned_candidate_gate_model_path = str(
            rospy.get_param("~learned_candidate_gate_model_path", "")
        ).strip()
        self.learned_candidate_gate = None
        if self.use_learned_candidate_gate:
            if not self.use_candidate_adaptive_fusion:
                raise ValueError(
                    "learned candidate gate requires candidate adaptive fusion"
                )
            if not self.learned_candidate_gate_model_path:
                raise ValueError(
                    "learned candidate gate model path is required"
                )
            self.learned_candidate_gate = (
                OnlineCandidateReliabilityGate.from_json(
                    os.path.expanduser(
                        self.learned_candidate_gate_model_path
                    )
                )
            )
        self.use_hierarchical_candidate_gate = bool(
            rospy.get_param("~use_hierarchical_candidate_gate", False)
        )
        self.hierarchical_candidate_gate_model_path = str(
            rospy.get_param(
                "~hierarchical_candidate_gate_model_path",
                "",
            )
        ).strip()
        self.hierarchical_candidate_gate = None
        if (
            self.use_hierarchical_candidate_gate
            and self.use_learned_candidate_gate
        ):
            raise ValueError(
                "learned and hierarchical candidate gates are mutually exclusive"
            )
        if self.use_hierarchical_candidate_gate:
            if (
                not self.use_candidate_adaptive_fusion
                or self.candidate_source != "external_detector"
            ):
                raise ValueError(
                    "hierarchical gate requires adaptive external candidates"
                )
            if not self.hierarchical_candidate_gate_model_path:
                raise ValueError(
                    "hierarchical candidate gate model path is required"
                )
            self.hierarchical_candidate_gate = (
                HierarchicalMultisourceCandidateGate.from_json(
                    os.path.expanduser(
                        self.hierarchical_candidate_gate_model_path
                    )
                )
            )
        self.radar_in_base_occupancy = bool(rospy.get_param("~radar_in_base_occupancy", True))
        self.use_temporal_modality_reliability = bool(
            rospy.get_param("~use_temporal_modality_reliability", False)
        )
        self.allow_modality_dropout = bool(rospy.get_param("~allow_modality_dropout", False))
        self.process_on_lidar_callback = bool(rospy.get_param("~process_on_lidar_callback", False))
        self.lidar_processing_delay_sec = max(
            0.0,
            float(rospy.get_param("~lidar_processing_delay_sec", 0.02)),
        )
        self.use_adaptive_sync_wait = bool(
            rospy.get_param("~use_adaptive_sync_wait", False)
        )
        self.adaptive_sync_wait_modalities = list(
            rospy.get_param(
                "~adaptive_sync_wait_modalities",
                ["camera", "semantic_mask", "radar"],
            )
        )
        self.sync_wait_controller = None
        self.adaptive_sync_wait_poll_sec = max(
            0.0005,
            float(
                rospy.get_param(
                    "~adaptive_sync_wait_poll_sec",
                    0.002,
                )
            ),
        )
        if self.use_adaptive_sync_wait:
            self.sync_wait_controller = AdaptiveSyncWaitController(
                initial_wait_sec=self.lidar_processing_delay_sec,
                min_wait_sec=float(
                    rospy.get_param("~adaptive_sync_wait_min_sec", 0.01)
                ),
                max_wait_sec=float(
                    rospy.get_param("~adaptive_sync_wait_max_sec", 0.04)
                ),
                increase_step_sec=float(
                    rospy.get_param(
                        "~adaptive_sync_wait_increase_step_sec",
                        0.005,
                    )
                ),
                decrease_step_sec=float(
                    rospy.get_param(
                        "~adaptive_sync_wait_decrease_step_sec",
                        0.002,
                    )
                ),
                complete_frames_before_decrease=int(
                    rospy.get_param(
                        "~adaptive_sync_wait_complete_frames",
                        3,
                    )
                ),
            )
        self.reject_out_of_order_lidar = bool(
            rospy.get_param("~reject_out_of_order_lidar", True)
        )
        self.lidar_trigger_stamp_epsilon_sec = max(
            0.0,
            float(rospy.get_param("~lidar_trigger_stamp_epsilon_sec", 1e-9)),
        )
        default_sensor_queue_size = 20 if self.process_on_lidar_callback else 1
        self.sensor_queue_size = max(
            1,
            int(rospy.get_param("~sensor_queue_size", default_sensor_queue_size)),
        )
        self.cache.max_messages_per_modality = self.sensor_queue_size
        self.lidar_trigger_guard = MonotonicStampGuard(
            epsilon_sec=self.lidar_trigger_stamp_epsilon_sec
        )
        self._trigger_stats = {
            "accepted": 0,
            "duplicate": 0,
            "out_of_order": 0,
            "missing_required": 0,
        }
        self._active_sync_reference = None
        self._active_sync_snapshot = None
        self.multimodal_fusion_threshold = float(rospy.get_param("~multimodal_fusion_threshold", 0.5))
        self.lidar_fusion_prior = float(rospy.get_param("~lidar_fusion_prior", 1.0))
        self.camera_fusion_prior = float(rospy.get_param("~camera_fusion_prior", 0.5))
        self.radar_fusion_prior = float(rospy.get_param("~radar_fusion_prior", 0.75))
        self.min_lidar_fusion_weight = float(rospy.get_param("~min_lidar_fusion_weight", 0.2))
        self.lidar_quality_saturation_points = int(rospy.get_param("~lidar_quality_saturation_points", 5000))
        self.lidar_quality_range_deviation_quantile = float(
            rospy.get_param(
                "~lidar_quality_range_deviation_quantile",
                self.auto_range_filter_score_quantile,
            )
        )
        self.lidar_quality_clean_range_deviation = float(
            rospy.get_param("~lidar_quality_clean_range_deviation", 0.07)
        )
        self.lidar_quality_degraded_range_deviation = float(
            rospy.get_param("~lidar_quality_degraded_range_deviation", 0.16)
        )
        self.lidar_quality_min_noise_quality = float(
            rospy.get_param("~lidar_quality_min_noise_quality", 0.10)
        )
        self.use_relative_lidar_density_quality = bool(
            rospy.get_param("~use_relative_lidar_density_quality", True)
        )
        self.relative_lidar_density_tracker = CausalRelativeDensityTracker(
            envelope_decay=rospy.get_param(
                "~relative_lidar_density_envelope_decay", 0.999
            ),
            warmup_frames=rospy.get_param(
                "~relative_lidar_density_warmup_frames", 10
            ),
        )
        self.camera_semantic_image_quality_floor = float(
            rospy.get_param("~camera_semantic_image_quality_floor", 0.25)
        )
        self.camera_sharpness_degraded_laplacian = float(
            rospy.get_param("~camera_sharpness_degraded_laplacian", 0.20)
        )
        self.camera_sharpness_clean_laplacian = float(
            rospy.get_param("~camera_sharpness_clean_laplacian", 0.80)
        )
        self.camera_noise_clean_laplacian = float(
            rospy.get_param("~camera_noise_clean_laplacian", 4.0)
        )
        self.camera_noise_degraded_laplacian = float(
            rospy.get_param("~camera_noise_degraded_laplacian", 20.0)
        )
        self.camera_image_min_degradation_quality = float(
            rospy.get_param("~camera_image_min_degradation_quality", 0.05)
        )
        self.modality_reliability_controller = TemporalModalityReliabilityController(
            rise_alpha=float(rospy.get_param("~modality_reliability_rise_alpha", 0.20)),
            fall_alpha=float(rospy.get_param("~modality_reliability_fall_alpha", 0.70)),
            failure_threshold=float(rospy.get_param("~modality_failure_threshold", 0.05)),
            recovery_threshold=float(rospy.get_param("~modality_recovery_threshold", 0.12)),
            use_smoothed_weights=bool(rospy.get_param("~modality_use_smoothed_weights", False)),
        )
        self._last_modality_qualities = {}
        self._last_frame_modality_qualities = {}
        self._last_modality_weights = {}
        self._last_reliability_state = "uninitialized"
        self.camera_horizontal_fov_deg = float(rospy.get_param("~camera_horizontal_fov_deg", 69.0))
        self.camera_vertical_fov_deg = float(rospy.get_param("~camera_vertical_fov_deg", 42.0))
        self.camera_support_min_depth_m = float(rospy.get_param("~camera_support_min_depth_m", 0.2))
        self.camera_support_dilation_cells = int(float(rospy.get_param("~camera_support_dilation_cells", 1)))
        self.camera_support_mode = rospy.get_param("~camera_support_mode", "intensity")
        self.camera_edge_percentile = float(rospy.get_param("~camera_edge_percentile", 95.0))
        self.camera_support_edge_weight = float(rospy.get_param("~camera_support_edge_weight", 0.6))
        self.camera_support_min_score = float(rospy.get_param("~camera_support_min_score", 0.0))
        self.camera_support_projection_mode = str(
            rospy.get_param("~camera_support_projection_mode", "fov")
        ).strip().lower()
        self.camera_projection_matrix_path = rospy.get_param(
            "~camera_projection_matrix_path", ""
        )
        self.camera_lidar_extrinsic_path = rospy.get_param(
            "~camera_lidar_extrinsic_path", ""
        )
        self.camera_projection_matrix = None
        self.camera_lidar_extrinsic = None
        if self.camera_projection_matrix_path or self.camera_lidar_extrinsic_path:
            if not (
                self.camera_projection_matrix_path
                and self.camera_lidar_extrinsic_path
            ):
                raise ValueError(
                    "both camera projection and LiDAR extrinsic paths are required"
                )
            (
                self.camera_projection_matrix,
                self.camera_lidar_extrinsic,
            ) = load_camera_calibration(
                self.camera_projection_matrix_path,
                self.camera_lidar_extrinsic_path,
            )
        if (
            self.camera_support_projection_mode == "calibrated"
            and self.camera_projection_matrix is None
        ):
            raise ValueError("calibrated Camera support requires calibration files")
        self.use_camera_ground_ipm = bool(
            rospy.get_param("~use_camera_ground_ipm", False)
        )
        self.camera_ground_z = float(rospy.get_param("~camera_ground_z", -1.5))
        self.camera_ground_ipm_min_range = float(
            rospy.get_param("~camera_ground_ipm_min_range", 1.0)
        )
        self.camera_ground_ipm_max_range = float(
            rospy.get_param("~camera_ground_ipm_max_range", 30.0)
        )
        self.camera_ground_ipm_min_component_area_px = int(
            rospy.get_param("~camera_ground_ipm_min_component_area_px", 128)
        )
        self.camera_ground_ipm_max_component_fraction = float(
            rospy.get_param("~camera_ground_ipm_max_component_fraction", 0.15)
        )
        self.camera_ground_ipm_reject_border_components = bool(
            rospy.get_param(
                "~camera_ground_ipm_reject_border_components", True
            )
        )
        self.camera_ground_ipm_column_stride_px = int(
            rospy.get_param("~camera_ground_ipm_column_stride_px", 8)
        )
        self.camera_ground_ipm_bottom_band_px = int(
            rospy.get_param("~camera_ground_ipm_bottom_band_px", 3)
        )
        self.camera_ground_ipm_dilation_cells = int(
            rospy.get_param("~camera_ground_ipm_dilation_cells", 2)
        )
        self.camera_ground_ipm_min_semantic_quality = float(
            rospy.get_param("~camera_ground_ipm_min_semantic_quality", 0.35)
        )
        self._last_camera_semantic_mask = None
        self._last_camera_raw_image_quality = 0.0
        self._last_camera_ground_components = 0
        self._last_camera_ground_points = 0
        self.semantic_mask_min_score = float(rospy.get_param("~semantic_mask_min_score", 0.5))
        self.semantic_mask_min_coverage = float(rospy.get_param("~semantic_mask_min_coverage", 0.001))
        self.semantic_mask_saturation_coverage = float(rospy.get_param("~semantic_mask_saturation_coverage", 0.12))
        self.semantic_mask_overcoverage_start = float(rospy.get_param("~semantic_mask_overcoverage_start", 0.5))
        self.semantic_mask_max_coverage = float(rospy.get_param("~semantic_mask_max_coverage", 0.9))
        self.semantic_mask_overcoverage_min_quality = float(
            rospy.get_param("~semantic_mask_overcoverage_min_quality", 0.05)
        )
        self.use_reliability_gated_camera_candidates = bool(
            rospy.get_param("~use_reliability_gated_camera_candidates", False)
        )
        self.force_camera_candidate_generation = bool(
            rospy.get_param("~force_camera_candidate_generation", False)
        )
        self.camera_candidate_max_lidar_quality = float(
            rospy.get_param("~camera_candidate_max_lidar_quality", 0.12)
        )
        self.camera_candidate_max_radar_quality = float(
            rospy.get_param("~camera_candidate_max_radar_quality", 0.12)
        )
        self.camera_candidate_min_image_quality = float(
            rospy.get_param("~camera_candidate_min_image_quality", 0.10)
        )
        self.camera_candidate_min_semantic_quality = float(
            rospy.get_param("~camera_candidate_min_semantic_quality", 0.10)
        )
        self.camera_candidate_horizontal_fov_deg = float(
            rospy.get_param("~camera_candidate_horizontal_fov_deg", 90.0)
        )
        self.camera_candidate_vertical_fov_deg = float(
            rospy.get_param("~camera_candidate_vertical_fov_deg", 65.0)
        )
        self.camera_candidate_assumed_height = float(
            rospy.get_param("~camera_candidate_assumed_height", 1.5)
        )
        self.camera_candidate_min_component_area_px = int(
            rospy.get_param("~camera_candidate_min_component_area_px", 32)
        )
        self.camera_candidate_max_component_fraction = float(
            rospy.get_param("~camera_candidate_max_component_fraction", 0.20)
        )
        self.camera_candidate_min_pixel_height = int(
            rospy.get_param("~camera_candidate_min_pixel_height", 8)
        )
        self.camera_candidate_min_range = float(
            rospy.get_param("~camera_candidate_min_range", 1.0)
        )
        self.camera_candidate_max_range = float(
            rospy.get_param("~camera_candidate_max_range", 12.0)
        )
        self.camera_candidate_radial_size = float(
            rospy.get_param("~camera_candidate_radial_size", 1.2)
        )
        self.camera_candidate_min_lateral_size = float(
            rospy.get_param("~camera_candidate_min_lateral_size", 0.4)
        )
        self.camera_candidate_max_lateral_size = float(
            rospy.get_param("~camera_candidate_max_lateral_size", 1.8)
        )
        self.camera_candidate_lateral_padding = float(
            rospy.get_param("~camera_candidate_lateral_padding", 0.15)
        )
        self.camera_candidate_merge_iou = float(
            rospy.get_param("~camera_candidate_merge_iou", 0.30)
        )
        for name, value in (
            ("camera_candidate_max_lidar_quality", self.camera_candidate_max_lidar_quality),
            ("camera_candidate_max_radar_quality", self.camera_candidate_max_radar_quality),
            ("camera_candidate_min_image_quality", self.camera_candidate_min_image_quality),
            ("camera_candidate_min_semantic_quality", self.camera_candidate_min_semantic_quality),
            ("camera_candidate_merge_iou", self.camera_candidate_merge_iou),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("{} must be in [0, 1]".format(name))
        self._last_camera_semantic_quality = 0.0
        self._last_camera_semantic_candidate_count = 0
        self.use_camera_semantic_gate = bool(rospy.get_param("~use_camera_semantic_gate", False))
        self.camera_semantic_gate_dilation_cells = int(float(rospy.get_param("~camera_semantic_gate_dilation_cells", 1)))
        self.radar_support_dilation_cells = int(float(rospy.get_param("~radar_support_dilation_cells", 2)))
        self.radar_quality_min_points = int(rospy.get_param("~radar_quality_min_points", 3))
        self.radar_quality_saturation_points = int(rospy.get_param("~radar_quality_saturation_points", 64))
        self.use_radar_spatial_consistency = bool(
            rospy.get_param("~use_radar_spatial_consistency", True)
        )
        self.radar_spatial_neighbor_radius = float(
            rospy.get_param("~radar_spatial_neighbor_radius", 0.75)
        )
        self.radar_spatial_min_neighbors = int(
            rospy.get_param("~radar_spatial_min_neighbors", 2)
        )
        self.radar_min_spatial_quality = float(
            rospy.get_param("~radar_min_spatial_quality", 0.25)
        )
        self.filter_radar_spatial_outliers = bool(
            rospy.get_param("~filter_radar_spatial_outliers", True)
        )
        self.use_radar_temporal_consistency = bool(
            rospy.get_param("~use_radar_temporal_consistency", False)
        )
        self.radar_temporal_neighbor_radius = float(
            rospy.get_param("~radar_temporal_neighbor_radius", 0.75)
        )
        self.radar_temporal_min_neighbors = int(
            rospy.get_param("~radar_temporal_min_neighbors", 1)
        )
        self.use_adaptive_radar_temporal_fallback = bool(
            rospy.get_param("~use_adaptive_radar_temporal_fallback", False)
        )
        self.radar_temporal_fallback_min_inliers = int(
            rospy.get_param("~radar_temporal_fallback_min_inliers", 3)
        )
        self.radar_temporal_fallback_min_ratio = float(
            rospy.get_param("~radar_temporal_fallback_min_ratio", 0.01)
        )
        self.radar_temporal_fallback_min_spatial_ratio = float(
            rospy.get_param(
                "~radar_temporal_fallback_min_spatial_ratio", 0.80
            )
        )
        self.radar_temporal_fallback_quality_scale = float(
            rospy.get_param("~radar_temporal_fallback_quality_scale", 0.70)
        )
        self._last_radar_spatial_inlier_ratio = 1.0
        self._last_radar_temporal_inlier_ratio = 1.0
        self._last_radar_temporal_fallback = False
        self.use_reliability_gated_radar_memory = bool(
            rospy.get_param("~use_reliability_gated_radar_memory", False)
        )
        self.radar_memory_max_lidar_quality = float(
            rospy.get_param("~radar_memory_max_lidar_quality", 0.12)
        )
        self.radar_memory_support_scale = float(
            rospy.get_param("~radar_memory_support_scale", 0.60)
        )
        if not 0.0 <= self.radar_memory_max_lidar_quality <= 1.0:
            raise ValueError("radar_memory_max_lidar_quality must be in [0, 1]")
        if not 0.0 <= self.radar_memory_support_scale <= 1.0:
            raise ValueError("radar_memory_support_scale must be in [0, 1]")
        self._last_radar_memory_active = False
        self._last_radar_memory_points = 0
        self._last_radar_memory_stamp = None
        self.use_reliability_gated_radar_candidates = bool(
            rospy.get_param("~use_reliability_gated_radar_candidates", False)
        )
        self.force_radar_candidate_generation = bool(
            rospy.get_param("~force_radar_candidate_generation", False)
        )
        self.radar_candidate_max_lidar_quality = float(
            rospy.get_param("~radar_candidate_max_lidar_quality", 0.12)
        )
        self.radar_candidate_cluster_radius = float(
            rospy.get_param("~radar_candidate_cluster_radius", 0.65)
        )
        self.radar_candidate_min_points = int(
            rospy.get_param("~radar_candidate_min_points", 2)
        )
        self.radar_candidate_radial_extent = float(
            rospy.get_param("~radar_candidate_radial_extent", 0.8)
        )
        self.radar_candidate_padding = float(
            rospy.get_param("~radar_candidate_padding", 0.15)
        )
        self.radar_candidate_max_surface_span = float(
            rospy.get_param("~radar_candidate_max_surface_span", 1.5)
        )
        self.radar_candidate_max_range = float(
            rospy.get_param("~radar_candidate_max_range", 10.0)
        )
        self.radar_candidate_merge_iou = float(
            rospy.get_param("~radar_candidate_merge_iou", 0.30)
        )
        if not 0.0 <= self.radar_candidate_max_lidar_quality <= 1.0:
            raise ValueError("radar_candidate_max_lidar_quality must be in [0, 1]")
        if not 0.0 <= self.radar_candidate_merge_iou <= 1.0:
            raise ValueError("radar_candidate_merge_iou must be in [0, 1]")
        self._last_radar_surface_candidate_count = 0
        self._last_camera_semantic_candidate_count = 0
        self._last_radar_candidate_gate_active = False
        self._last_camera_candidate_gate_active = False
        self._last_lidar_range_deviation = 0.0
        self.use_radar_intensity_filter = bool(
            rospy.get_param("~use_radar_intensity_filter", False)
        )
        self.radar_intensity_field = rospy.get_param("~radar_intensity_field", "intensity")
        self.radar_min_intensity = float(rospy.get_param("~radar_min_intensity", 0.30))
        self._last_radar_intensity_inlier_ratio = 1.0
        self._last_radar_history_stamp = None
        self._last_radar_temporal_mask = None
        self.connectivity = int(rospy.get_param("~connectivity", 8))
        self.publish_grid = bool(rospy.get_param("~publish_grid", True))
        self.publish_markers = bool(rospy.get_param("~publish_markers", True))
        self.use_vins_temporal_alignment = bool(
            rospy.get_param("~use_vins_temporal_alignment", self.method == "bev_vins_fusion")
        )
        self.use_vins_motion_consistency = bool(
            rospy.get_param("~use_vins_motion_consistency", False)
        )
        self.vins_clean_speed_mps = float(
            rospy.get_param("~vins_clean_speed_mps", 10.0)
        )
        self.vins_rejected_speed_mps = float(
            rospy.get_param("~vins_rejected_speed_mps", 20.0)
        )
        self.vins_clean_angular_rate_rps = float(
            rospy.get_param("~vins_clean_angular_rate_rps", 0.8)
        )
        self.vins_rejected_angular_rate_rps = float(
            rospy.get_param("~vins_rejected_angular_rate_rps", 1.5)
        )
        self.vins_min_alignment_quality = float(
            rospy.get_param("~vins_min_alignment_quality", 0.2)
        )
        self._previous_vins_observation = None
        self._cached_vins_stamp = None
        self._cached_vins_transform = None
        self._last_vins_quality = 0.0
        self._last_vins_used = False
        self._last_fusion_confidence_scale = 0.0
        self._last_candidate_support_maps = {}
        self._last_candidate_radar_support_points = np.empty(
            (0, 3),
            dtype=float,
        )
        self._last_candidate_radar_support_intensities = np.empty(
            0,
            dtype=float,
        )
        self._last_modality_sync_offsets = {}
        self._last_modality_sync_qualities = {}
        self._last_pipeline_timings_ms = {}
        self._last_external_candidate_prefilter_stats = {
            "input_count": 0,
            "kept_count": 0,
            "dropped_count": 0,
            "independent_input_count": 0,
            "independent_kept_count": 0,
            "independent_dropped_count": 0,
        }
        self._last_adaptive_detail_timings_ms = {}
        self.candidate_temporal_tracker = CandidateTemporalConfirmationTracker(
            match_distance=self.candidate_temporal_match_distance,
            grace_frames=self.candidate_temporal_grace_frames,
            max_unconfirmed_frames=(
                self.candidate_temporal_max_unconfirmed_frames
            ),
            unconfirmed_decay=self.candidate_temporal_unconfirmed_decay,
            max_track_age_sec=self.candidate_temporal_max_track_age,
            min_active_modalities=(
                self.candidate_temporal_min_active_modalities
            ),
            require_initial_confirmation=(
                self.candidate_temporal_require_initial_confirmation
            ),
        )
        self.detection_csv_path = rospy.get_param("~detection_csv_path", "")
        self.reliability_csv_path = rospy.get_param("~reliability_csv_path", "")
        self.candidate_reliability_csv_path = rospy.get_param(
            "~candidate_reliability_csv_path", ""
        )
        self.sync_diagnostics_csv_path = rospy.get_param(
            "~sync_diagnostics_csv_path", ""
        )
        self.fusion_latency_csv_path = rospy.get_param(
            "~fusion_latency_csv_path", ""
        )
        self.fusion_latency_budget_ms = float(
            rospy.get_param("~fusion_latency_budget_ms", 100.0)
        )
        if self.fusion_latency_budget_ms <= 0.0:
            raise ValueError("fusion latency budget must be positive")
        self.temporal_buffer = TemporalPointCloudBuffer(
            max_frames=int(rospy.get_param("~temporal_max_frames", 3)),
            max_age_sec=float(rospy.get_param("~temporal_max_age", 0.5)),
            decay_tau_sec=float(rospy.get_param("~temporal_decay_tau", 0.0)),
        )
        self.radar_temporal_buffer = TemporalPointCloudBuffer(
            max_frames=int(rospy.get_param("~radar_temporal_max_frames", 2)),
            max_age_sec=float(rospy.get_param("~radar_temporal_max_age", 0.5)),
            decay_tau_sec=0.0,
        )
        self.radar_memory_buffer = TemporalPointCloudBuffer(
            max_frames=int(rospy.get_param("~radar_memory_max_frames", 8)),
            max_age_sec=float(rospy.get_param("~radar_memory_max_age", 1.5)),
            decay_tau_sec=float(rospy.get_param("~radar_memory_decay_tau", 0.75)),
        )
        self._csv_stream = None
        self._csv_writer = None
        self._reliability_csv_stream = None
        self._reliability_csv_writer = None
        self._candidate_reliability_csv_stream = None
        self._candidate_reliability_csv_writer = None
        self._sync_diagnostics_csv_stream = None
        self._sync_diagnostics_csv_writer = None
        self._fusion_latency_csv_stream = None
        self._fusion_latency_csv_writer = None
        self._last_processed_stamps = None
        self._shutting_down = False
        self._processing_lock = threading.Lock()

        self.status_pub = rospy.Publisher("~status", String, queue_size=1)
        self.modality_health_pub = rospy.Publisher(
            "~modality_health", String, queue_size=4
        )
        self.grid_pub = rospy.Publisher("~occupancy_grid", OccupancyGrid, queue_size=1, latch=True)
        self.marker_pub = rospy.Publisher("~obstacle_markers", MarkerArray, queue_size=1)

        camera_topic = rospy.get_param("~camera_topic", "/iris_0/stereo_camera/left/image_raw")
        semantic_mask_topic = rospy.get_param("~semantic_mask_topic", "")
        lidar_topic = rospy.get_param("~lidar_topic", "/iris_0/lidar/points")
        radar_topic = rospy.get_param("~radar_topic", "/iris_0/radar/points")
        odom_topic = rospy.get_param("~odom_topic", "/iris_0/vins_estimator/odometry")
        external_detector_topic = rospy.get_param(
            "~external_detector_topic",
            "",
        )
        self.topic_by_modality = {
            "camera": camera_topic,
            "semantic_mask": semantic_mask_topic,
            "lidar": lidar_topic,
            "radar": radar_topic,
            "odom": odom_topic,
            "external_detector": external_detector_topic,
        }
        exact_support_ratios = None
        exact_candidate_diagnostics = {}
        if self.candidate_source == "external_detector":
            if not external_detector_topic:
                raise ValueError(
                    "external_detector_topic is required for external candidates"
                )
            if "external_detector" not in required_modalities:
                required_modalities = list(required_modalities) + [
                    "external_detector"
                ]
        self.adaptive_sync_wait_modalities = [
            modality
            for modality in self.adaptive_sync_wait_modalities
            if self.topic_by_modality.get(modality, "")
        ]
        self.required_modalities = [
            modality for modality in required_modalities if self.topic_by_modality.get(modality, "")
        ]
        if self.allow_modality_dropout and self.use_adaptive_multimodal_fusion:
            optional = {"camera", "semantic_mask", "radar", "odom"}
            self.required_modalities = [
                modality for modality in self.required_modalities if modality not in optional
            ]
        self.point_cloud_modalities = [
            modality for modality in point_cloud_modalities if self.topic_by_modality.get(modality, "")
        ]

        self.subscribers = []
        self._add_subscriber("camera", camera_topic, Image, queue_size=self.sensor_queue_size)
        self._add_subscriber("semantic_mask", semantic_mask_topic, Image, queue_size=self.sensor_queue_size)
        self._add_subscriber("lidar", lidar_topic, PointCloud2, queue_size=self.sensor_queue_size)
        self._add_subscriber("radar", radar_topic, PointCloud2, queue_size=self.sensor_queue_size)
        self._add_subscriber("odom", odom_topic, Odometry, queue_size=max(10, self.sensor_queue_size))
        self._add_subscriber(
            "external_detector",
            external_detector_topic,
            Detection3DArray,
            queue_size=self.sensor_queue_size,
        )

        status_rate = rospy.get_param("~status_rate", 2.0)
        self._open_detection_csv()
        self._open_reliability_csv()
        self._open_candidate_reliability_csv()
        self._open_sync_diagnostics_csv()
        self._open_fusion_latency_csv()
        self.timer = rospy.Timer(rospy.Duration(1.0 / status_rate), self._on_timer)
        rospy.on_shutdown(self._shutdown)

    def _add_subscriber(self, modality, topic, message_type, queue_size):
        if not topic:
            return
        self.subscribers.append(
            rospy.Subscriber(topic, message_type, self._callback, callback_args=modality, queue_size=queue_size)
        )

    @staticmethod
    def _optional_positive_param(name):
        value = float(rospy.get_param(name, 0.0))
        return value if value > 0.0 else None

    @staticmethod
    def _optional_float_param(name):
        value = rospy.get_param(name, "")
        if value in ("", None):
            return None
        return float(value)

    def _callback(self, message, modality):
        trigger_stamp = None
        callback_arrival_monotonic = time.perf_counter()
        with self._processing_lock:
            self.cache.update(modality, message)
            if (
                self.process_on_lidar_callback
                and modality == "lidar"
                and not self._shutting_down
                and not rospy.is_shutdown()
            ):
                trigger_stamp = message.header.stamp
        if trigger_stamp is not None:
            adaptive_wait_deadline_sec = (
                self.sync_wait_controller.current_wait_sec
                if self.sync_wait_controller is not None
                else self.lidar_processing_delay_sec
            )
            configured_sync_wait_sec = (
                self.sync_wait_controller.min_wait_sec
                if self.sync_wait_controller is not None
                else adaptive_wait_deadline_sec
            )
            timer = threading.Timer(
                configured_sync_wait_sec,
                self._process_triggered_lidar,
                args=(
                    trigger_stamp,
                    callback_arrival_monotonic,
                    configured_sync_wait_sec,
                    adaptive_wait_deadline_sec,
                ),
            )
            timer.daemon = True
            timer.start()

    def _process_triggered_lidar(
        self,
        stamp,
        callback_arrival_monotonic=None,
        configured_sync_wait_sec=None,
        adaptive_wait_deadline_sec=None,
    ):
        if self._shutting_down or rospy.is_shutdown():
            return
        with self._processing_lock:
            processing_start_monotonic = time.perf_counter()
            if callback_arrival_monotonic is None:
                callback_arrival_monotonic = processing_start_monotonic
            if configured_sync_wait_sec is None:
                configured_sync_wait_sec = self.lidar_processing_delay_sec
            reference = stamp.to_sec()
            if (
                self.sync_wait_controller is not None
                and adaptive_wait_deadline_sec is not None
            ):
                preview = self.cache.synchronized_snapshot(
                    self.adaptive_sync_wait_modalities,
                    reference,
                )
                ready_early = snapshot_ready_for_early_processing(
                    preview,
                    self.adaptive_sync_wait_modalities,
                    reference,
                    self.multimodal_sync_config.clean_offset_sec,
                )
                elapsed_sec = max(
                    0.0,
                    processing_start_monotonic
                    - callback_arrival_monotonic,
                )
                remaining_sec = (
                    float(adaptive_wait_deadline_sec) - elapsed_sec
                )
                if not ready_early and remaining_sec > 0.0005:
                    next_poll_sec = min(
                        self.adaptive_sync_wait_poll_sec,
                        remaining_sec,
                    )
                    timer = threading.Timer(
                        next_poll_sec,
                        self._process_triggered_lidar,
                        args=(
                            stamp,
                            callback_arrival_monotonic,
                            elapsed_sec + next_poll_sec,
                            adaptive_wait_deadline_sec,
                        ),
                    )
                    timer.daemon = True
                    timer.start()
                    return
                if (
                    not ready_early
                    and any(
                        preview.get(modality) is None
                        for modality in self.adaptive_sync_wait_modalities
                    )
                    and self.sync_wait_controller.current_wait_sec
                    < self.sync_wait_controller.max_wait_sec
                ):
                    adaptive_wait_deadline_sec = (
                        self.sync_wait_controller.observe(False)
                    )
                    remaining_sec = max(
                        0.0,
                        adaptive_wait_deadline_sec - elapsed_sec,
                    )
                    if remaining_sec > 0.0005:
                        next_poll_sec = min(
                            self.adaptive_sync_wait_poll_sec,
                            remaining_sec,
                        )
                        timer = threading.Timer(
                            next_poll_sec,
                            self._process_triggered_lidar,
                            args=(
                                stamp,
                                callback_arrival_monotonic,
                                elapsed_sec + next_poll_sec,
                                adaptive_wait_deadline_sec,
                            ),
                        )
                        timer.daemon = True
                        timer.start()
                        return
                configured_sync_wait_sec = min(
                    elapsed_sec,
                    float(adaptive_wait_deadline_sec),
                )
            if self.reject_out_of_order_lidar:
                accepted, reason = self.lidar_trigger_guard.accept(reference)
                if not accepted:
                    self._trigger_stats[reason] += 1
                    self._write_sync_diagnostics(reference, reason, {})
                    return
            else:
                self._trigger_stats["accepted"] += 1
            if self.reject_out_of_order_lidar:
                self._trigger_stats["accepted"] += 1

            snapshot = self.cache.synchronized_snapshot(
                [
                    name
                    for name, topic in self.topic_by_modality.items()
                    if topic
                ],
                reference,
            )
            if not self.cache.snapshot_is_ready(
                snapshot,
                self.required_modalities,
            ):
                self._trigger_stats["missing_required"] += 1
                self._write_sync_diagnostics(
                    reference,
                    "missing_required",
                    snapshot,
                )
                return
            self._write_sync_diagnostics(reference, "accepted", snapshot)
            sync_inputs_complete = all(
                modality in snapshot
                for modality in self.adaptive_sync_wait_modalities
            )
            next_sync_wait_sec = float(configured_sync_wait_sec)
            if self.sync_wait_controller is not None:
                next_sync_wait_sec = self.sync_wait_controller.observe(
                    sync_inputs_complete
                )
            obstacle_count = self._process_bev(
                stamp,
                synchronized_snapshot=snapshot,
            )
            processing_end_monotonic = time.perf_counter()
            latency = compute_fusion_latency(
                trigger_stamp_sec=reference,
                ros_output_stamp_sec=rospy.Time.now().to_sec(),
                callback_arrival_monotonic=callback_arrival_monotonic,
                processing_start_monotonic=processing_start_monotonic,
                processing_end_monotonic=processing_end_monotonic,
                configured_sync_wait_sec=configured_sync_wait_sec,
                selected_offsets_sec=[
                    item.offset_sec for item in snapshot.values()
                ],
                budget_ms=self.fusion_latency_budget_ms,
            )
            attributed_ms = sum(self._last_pipeline_timings_ms.values())
            latency.update(self._last_pipeline_timings_ms)
            latency.update(self._last_adaptive_detail_timings_ms)
            latency["pipeline_unattributed_ms"] = max(
                0.0,
                latency["processing_ms"] - attributed_ms,
            )
            latency["sync_inputs_complete"] = sync_inputs_complete
            latency["next_sync_wait_ms"] = next_sync_wait_sec * 1000.0
            self._write_fusion_latency(
                reference,
                obstacle_count,
                latency,
            )

    def _on_timer(self, event):
        if self._shutting_down or rospy.is_shutdown():
            return
        with self._processing_lock:
            now = event.current_real.to_sec()
            ready = self.cache.is_ready(self.required_modalities, now)
            ages = []
            for name in self.required_modalities:
                age = self.cache.age(name, now)
                ages.append("{}={}".format(name, "none" if age is None else "{:.3f}s".format(age)))
            reliability = ""
            if self.use_temporal_modality_reliability:
                quality_text = ",".join(
                    "{}:{:.2f}".format(name, value)
                    for name, value in sorted(self._last_modality_qualities.items())
                )
                weight_text = ",".join(
                    "{}:{:.2f}".format(name, value)
                    for name, value in sorted(self._last_modality_weights.items())
                )
                reliability = " state={} q=[{}] w=[{}]".format(
                    self._last_reliability_state,
                    quality_text,
                    weight_text,
                )
            if self.process_on_lidar_callback:
                reliability += (
                    " trigger=[ok:{accepted},dup:{duplicate},ooo:{out_of_order},"
                    "missing:{missing_required}]"
                ).format(**self._trigger_stats)
            self.status_pub.publish(
                String(data="ready={} {}{}".format(ready, " ".join(ages), reliability))
            )
            controller = self.modality_reliability_controller
            modalities = {}
            for name in ("camera", "lidar", "radar"):
                quality = float(self._last_modality_qualities.get(name, 0.0))
                modalities[name] = {
                    "quality": quality,
                    "weight": float(self._last_modality_weights.get(name, 0.0)),
                    "active": bool(
                        controller.active.get(name, quality > 0.0)
                        if self.use_temporal_modality_reliability
                        else quality > 0.0
                    ),
                }
            self.modality_health_pub.publish(
                String(
                    data=json.dumps(
                        {
                            "format_version": 1,
                            "timestamp": now,
                            "state": self._last_reliability_state,
                            "modalities": modalities,
                            "vins": {
                                "quality": float(self._last_vins_quality),
                                "used": bool(self._last_vins_used),
                            },
                        },
                        sort_keys=True,
                    )
                )
            )
            if ready and not self.process_on_lidar_callback:
                self._process_bev(event.current_real)

    def _process_bev(self, stamp, synchronized_snapshot=None):
        previous_reference = self._active_sync_reference
        previous_snapshot = self._active_sync_snapshot
        if synchronized_snapshot is not None:
            self._active_sync_reference = stamp.to_sec()
            self._active_sync_snapshot = synchronized_snapshot
        try:
            return self._process_bev_from_snapshot(stamp)
        finally:
            self._active_sync_reference = previous_reference
            self._active_sync_snapshot = previous_snapshot

    def _process_bev_from_snapshot(self, stamp):
        self._last_pipeline_timings_ms = {}
        self._last_adaptive_detail_timings_ms = {}
        self._last_external_candidate_prefilter_stats = {
            "input_count": 0,
            "kept_count": 0,
            "dropped_count": 0,
            "independent_input_count": 0,
            "independent_kept_count": 0,
            "independent_dropped_count": 0,
        }
        self._last_radar_surface_candidate_count = 0
        self._last_camera_semantic_candidate_count = 0
        self._last_radar_candidate_gate_active = False
        self._last_camera_candidate_gate_active = False
        stage_start = time.perf_counter()
        point_messages = []
        stamps = []
        for name in self.point_cloud_modalities:
            message = self._message_for_stamp(name, stamp)
            if message is None:
                continue
            age = self._message_age_for_stamp(name, stamp)
            if age is None or age > self.cache.max_age_sec:
                continue
            point_messages.append((name, message))
            stamps.append(self._message_stamp_for_stamp(name, stamp))

        if not point_messages:
            return

        stamp_key = tuple(stamps)
        if stamp_key == self._last_processed_stamps:
            return
        self._last_processed_stamps = stamp_key

        point_arrays = []
        modality_arrays = {}
        radar_intensity_lookup = {}
        for modality, message in point_messages:
            intensities = None
            try:
                if modality == "radar" and self.use_radar_intensity_filter:
                    try:
                        points, intensities = pointcloud2_to_xyz_intensity_arrays(
                            message,
                            intensity_field=self.radar_intensity_field,
                        )
                        intensity_mask = radar_intensity_inlier_mask(
                            intensities,
                            min_intensity=self.radar_min_intensity,
                        )
                        self._last_radar_intensity_inlier_ratio = (
                            float(np.mean(intensity_mask)) if intensity_mask.size else 0.0
                        )
                        points = points[intensity_mask]
                        intensities = intensities[intensity_mask]
                    except ValueError as exc:
                        rospy.logwarn_throttle(
                            5.0,
                            "Radar intensity filter unavailable, using XYZ only: {}".format(exc),
                        )
                        points = pointcloud2_to_xyz_array(message)
                        self._last_radar_intensity_inlier_ratio = 1.0
                else:
                    points = pointcloud2_to_xyz_array(message)
                    if modality == "radar":
                        self._last_radar_intensity_inlier_ratio = 1.0
            except Exception as exc:  # ROS adapters can fail on malformed PointCloud2 fields.
                rospy.logwarn_throttle(2.0, "PointCloud2 conversion failed: {}".format(exc))
                continue
            try:
                points = transform_point_cloud_xyz(points, self._point_cloud_transform_for(modality))
            except Exception as exc:
                rospy.logwarn_throttle(
                    2.0,
                    "{} PointCloud2 transform failed: {}".format(modality, exc),
                )
                continue
            if modality == "radar" and intensities is not None:
                radar_intensity_lookup.update(
                    {
                        tuple(point): float(intensity)
                        for point, intensity in zip(points, intensities)
                    }
                )
            points = self._filter_points_by_range(points)
            points = self._filter_points_by_height(points, modality=modality)
            points = self._downsample_points(points)
            if points.size:
                point_arrays.append(points)
                modality_arrays.setdefault(modality, []).append(points)

        if not point_arrays and not self.use_reliability_gated_camera_candidates:
            return
        self._last_pipeline_timings_ms["input_preparation_ms"] = (
            time.perf_counter() - stage_start
        ) * 1000.0

        stage_start = time.perf_counter()
        points_xyz = (
            np.vstack(point_arrays)
            if point_arrays
            else np.empty((0, 3), dtype=float)
        )
        lidar_like_arrays = [
            array for modality, arrays in modality_arrays.items() if modality != "radar" for array in arrays
        ]
        radar_arrays = modality_arrays.get("radar", [])
        lidar_like_points = (
            np.vstack(lidar_like_arrays)
            if lidar_like_arrays
            else np.empty((0, 3), dtype=float)
        )
        radar_points = np.vstack(radar_arrays) if radar_arrays else np.empty((0, 3), dtype=float)
        base_points_xyz = points_xyz
        if self.use_adaptive_multimodal_fusion and not self.radar_in_base_occupancy:
            base_points_xyz = lidar_like_points
        bev_points, point_weights = self._vins_aligned_points(stamp, base_points_xyz)
        current_features = point_cloud_to_bev_feature_maps(points_xyz, self.grid_spec)
        features = point_cloud_to_bev_feature_maps(
            bev_points,
            self.grid_spec,
            weights=point_weights,
            min_occupancy_weight=self.min_occupancy_weight,
        )
        self._last_pipeline_timings_ms["bev_feature_ms"] = (
            time.perf_counter() - stage_start
        ) * 1000.0

        stage_start = time.perf_counter()
        detection_occupancy = features["occupancy"]
        detection_score_map = features["density"]
        if self.use_adaptive_multimodal_fusion:
            detection_occupancy, detection_score_map = self._adaptive_multimodal_detection_maps(
                stamp,
                detection_occupancy,
                detection_score_map,
                lidar_like_points,
                radar_points,
                radar_intensity_lookup,
            )
        self._last_pipeline_timings_ms["adaptive_fusion_ms"] = (
            time.perf_counter() - stage_start
        ) * 1000.0
        if self.use_current_support_gate:
            detection_occupancy = gate_occupancy_by_support(
                detection_occupancy,
                current_features["occupancy"],
                dilation_cells=self.current_support_dilation_cells,
            )

        stage_start = time.perf_counter()
        if self.candidate_source == "external_detector":
            external_message = self._message_for_stamp(
                "external_detector",
                stamp,
            )
            if external_message is None:
                return
            obstacles = detection_message_to_obstacles(
                external_message,
                self.grid_spec,
                minimum_score=self.external_detector_min_score,
            )
            (
                obstacles,
                self._last_external_candidate_prefilter_stats,
            ) = prefilter_independent_candidates(
                obstacles,
                topk_per_modality=(
                    self.external_independent_candidate_topk
                ),
                nms_iou=self.external_independent_candidate_nms_iou,
            )
            detection_occupancy = candidate_occupancy_from_obstacles(
                obstacles,
                self.grid_spec,
            )
            (
                exact_support_ratios,
                exact_candidate_diagnostics,
            ) = exact_candidate_evidence(
                obstacles,
                lidar_like_points,
                self._last_candidate_radar_support_points,
                self._last_candidate_support_maps.get("camera"),
                self.grid_spec,
                radar_intensities=(
                    self._last_candidate_radar_support_intensities
                ),
                lidar_point_saturation=(
                    self.external_lidar_point_saturation
                ),
                radar_point_saturation=(
                    self.external_radar_point_saturation
                ),
                camera_coverage_reference=(
                    self.external_camera_coverage_reference
                ),
                lidar_vertical_margin=(
                    self.external_lidar_vertical_margin
                ),
            )
        elif self.use_dual_obstacle_fusion:
            current_obstacles = detect_occupied_obstacles(
                current_features["occupancy"],
                self.grid_spec,
                min_cells=self.min_obstacle_cells,
                connectivity=self.connectivity,
                score_map=detection_score_map,
                max_size_x=self.max_obstacle_size_x,
                max_size_y=self.max_obstacle_size_y,
                min_cells_by_distance=self.min_obstacle_cells_by_distance,
                max_size_x_by_distance=self.max_obstacle_size_x_by_distance,
                max_size_y_by_distance=self.max_obstacle_size_y_by_distance,
                max_aspect_ratio=self.max_obstacle_aspect_ratio,
                max_aspect_ratio_by_distance=self.max_obstacle_aspect_ratio_by_distance,
            )
            temporal_obstacles = detect_occupied_obstacles(
                detection_occupancy,
                self.grid_spec,
                min_cells=self.min_obstacle_cells,
                connectivity=self.connectivity,
                score_map=detection_score_map,
                support_map=current_features["occupancy"],
                min_support_ratio=self.min_current_support_ratio,
                support_dilation_cells=self.current_support_dilation_cells,
                split_components_by_support=self.use_support_component_split,
                max_size_x=self.max_obstacle_size_x,
                max_size_y=self.max_obstacle_size_y,
                min_cells_by_distance=self.min_obstacle_cells_by_distance,
                max_size_x_by_distance=self.max_obstacle_size_x_by_distance,
                max_size_y_by_distance=self.max_obstacle_size_y_by_distance,
                max_aspect_ratio=self.max_obstacle_aspect_ratio,
                max_aspect_ratio_by_distance=self.max_obstacle_aspect_ratio_by_distance,
            )
            obstacles = merge_obstacle_candidates(
                current_obstacles,
                temporal_obstacles,
                iou_threshold=self.dual_fusion_iou_threshold,
            )
            detection_occupancy = np.maximum(detection_occupancy, current_features["occupancy"])
        else:
            obstacles = detect_occupied_obstacles(
                detection_occupancy,
                self.grid_spec,
                min_cells=self.min_obstacle_cells,
                connectivity=self.connectivity,
                score_map=detection_score_map,
                support_map=current_features["occupancy"],
                min_support_ratio=self.min_current_support_ratio,
                support_dilation_cells=self.current_support_dilation_cells,
                split_components_by_support=self.use_support_component_split,
                max_size_x=self.max_obstacle_size_x,
                max_size_y=self.max_obstacle_size_y,
                min_cells_by_distance=self.min_obstacle_cells_by_distance,
                max_size_x_by_distance=self.max_obstacle_size_x_by_distance,
                max_size_y_by_distance=self.max_obstacle_size_y_by_distance,
                max_aspect_ratio=self.max_obstacle_aspect_ratio,
                max_aspect_ratio_by_distance=self.max_obstacle_aspect_ratio_by_distance,
            )

        radar_candidate_gate_active = (
            self.force_radar_candidate_generation
            or float(self._last_frame_modality_qualities.get("lidar", 1.0))
            <= self.radar_candidate_max_lidar_quality
        )
        self._last_radar_candidate_gate_active = bool(
            self.use_reliability_gated_radar_candidates
            and radar_candidate_gate_active
            and self._last_candidate_radar_support_points.size
        )
        if self._last_radar_candidate_gate_active:
            radar_candidates = radar_surface_candidates(
                self._last_candidate_radar_support_points,
                self.grid_spec,
                cluster_radius=self.radar_candidate_cluster_radius,
                min_points=self.radar_candidate_min_points,
                radial_extent=self.radar_candidate_radial_extent,
                padding=self.radar_candidate_padding,
                max_surface_span=self.radar_candidate_max_surface_span,
                max_range=self.radar_candidate_max_range,
            )
            self._last_radar_surface_candidate_count = len(radar_candidates)
            obstacles = merge_obstacle_candidates(
                obstacles,
                radar_candidates,
                iou_threshold=self.radar_candidate_merge_iou,
            )

        camera_quality_gate_active = (
            self._last_camera_raw_image_quality
            >= self.camera_candidate_min_image_quality
            and self._last_camera_semantic_quality
            >= self.camera_candidate_min_semantic_quality
            and self._last_camera_semantic_mask is not None
        )
        camera_state_gate_active = (
            self.force_camera_candidate_generation
            or (
                float(self._last_frame_modality_qualities.get("lidar", 1.0))
                <= self.camera_candidate_max_lidar_quality
                and float(self._last_frame_modality_qualities.get("radar", 0.0))
                <= self.camera_candidate_max_radar_quality
            )
        )
        self._last_camera_candidate_gate_active = bool(
            self.use_reliability_gated_camera_candidates
            and camera_state_gate_active
            and camera_quality_gate_active
        )
        if self._last_camera_candidate_gate_active:
            camera_candidates = camera_semantic_candidates(
                self._last_camera_semantic_mask,
                self.grid_spec,
                horizontal_fov_deg=self.camera_candidate_horizontal_fov_deg,
                vertical_fov_deg=self.camera_candidate_vertical_fov_deg,
                assumed_height_m=self.camera_candidate_assumed_height,
                min_component_area_px=(
                    self.camera_candidate_min_component_area_px
                ),
                max_component_fraction=(
                    self.camera_candidate_max_component_fraction
                ),
                min_pixel_height=self.camera_candidate_min_pixel_height,
                min_range=self.camera_candidate_min_range,
                max_range=self.camera_candidate_max_range,
                radial_size=self.camera_candidate_radial_size,
                min_lateral_size=self.camera_candidate_min_lateral_size,
                max_lateral_size=self.camera_candidate_max_lateral_size,
                lateral_padding=self.camera_candidate_lateral_padding,
            )
            self._last_camera_semantic_candidate_count = len(camera_candidates)
            obstacles = merge_obstacle_candidates(
                obstacles,
                camera_candidates,
                iou_threshold=self.camera_candidate_merge_iou,
            )

        self._last_pipeline_timings_ms["obstacle_extraction_ms"] = (
            time.perf_counter() - stage_start
        ) * 1000.0
        stage_start = time.perf_counter()
        if self.use_candidate_adaptive_fusion:
            obstacles, candidate_diagnostics = adapt_obstacle_candidates(
                obstacles,
                detection_occupancy,
                self._last_candidate_support_maps,
                self._last_modality_qualities,
                self._last_modality_weights,
                threshold=self.candidate_reliability_threshold,
                evidence_mix=self.candidate_evidence_mix,
                quality_mix=self.candidate_quality_mix,
                support_floor=self.candidate_local_support_floor,
                support_exponent=self.candidate_local_support_exponent,
                participation_min_quality=self.candidate_participation_min_quality,
                participation_min_support=self.candidate_participation_min_support,
                support_ratios_by_obstacle=exact_support_ratios,
            )
            if exact_candidate_diagnostics:
                candidate_diagnostics = tuple(
                    {
                        **row,
                        **exact_candidate_diagnostics.get(
                            int(row["obstacle_id"]),
                            {},
                        ),
                    }
                    for row in candidate_diagnostics
                )
            if self.use_candidate_temporal_confirmation:
                tracking_transform = (
                    self._cached_vins_transform
                    if self._last_vins_used
                    and self._last_vins_quality
                    >= self.vins_min_alignment_quality
                    else None
                )
                obstacles, candidate_diagnostics = (
                    self.candidate_temporal_tracker.update(
                        obstacles,
                        candidate_diagnostics,
                        stamp.to_sec(),
                        world_from_body=tracking_transform,
                        allow_initial_unconfirmed=(
                            max(
                                self._last_modality_qualities.values(),
                                default=0.0,
                            )
                            < self.candidate_temporal_failsafe_quality_threshold
                        ),
                    )
                )
            if (
                self.learned_candidate_gate is not None
                or self.hierarchical_candidate_gate is not None
            ):
                diagnostic_by_id = {
                    int(row["obstacle_id"]): dict(row)
                    for row in candidate_diagnostics
                }
                active_diagnostics = []
                for obstacle in obstacles:
                    diagnostic = diagnostic_by_id[
                        int(obstacle.obstacle_id)
                    ]
                    for modality in ("lidar", "camera", "radar"):
                        diagnostic[
                            "{}_quality".format(modality)
                        ] = self._last_modality_qualities.get(
                            modality,
                            0.0,
                        )
                        diagnostic[
                            "{}_global_weight".format(modality)
                        ] = self._last_modality_weights.get(
                            modality,
                            0.0,
                        )
                        diagnostic[
                            "{}_sync_offset_sec".format(modality)
                        ] = self._last_modality_sync_offsets.get(
                            modality,
                            -1.0,
                        )
                        diagnostic[
                            "{}_sync_quality".format(modality)
                        ] = self._last_modality_sync_qualities.get(
                            modality,
                            0.0,
                        )
                    diagnostic["vins_quality"] = self._last_vins_quality
                    diagnostic["vins_used"] = self._last_vins_used
                    diagnostic["camera_raw_image_quality"] = (
                        self._last_camera_raw_image_quality
                    )
                    active_diagnostics.append(diagnostic)
                candidate_gate = (
                    self.hierarchical_candidate_gate
                    if self.hierarchical_candidate_gate is not None
                    else self.learned_candidate_gate
                )
                obstacles, candidate_diagnostics = (
                    candidate_gate.apply(
                        obstacles,
                        active_diagnostics,
                    )
                )
            detection_occupancy = occupancy_for_candidates(
                detection_occupancy,
                obstacles,
            )
            self._write_candidate_reliability(
                stamp.to_sec(),
                candidate_diagnostics,
            )

        self._last_pipeline_timings_ms["candidate_fusion_ms"] = (
            time.perf_counter() - stage_start
        ) * 1000.0
        stage_start = time.perf_counter()
        if self.publish_grid:
            self.grid_pub.publish(self._occupancy_grid_message(detection_occupancy, stamp))
        if self.publish_markers:
            self.marker_pub.publish(self._marker_array_message(obstacles, stamp))
        self._write_obstacles(stamp.to_sec(), obstacles)
        self._last_pipeline_timings_ms["output_ms"] = (
            time.perf_counter() - stage_start
        ) * 1000.0
        return len(obstacles)

    def _downsample_points(self, points_xyz):
        if self.max_points_per_frame <= 0 or points_xyz.shape[0] <= self.max_points_per_frame:
            return points_xyz
        step = int(np.ceil(float(points_xyz.shape[0]) / float(self.max_points_per_frame)))
        return points_xyz[::step][: self.max_points_per_frame]

    def _point_cloud_transform_for(self, modality):
        return self.point_cloud_transform_by_modality.get(modality, self.point_cloud_transform)

    def _filter_points_by_height(self, points_xyz, modality=None):
        return filter_points_by_height(
            points_xyz,
            min_point_z=self.min_point_z,
            max_point_z=self.max_point_z,
            use_adaptive_ground_filter=(
                self.use_adaptive_ground_filter and modality != "radar"
            ),
            adaptive_ground_quantile=self.adaptive_ground_quantile,
            adaptive_ground_clearance=self.adaptive_ground_clearance,
            adaptive_ground_min_points=self.adaptive_ground_min_points,
        )

    def _filter_points_by_range(self, points_xyz):
        should_filter = self.use_range_median_filter
        if not should_filter and self.use_auto_range_median_filter:
            score = lateral_range_deviation_score(
                points_xyz,
                lateral_resolution=self.range_filter_lateral_resolution,
                min_points_per_bin=self.auto_range_filter_min_points,
                score_quantile=self.auto_range_filter_score_quantile,
            )
            noisy_frame = score is not None and score >= self.auto_range_filter_threshold
            self._auto_range_filter_history.append(noisy_frame)
            noisy_frames = sum(1 for item in self._auto_range_filter_history if item)
            should_filter = noisy_frames >= max(1, self.auto_range_filter_min_noisy_frames)
            if should_filter:
                rospy.loginfo_throttle(
                    5.0,
                    "Auto range median filter enabled for noisy point cloud: score={:.4f} threshold={:.4f} noisy_frames={}/{}".format(
                        score if score is not None else float("nan"),
                        self.auto_range_filter_threshold,
                        noisy_frames,
                        self._auto_range_filter_history.maxlen,
                    ),
                )
        if not should_filter:
            return points_xyz
        return filter_points_by_lateral_range_median(
            points_xyz,
            lateral_resolution=self.range_filter_lateral_resolution,
            max_range_deviation=self.range_filter_max_deviation,
            min_points_per_bin=self.range_filter_min_points,
        )

    def _adaptive_multimodal_detection_maps(
        self,
        stamp,
        base_occupancy,
        base_score_map,
        lidar_like_points,
        radar_points,
        radar_intensity_lookup=None,
    ):
        detail_timings = {
            "adaptive_lidar_quality_ms": 0.0,
            "adaptive_camera_total_ms": 0.0,
            "adaptive_camera_quality_ms": 0.0,
            "adaptive_camera_projection_ms": 0.0,
            "adaptive_camera_ground_ipm_ms": 0.0,
            "adaptive_radar_ms": 0.0,
            "adaptive_weighting_fusion_ms": 0.0,
        }
        lidar_stage_start = time.perf_counter()
        support_maps = {}
        self._last_candidate_radar_support_points = np.empty(
            (0, 3),
            dtype=float,
        )
        self._last_candidate_radar_support_intensities = np.empty(
            0,
            dtype=float,
        )
        timing_qualities = {
            modality: self._sensor_timing_quality(modality, stamp)
            for modality in ("lidar", "camera", "radar")
        }
        if (
            self.use_temporal_modality_reliability
            and self.lidar_quality_min_noise_quality < 1.0 - 1.0e-9
        ):
            lidar_range_deviation = lateral_range_deviation_score(
                lidar_like_points,
                lateral_resolution=self.range_filter_lateral_resolution,
                min_points_per_bin=self.auto_range_filter_min_points,
                score_quantile=self.lidar_quality_range_deviation_quantile,
            )
            self._last_lidar_range_deviation = (
                float(lidar_range_deviation)
                if lidar_range_deviation is not None
                else 0.0
            )
            lidar_quality = lidar_quality_score(
                lidar_like_points,
                saturation_points=self.lidar_quality_saturation_points,
                range_deviation=lidar_range_deviation,
                clean_range_deviation=self.lidar_quality_clean_range_deviation,
                degraded_range_deviation=self.lidar_quality_degraded_range_deviation,
                min_noise_quality=self.lidar_quality_min_noise_quality,
            )
        else:
            self._last_lidar_range_deviation = 0.0
            lidar_quality = min(
                1.0,
                float(lidar_like_points.shape[0])
                / float(max(1, self.lidar_quality_saturation_points)),
            )
        lidar_quality *= timing_qualities["lidar"]
        if self.use_relative_lidar_density_quality:
            lidar_quality = min(
                lidar_quality,
                self.relative_lidar_density_tracker.update(
                    lidar_like_points.shape[0]
                ),
            )
        qualities = {"lidar": lidar_quality}
        detail_timings["adaptive_lidar_quality_ms"] = (
            time.perf_counter() - lidar_stage_start
        ) * 1000.0

        camera_stage_start = time.perf_counter()
        self._last_camera_raw_image_quality = 0.0
        camera_message = self._message_for_stamp("camera", stamp)
        if camera_message is not None:
            try:
                camera_quality_start = time.perf_counter()
                gray = image_to_gray_array(
                    camera_message,
                    prefer_uint8=True,
                )
                if self.use_temporal_modality_reliability:
                    raw_image_quality = robust_image_quality_score(
                        gray,
                        sharpness_degraded_laplacian=self.camera_sharpness_degraded_laplacian,
                        sharpness_clean_laplacian=self.camera_sharpness_clean_laplacian,
                        noise_clean_laplacian=self.camera_noise_clean_laplacian,
                        noise_degraded_laplacian=self.camera_noise_degraded_laplacian,
                        min_degradation_quality=self.camera_image_min_degradation_quality,
                    )
                else:
                    raw_image_quality = image_quality_score(gray)
                camera_age_quality = timing_qualities["camera"]
                self._last_camera_raw_image_quality = float(
                    raw_image_quality * camera_age_quality
                )
                camera_quality = raw_image_quality * camera_age_quality
                camera_score_image, semantic_quality = self._camera_score_image(gray, stamp)
                detail_timings["adaptive_camera_quality_ms"] = (
                    time.perf_counter() - camera_quality_start
                ) * 1000.0
                if semantic_quality is not None:
                    self._last_modality_sync_offsets["camera"] = max(
                        self._last_modality_sync_offsets.get("camera", -1.0),
                        self._last_modality_sync_offsets.get(
                            "semantic_mask",
                            -1.0,
                        ),
                    )
                    self._last_modality_sync_qualities["camera"] = min(
                        self._last_modality_sync_qualities.get("camera", 0.0),
                        self._last_modality_sync_qualities.get(
                            "semantic_mask",
                            0.0,
                        ),
                    )
                    if self.use_temporal_modality_reliability:
                        camera_quality = camera_semantic_reliability_score(
                            raw_image_quality * camera_age_quality,
                            semantic_quality,
                            image_quality_floor=self.camera_semantic_image_quality_floor,
                        )
                    else:
                        camera_quality = semantic_quality
                camera_projection_start = time.perf_counter()
                if not lidar_like_points.size:
                    camera_support = np.zeros_like(
                        base_occupancy, dtype=np.float32
                    )
                elif self.camera_support_projection_mode == "calibrated":
                    camera_support = calibrated_camera_support_from_lidar_points(
                        lidar_like_points,
                        camera_score_image,
                        self.grid_spec,
                        self.camera_projection_matrix,
                        self.camera_lidar_extrinsic,
                        min_camera_depth=self.camera_support_min_depth_m,
                        dilation_cells=self.camera_support_dilation_cells,
                    )
                else:
                    camera_support = camera_support_from_body_points(
                        lidar_like_points,
                        camera_score_image,
                        self.grid_spec,
                        horizontal_fov_deg=self.camera_horizontal_fov_deg,
                        vertical_fov_deg=self.camera_vertical_fov_deg,
                        min_depth_m=self.camera_support_min_depth_m,
                        dilation_cells=self.camera_support_dilation_cells,
                    )
                detail_timings["adaptive_camera_projection_ms"] = (
                    time.perf_counter() - camera_projection_start
                ) * 1000.0
                self._last_camera_ground_components = 0
                self._last_camera_ground_points = 0
                camera_ground_start = time.perf_counter()
                if (
                    self.use_camera_ground_ipm
                    and self.camera_projection_matrix is not None
                    and self._last_camera_semantic_mask is not None
                    and semantic_quality is not None
                    and semantic_quality
                    >= self.camera_ground_ipm_min_semantic_quality
                ):
                    ground_support, ground_stats = semantic_mask_ground_support(
                        self._last_camera_semantic_mask,
                        self.grid_spec,
                        self.camera_projection_matrix,
                        self.camera_lidar_extrinsic,
                        ground_z=self.camera_ground_z,
                        min_range_m=self.camera_ground_ipm_min_range,
                        max_range_m=self.camera_ground_ipm_max_range,
                        min_component_area_px=(
                            self.camera_ground_ipm_min_component_area_px
                        ),
                        max_component_fraction=(
                            self.camera_ground_ipm_max_component_fraction
                        ),
                        reject_border_components=(
                            self.camera_ground_ipm_reject_border_components
                        ),
                        column_stride_px=(
                            self.camera_ground_ipm_column_stride_px
                        ),
                        bottom_band_px=self.camera_ground_ipm_bottom_band_px,
                        dilation_cells=self.camera_ground_ipm_dilation_cells,
                    )
                    camera_support = np.maximum(camera_support, ground_support)
                    self._last_camera_ground_components = int(
                        ground_stats["accepted_components"]
                    )
                    self._last_camera_ground_points = int(
                        ground_stats["ground_points"]
                    )
                detail_timings["adaptive_camera_ground_ipm_ms"] = (
                    time.perf_counter() - camera_ground_start
                ) * 1000.0
                support_maps["camera"] = camera_support
                qualities["camera"] = camera_quality
            except Exception as exc:
                qualities["camera"] = 0.0
                rospy.logwarn_throttle(2.0, "Camera support BEV generation failed: {}".format(exc))
        detail_timings["adaptive_camera_total_ms"] = (
            time.perf_counter() - camera_stage_start
        ) * 1000.0

        radar_stage_start = time.perf_counter()
        if radar_points.size:
            radar_support_points = radar_points
            radar_spatial_inlier_ratio = 1.0
            self._last_radar_temporal_fallback = False
            if self.use_temporal_modality_reliability and self.use_radar_spatial_consistency:
                radar_inlier_mask = radar_spatial_inlier_mask(
                    radar_points,
                    neighbor_radius=self.radar_spatial_neighbor_radius,
                    min_neighbors=self.radar_spatial_min_neighbors,
                )
                radar_spatial_inlier_ratio = float(np.mean(radar_inlier_mask))
                if self.use_radar_temporal_consistency:
                    temporal_mask = self._radar_temporal_consistency_mask(
                        stamp,
                        radar_points,
                    )
                    self._last_radar_temporal_inlier_ratio = float(np.mean(temporal_mask))
                    if self.use_adaptive_radar_temporal_fallback:
                        radar_inlier_mask, temporal_fallback = (
                            adaptive_radar_consistency_mask(
                                radar_inlier_mask,
                                temporal_mask,
                                min_temporal_inliers=(
                                    self.radar_temporal_fallback_min_inliers
                                ),
                                min_temporal_ratio=(
                                    self.radar_temporal_fallback_min_ratio
                                ),
                                fallback_min_spatial_ratio=(
                                    self.radar_temporal_fallback_min_spatial_ratio
                                ),
                            )
                        )
                        self._last_radar_temporal_fallback = temporal_fallback
                    else:
                        radar_inlier_mask &= temporal_mask
                if self.filter_radar_spatial_outliers:
                    radar_support_points = radar_points[radar_inlier_mask]
            self._last_radar_spatial_inlier_ratio = radar_spatial_inlier_ratio
            self._last_candidate_radar_support_points = radar_support_points
            if radar_intensity_lookup:
                self._last_candidate_radar_support_intensities = np.asarray(
                    [
                        radar_intensity_lookup.get(tuple(point), 0.0)
                        for point in radar_support_points
                    ],
                    dtype=float,
                )
            else:
                self._last_candidate_radar_support_intensities = np.zeros(
                    radar_support_points.shape[0],
                    dtype=float,
                )
            current_radar_support = support_from_points(
                radar_support_points,
                self.grid_spec,
                dilation_cells=self.radar_support_dilation_cells,
            )
            support_maps["radar"] = self._radar_support_with_memory(
                stamp,
                radar_support_points,
                current_radar_support,
                lidar_quality,
            )
            qualities["radar"] = radar_quality_score(
                radar_points,
                min_points=self.radar_quality_min_points,
                saturation_points=self.radar_quality_saturation_points,
                spatial_inlier_ratio=(
                    radar_spatial_inlier_ratio
                    if self.use_temporal_modality_reliability
                    and self.use_radar_spatial_consistency
                    else None
                ),
                min_spatial_quality=self.radar_min_spatial_quality,
            ) * timing_qualities["radar"]
            if self._last_radar_temporal_fallback:
                qualities["radar"] *= float(
                    np.clip(self.radar_temporal_fallback_quality_scale, 0.0, 1.0)
                )
        else:
            self._last_radar_spatial_inlier_ratio = 0.0
            self._last_radar_temporal_inlier_ratio = 0.0
            self._last_radar_temporal_fallback = False
            self._last_radar_memory_active = False
            self._last_radar_memory_points = 0
            qualities["radar"] = 0.0
        detail_timings["adaptive_radar_ms"] = (
            time.perf_counter() - radar_stage_start
        ) * 1000.0
        self._last_frame_modality_qualities = dict(qualities)

        weighting_stage_start = time.perf_counter()
        priors = {"lidar": self.lidar_fusion_prior}
        if self.topic_by_modality.get("camera"):
            priors["camera"] = self.camera_fusion_prior
        if self.topic_by_modality.get("radar"):
            priors["radar"] = self.radar_fusion_prior
        if self.use_temporal_modality_reliability:
            weights = self.modality_reliability_controller.update(
                qualities,
                priors=priors,
                min_lidar_weight=self.min_lidar_fusion_weight,
            )
            self._last_modality_qualities = dict(
                self.modality_reliability_controller.smoothed_qualities
            )
            self._last_reliability_state = self.modality_reliability_controller.state_label()
        else:
            weights = adaptive_support_weights(
                qualities,
                priors=priors,
                min_lidar_weight=self.min_lidar_fusion_weight,
            )
            self._last_modality_qualities = dict(qualities)
            self._last_reliability_state = "frame_adaptive"
        if (
            self.use_vins_motion_consistency
            and self._last_vins_quality < self.vins_min_alignment_quality
        ):
            if self._last_reliability_state == "normal":
                self._last_reliability_state = "degraded:vins"
            elif "vins" not in self._last_reliability_state:
                self._last_reliability_state += ",vins"
        self._last_modality_weights = dict(weights)
        self._last_candidate_support_maps = {
            "lidar": support_from_points(
                lidar_like_points,
                self.grid_spec,
                dilation_cells=0,
            ),
            **support_maps,
        }
        self._write_reliability(stamp.to_sec(), qualities, weights)
        fused_occupancy, fused_score = fuse_bev_support_maps(
            base_occupancy,
            support_maps=support_maps,
            weights=weights,
            threshold=self.multimodal_fusion_threshold,
        )
        if self._uses_semantic_mask() and self.use_camera_semantic_gate and "camera" in support_maps:
            fused_occupancy = gate_occupancy_by_support(
                fused_occupancy,
                support_maps["camera"],
                dilation_cells=self.camera_semantic_gate_dilation_cells,
            )
        self._last_fusion_confidence_scale = max(qualities.values()) if qualities else 0.0
        calibrated_score = calibrated_fusion_score(
            base_score_map,
            fused_score,
            lidar_quality=lidar_quality,
            modality_qualities=qualities,
        )
        detail_timings["adaptive_weighting_fusion_ms"] = (
            time.perf_counter() - weighting_stage_start
        ) * 1000.0
        self._last_adaptive_detail_timings_ms = detail_timings
        return fused_occupancy, calibrated_score

    def _camera_score_image(self, gray, stamp):
        self._last_camera_semantic_mask = None
        self._last_camera_semantic_quality = 0.0
        mode = self._camera_support_mode()
        if (
            self._uses_semantic_mask()
            or self.use_reliability_gated_camera_candidates
        ):
            mask_message = self._message_for_stamp("semantic_mask", stamp)
            if mask_message is not None:
                mask = image_to_gray_array(
                    mask_message,
                    prefer_uint8=True,
                )
                self._last_camera_semantic_mask = mask
                score = camera_support_score_image(
                    mask,
                    mode="semantic_mask",
                    min_score=self.semantic_mask_min_score,
                )
                quality = semantic_mask_quality_score(
                    score,
                    min_coverage=self.semantic_mask_min_coverage,
                    saturation_coverage=self.semantic_mask_saturation_coverage,
                    overcoverage_start=self.semantic_mask_overcoverage_start,
                    max_coverage=self.semantic_mask_max_coverage,
                    overcoverage_min_quality=self.semantic_mask_overcoverage_min_quality,
                ) * self._sensor_timing_quality("semantic_mask", stamp)
                self._last_camera_semantic_quality = float(quality)
                if self._uses_semantic_mask():
                    return score, quality
            elif self._uses_semantic_mask():
                rospy.logwarn_throttle(2.0, "camera_support_mode=semantic_mask but semantic mask message is unavailable")
        return (
            camera_support_score_image(
                gray,
                mode=mode,
                edge_percentile=self.camera_edge_percentile,
                edge_weight=self.camera_support_edge_weight,
                min_score=self.camera_support_min_score,
            ),
            None,
        )

    def _radar_temporal_consistency_mask(self, stamp, radar_points):
        radar_stamp = self._message_stamp_for_stamp("radar", stamp)
        if (
            radar_stamp is not None
            and radar_stamp == self._last_radar_history_stamp
            and self._last_radar_temporal_mask is not None
            and self._last_radar_temporal_mask.shape[0] == radar_points.shape[0]
        ):
            return self._last_radar_temporal_mask.copy()

        world_from_body, vins_quality = self._vins_transform_and_quality(stamp)
        if world_from_body is None or vins_quality < self.vins_min_alignment_quality:
            return np.ones((radar_points.shape[0],), dtype=bool)
        try:
            history = self.radar_temporal_buffer.aligned_points(
                world_from_body,
                stamp.to_sec(),
            )
            if history.size:
                mask = radar_temporal_inlier_mask(
                    radar_points,
                    history,
                    neighbor_radius=self.radar_temporal_neighbor_radius,
                    min_neighbors=self.radar_temporal_min_neighbors,
                )
            else:
                mask = np.ones((radar_points.shape[0],), dtype=bool)
            self.radar_temporal_buffer.add(
                stamp.to_sec(),
                radar_points,
                world_from_body,
            )
        except Exception as exc:
            rospy.logwarn_throttle(
                2.0,
                "Radar temporal consistency failed: {}".format(exc),
            )
            return np.ones((radar_points.shape[0],), dtype=bool)

        self._last_radar_history_stamp = radar_stamp
        self._last_radar_temporal_mask = mask.copy()
        return mask

    def _radar_support_with_memory(
        self,
        stamp,
        radar_support_points,
        current_support,
        lidar_quality,
    ):
        self._last_radar_memory_active = False
        self._last_radar_memory_points = 0
        if not self.use_reliability_gated_radar_memory:
            return current_support

        world_from_body, vins_quality = self._vins_transform_and_quality(stamp)
        if world_from_body is None or vins_quality < self.vins_min_alignment_quality:
            self.radar_memory_buffer.clear()
            self._last_radar_memory_stamp = None
            return current_support
        try:
            history_points, history_weights = (
                self.radar_memory_buffer.aligned_points_with_weights(
                    world_from_body,
                    stamp.to_sec(),
                )
            )
            radar_stamp = self._message_stamp_for_stamp("radar", stamp)
            if radar_stamp is None:
                radar_stamp = stamp.to_sec()
            if radar_stamp != self._last_radar_memory_stamp:
                self.radar_memory_buffer.add(
                    stamp.to_sec(),
                    radar_support_points,
                    world_from_body,
                )
                self._last_radar_memory_stamp = radar_stamp
            if history_points.size:
                temporal_support = support_from_points(
                    history_points,
                    self.grid_spec,
                    weights=history_weights,
                    dilation_cells=self.radar_support_dilation_cells,
                )
            else:
                temporal_support = np.zeros_like(current_support, dtype=np.float32)
            fused, active = reliability_gated_temporal_support(
                current_support,
                temporal_support,
                lidar_quality=lidar_quality,
                max_lidar_quality=self.radar_memory_max_lidar_quality,
                temporal_scale=self.radar_memory_support_scale,
            )
            self._last_radar_memory_active = active
            self._last_radar_memory_points = int(history_points.shape[0])
            return fused
        except Exception as exc:
            rospy.logwarn_throttle(
                2.0,
                "Radar temporal memory failed: {}".format(exc),
            )
            self.radar_memory_buffer.clear()
            self._last_radar_memory_stamp = None
            return current_support

    def _camera_support_mode(self):
        return str(self.camera_support_mode or "intensity").strip().lower()

    def _uses_semantic_mask(self):
        return self._camera_support_mode() in ("semantic_mask", "mask")

    def _age_quality(self, modality, stamp):
        return health_score(
            self._message_age_for_stamp(modality, stamp),
            config=self._fusion_health_config(),
        )

    def _sensor_timing_quality(self, modality, stamp):
        offset = self._message_age_for_stamp(modality, stamp)
        age_quality = health_score(
            offset,
            config=self._fusion_health_config(),
        )
        sync_quality = timestamp_alignment_quality(
            offset,
            self.multimodal_sync_config,
        )
        quality = (
            min(age_quality, sync_quality)
            if self.use_multimodal_sync_quality
            else age_quality
        )
        self._last_modality_sync_offsets[modality] = (
            -1.0 if offset is None else float(offset)
        )
        self._last_modality_sync_qualities[modality] = float(quality)
        return float(quality)

    def _message_for_stamp(self, modality, stamp):
        if not self.process_on_lidar_callback:
            return self.cache.get(modality)
        reference = stamp.to_sec() if hasattr(stamp, "to_sec") else float(stamp)
        if (
            self._active_sync_snapshot is not None
            and self._active_sync_reference is not None
            and abs(reference - self._active_sync_reference)
            <= self.lidar_trigger_stamp_epsilon_sec
        ):
            item = self._active_sync_snapshot.get(modality)
            return None if item is None else item.message
        return self.cache.nearest(modality, reference)

    def _message_stamp_for_stamp(self, modality, stamp):
        if not self.process_on_lidar_callback:
            return self.cache.stamp(modality)
        reference = stamp.to_sec() if hasattr(stamp, "to_sec") else float(stamp)
        if (
            self._active_sync_snapshot is not None
            and self._active_sync_reference is not None
            and abs(reference - self._active_sync_reference)
            <= self.lidar_trigger_stamp_epsilon_sec
        ):
            item = self._active_sync_snapshot.get(modality)
            return None if item is None else item.stamp
        return self.cache.nearest_stamp(modality, reference)

    def _message_age_for_stamp(self, modality, stamp):
        reference = stamp.to_sec() if hasattr(stamp, "to_sec") else float(stamp)
        if not self.process_on_lidar_callback:
            return self.cache.age(modality, reference)
        if (
            self._active_sync_snapshot is not None
            and self._active_sync_reference is not None
            and abs(reference - self._active_sync_reference)
            <= self.lidar_trigger_stamp_epsilon_sec
        ):
            item = self._active_sync_snapshot.get(modality)
            return None if item is None else item.offset_sec
        return self.cache.nearest_age(modality, reference)

    def _fusion_health_config(self):
        from bev_obstacle_detection.fusion_core import FusionHealthConfig

        return FusionHealthConfig(max_age_sec=self.cache.max_age_sec, min_score=0.0)

    def _vins_aligned_points(self, stamp, points_xyz):
        if not self.use_vins_temporal_alignment:
            self._last_vins_quality = 0.0
            self._last_vins_used = False
            return points_xyz, None
        world_from_body, quality = self._vins_transform_and_quality(stamp)
        if world_from_body is None or quality < self.vins_min_alignment_quality:
            self._last_vins_used = False
            self.temporal_buffer.clear()
            self.radar_temporal_buffer.clear()
            return points_xyz, None
        self._last_vins_used = True
        self.temporal_buffer.add(stamp.to_sec(), points_xyz, world_from_body)
        aligned, weights = self.temporal_buffer.aligned_points_with_weights(world_from_body, stamp.to_sec())
        return (aligned, weights) if aligned.size else (points_xyz, None)

    def _vins_transform_and_quality(self, stamp):
        reference = stamp.to_sec() if hasattr(stamp, "to_sec") else float(stamp)
        if self._cached_vins_stamp == reference:
            return self._cached_vins_transform, self._last_vins_quality

        self._cached_vins_stamp = reference
        self._cached_vins_transform = None
        self._last_vins_quality = 0.0
        odom = self._message_for_stamp("odom", stamp)
        age_quality = self._age_quality("odom", stamp)
        if odom is None or age_quality <= 0.0:
            return None, 0.0
        try:
            transform = odometry_to_transform(odom)
        except Exception as exc:
            rospy.logwarn_throttle(
                2.0, "VINS odometry conversion failed: {}".format(exc)
            )
            return None, 0.0

        quality = age_quality
        if self.use_vins_motion_consistency and self._previous_vins_observation:
            previous_stamp, previous_transform = self._previous_vins_observation
            quality *= vins_motion_quality(
                previous_transform,
                transform,
                reference - previous_stamp,
                clean_speed_mps=self.vins_clean_speed_mps,
                rejected_speed_mps=self.vins_rejected_speed_mps,
                clean_angular_rate_rps=self.vins_clean_angular_rate_rps,
                rejected_angular_rate_rps=self.vins_rejected_angular_rate_rps,
            )
        self._previous_vins_observation = (reference, transform)
        self._cached_vins_transform = transform
        self._last_vins_quality = float(quality)
        if quality < self.vins_min_alignment_quality:
            rospy.logwarn_throttle(
                2.0,
                "VINS motion rejected: quality={:.3f} threshold={:.3f}".format(
                    quality, self.vins_min_alignment_quality
                ),
            )
        return transform, self._last_vins_quality

    def _occupancy_grid_message(self, occupancy, stamp):
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.info.resolution = self.grid_spec.resolution
        msg.info.width = self.grid_spec.width
        msg.info.height = self.grid_spec.height
        msg.info.origin.position.x = self.grid_spec.x_min
        msg.info.origin.position.y = self.grid_spec.y_min
        msg.info.origin.orientation.w = 1.0
        ros_grid = np.flipud((np.asarray(occupancy) > 0).astype(np.int8) * 100)
        msg.data = ros_grid.reshape(-1).astype(int).tolist()
        return msg

    def _marker_array_message(self, obstacles, stamp):
        marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.header.stamp = stamp
        clear_marker.header.frame_id = self.frame_id
        clear_marker.ns = "bev_obstacles"
        clear_marker.id = 0
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        for obstacle in obstacles:
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = self.frame_id
            marker.ns = "bev_obstacles"
            marker.id = obstacle.obstacle_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = obstacle.center_x
            marker.pose.position.y = obstacle.center_y
            marker.pose.position.z = obstacle.center_z
            marker.pose.orientation.z = np.sin(obstacle.yaw * 0.5)
            marker.pose.orientation.w = np.cos(obstacle.yaw * 0.5)
            marker.scale.x = max(obstacle.size_x, self.grid_spec.resolution)
            marker.scale.y = max(obstacle.size_y, self.grid_spec.resolution)
            marker.scale.z = max(obstacle.size_z, self.grid_spec.resolution)
            marker.color.r = 0.1
            marker.color.g = 0.85
            marker.color.b = 0.35
            marker.color.a = 0.55
            marker.lifetime = rospy.Duration(0.5)
            marker_array.markers.append(marker)
        return marker_array

    def _open_detection_csv(self):
        if not self.detection_csv_path:
            return
        directory = os.path.dirname(self.detection_csv_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        write_header = not os.path.exists(self.detection_csv_path) or os.path.getsize(self.detection_csv_path) == 0
        self._csv_stream = open(self.detection_csv_path, "a", newline="", encoding="utf-8")
        fieldnames = [
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
            "class_id",
            "class_name",
            "center_z",
            "size_z",
            "yaw",
            "source",
            "sample_token",
        ]
        self._csv_writer = csv.DictWriter(self._csv_stream, fieldnames=fieldnames)
        if write_header:
            self._csv_writer.writeheader()

    def _write_obstacles(self, timestamp, obstacles):
        if self._csv_writer is None or self._csv_stream is None or self._csv_stream.closed:
            return
        for obstacle in obstacles:
            self._csv_writer.writerow(
                {
                    "timestamp": "{:.9f}".format(timestamp),
                    "method": self.method,
                    "frame_id": self.frame_id,
                    "obstacle_id": obstacle.obstacle_id,
                    "center_x": "{:.4f}".format(obstacle.center_x),
                    "center_y": "{:.4f}".format(obstacle.center_y),
                    "size_x": "{:.4f}".format(obstacle.size_x),
                    "size_y": "{:.4f}".format(obstacle.size_y),
                    "cell_count": obstacle.cell_count,
                    "score": "{:.4f}".format(obstacle.score),
                    "class_id": obstacle.class_id,
                    "class_name": obstacle.class_name,
                    "center_z": "{:.4f}".format(obstacle.center_z),
                    "size_z": "{:.4f}".format(obstacle.size_z),
                    "yaw": "{:.6f}".format(obstacle.yaw),
                    "source": obstacle.source,
                    "sample_token": obstacle.sample_token,
                }
            )
        self._csv_stream.flush()

    def _close_detection_csv(self):
        if self._csv_stream is not None:
            self._csv_stream.close()
            self._csv_stream = None
        self._csv_writer = None

    def _open_reliability_csv(self):
        if not self.reliability_csv_path:
            return
        directory = os.path.dirname(self.reliability_csv_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        write_header = (
            not os.path.exists(self.reliability_csv_path)
            or os.path.getsize(self.reliability_csv_path) == 0
        )
        self._reliability_csv_stream = open(
            self.reliability_csv_path,
            "a",
            newline="",
            encoding="utf-8",
        )
        fieldnames = ["timestamp", "state"]
        for modality in ("lidar", "camera", "radar"):
            fieldnames.extend(
                [
                    "{}_quality".format(modality),
                    "{}_smoothed_quality".format(modality),
                    "{}_active".format(modality),
                    "{}_weight".format(modality),
                    "{}_sync_offset_sec".format(modality),
                    "{}_sync_quality".format(modality),
                ]
            )
        fieldnames.append("lidar_range_deviation")
        fieldnames.append("radar_spatial_inlier_ratio")
        fieldnames.append("radar_temporal_inlier_ratio")
        fieldnames.append("radar_temporal_fallback")
        fieldnames.append("radar_memory_active")
        fieldnames.append("radar_memory_points")
        fieldnames.append("radar_candidate_gate_active")
        fieldnames.append("radar_surface_candidate_count")
        fieldnames.append("camera_candidate_gate_active")
        fieldnames.append("camera_semantic_candidate_count")
        fieldnames.append("radar_intensity_inlier_ratio")
        fieldnames.append("vins_quality")
        fieldnames.append("vins_used")
        fieldnames.append("fusion_confidence_scale")
        fieldnames.append("camera_raw_image_quality")
        fieldnames.append("camera_semantic_quality")
        fieldnames.append("camera_ground_components")
        fieldnames.append("camera_ground_points")
        self._reliability_csv_writer = csv.DictWriter(
            self._reliability_csv_stream,
            fieldnames=fieldnames,
        )
        if write_header:
            self._reliability_csv_writer.writeheader()

    def _write_reliability(self, timestamp, qualities, weights):
        if (
            self._reliability_csv_writer is None
            or self._reliability_csv_stream is None
            or self._reliability_csv_stream.closed
        ):
            return
        row = {
            "timestamp": "{:.9f}".format(timestamp),
            "state": self._last_reliability_state,
        }
        controller = self.modality_reliability_controller
        for modality in ("lidar", "camera", "radar"):
            quality = float(qualities.get(modality, 0.0))
            smoothed = float(controller.smoothed_qualities.get(modality, quality))
            active = controller.active.get(modality, quality > 0.0)
            row.update(
                {
                    "{}_quality".format(modality): "{:.6f}".format(quality),
                    "{}_smoothed_quality".format(modality): "{:.6f}".format(smoothed),
                    "{}_active".format(modality): str(bool(active)).lower(),
                    "{}_weight".format(modality): "{:.6f}".format(
                        float(weights.get(modality, 0.0))
                    ),
                    "{}_sync_offset_sec".format(modality): "{:.6f}".format(
                        float(
                            self._last_modality_sync_offsets.get(
                                modality,
                                -1.0,
                            )
                        )
                    ),
                    "{}_sync_quality".format(modality): "{:.6f}".format(
                        float(
                            self._last_modality_sync_qualities.get(
                                modality,
                                0.0,
                            )
                        )
                    ),
                }
            )
        row["radar_spatial_inlier_ratio"] = "{:.6f}".format(
            self._last_radar_spatial_inlier_ratio
        )
        row["lidar_range_deviation"] = "{:.6f}".format(
            self._last_lidar_range_deviation
        )
        row["radar_temporal_inlier_ratio"] = "{:.6f}".format(
            self._last_radar_temporal_inlier_ratio
        )
        row["radar_temporal_fallback"] = str(
            bool(self._last_radar_temporal_fallback)
        ).lower()
        row["radar_memory_active"] = str(
            bool(self._last_radar_memory_active)
        ).lower()
        row["radar_memory_points"] = self._last_radar_memory_points
        row["radar_candidate_gate_active"] = str(
            bool(self._last_radar_candidate_gate_active)
        ).lower()
        row["radar_surface_candidate_count"] = (
            self._last_radar_surface_candidate_count
        )
        row["camera_candidate_gate_active"] = str(
            bool(self._last_camera_candidate_gate_active)
        ).lower()
        row["camera_semantic_candidate_count"] = (
            self._last_camera_semantic_candidate_count
        )
        row["radar_intensity_inlier_ratio"] = "{:.6f}".format(
            self._last_radar_intensity_inlier_ratio
        )
        row["vins_quality"] = "{:.6f}".format(self._last_vins_quality)
        row["vins_used"] = str(bool(self._last_vins_used)).lower()
        row["fusion_confidence_scale"] = "{:.6f}".format(
            self._last_fusion_confidence_scale
        )
        row["camera_raw_image_quality"] = "{:.6f}".format(
            self._last_camera_raw_image_quality
        )
        row["camera_semantic_quality"] = "{:.6f}".format(
            self._last_camera_semantic_quality
        )
        row["camera_ground_components"] = self._last_camera_ground_components
        row["camera_ground_points"] = self._last_camera_ground_points
        self._reliability_csv_writer.writerow(row)
        self._reliability_csv_stream.flush()

    def _close_reliability_csv(self):
        if self._reliability_csv_stream is not None:
            self._reliability_csv_stream.close()
            self._reliability_csv_stream = None
        self._reliability_csv_writer = None

    def _open_sync_diagnostics_csv(self):
        if not self.sync_diagnostics_csv_path:
            return
        directory = os.path.dirname(self.sync_diagnostics_csv_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        write_header = (
            not os.path.exists(self.sync_diagnostics_csv_path)
            or os.path.getsize(self.sync_diagnostics_csv_path) == 0
        )
        self._sync_diagnostics_csv_stream = open(
            self.sync_diagnostics_csv_path,
            "a",
            newline="",
            encoding="utf-8",
        )
        fieldnames = ["trigger_timestamp", "decision"]
        for modality in (
            "lidar",
            "camera",
            "semantic_mask",
            "radar",
            "odom",
        ):
            fieldnames.extend(
                [
                    "{}_available".format(modality),
                    "{}_stamp".format(modality),
                    "{}_offset_sec".format(modality),
                ]
            )
        self._sync_diagnostics_csv_writer = csv.DictWriter(
            self._sync_diagnostics_csv_stream,
            fieldnames=fieldnames,
        )
        if write_header:
            self._sync_diagnostics_csv_writer.writeheader()

    def _write_sync_diagnostics(self, reference, decision, snapshot):
        if (
            self._sync_diagnostics_csv_writer is None
            or self._sync_diagnostics_csv_stream is None
            or self._sync_diagnostics_csv_stream.closed
        ):
            return
        row = {
            "trigger_timestamp": "{:.9f}".format(float(reference)),
            "decision": str(decision),
        }
        for modality in (
            "lidar",
            "camera",
            "semantic_mask",
            "radar",
            "odom",
        ):
            item = snapshot.get(modality)
            row["{}_available".format(modality)] = str(item is not None).lower()
            row["{}_stamp".format(modality)] = (
                "" if item is None else "{:.9f}".format(item.stamp)
            )
            row["{}_offset_sec".format(modality)] = (
                "" if item is None else "{:.9f}".format(item.offset_sec)
            )
        self._sync_diagnostics_csv_writer.writerow(row)
        self._sync_diagnostics_csv_stream.flush()

    def _close_sync_diagnostics_csv(self):
        if self._sync_diagnostics_csv_stream is not None:
            self._sync_diagnostics_csv_stream.close()
            self._sync_diagnostics_csv_stream = None
        self._sync_diagnostics_csv_writer = None

    def _open_fusion_latency_csv(self):
        if not self.fusion_latency_csv_path:
            return
        directory = os.path.dirname(self.fusion_latency_csv_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        write_header = (
            not os.path.exists(self.fusion_latency_csv_path)
            or os.path.getsize(self.fusion_latency_csv_path) == 0
        )
        self._fusion_latency_csv_stream = open(
            self.fusion_latency_csv_path,
            "a",
            newline="",
            encoding="utf-8",
        )
        fieldnames = [
            "trigger_timestamp",
            "processed",
            "obstacle_count",
            "configured_sync_wait_ms",
            "sync_inputs_complete",
            "next_sync_wait_ms",
            "queue_wait_ms",
            "contention_wait_ms",
            "processing_ms",
            "input_preparation_ms",
            "bev_feature_ms",
            "adaptive_fusion_ms",
            "obstacle_extraction_ms",
            "candidate_fusion_ms",
            "external_candidate_input_count",
            "external_candidate_kept_count",
            "external_candidate_dropped_count",
            "external_candidate_independent_input_count",
            "external_candidate_independent_kept_count",
            "external_candidate_independent_dropped_count",
            "output_ms",
            "pipeline_unattributed_ms",
            "adaptive_lidar_quality_ms",
            "adaptive_camera_total_ms",
            "adaptive_camera_quality_ms",
            "adaptive_camera_projection_ms",
            "adaptive_camera_ground_ipm_ms",
            "adaptive_radar_ms",
            "adaptive_weighting_fusion_ms",
            "total_compute_ms",
            "sensor_to_output_ms",
            "ros_time_valid",
            "max_selected_sync_offset_ms",
            "available_modalities",
            "deadline_budget_ms",
            "deadline_met",
        ]
        self._fusion_latency_csv_writer = csv.DictWriter(
            self._fusion_latency_csv_stream,
            fieldnames=fieldnames,
        )
        if write_header:
            self._fusion_latency_csv_writer.writeheader()

    def _write_fusion_latency(self, reference, obstacle_count, latency):
        if (
            self._fusion_latency_csv_writer is None
            or self._fusion_latency_csv_stream is None
            or self._fusion_latency_csv_stream.closed
        ):
            return
        processed = obstacle_count is not None
        row = {
            "trigger_timestamp": "{:.9f}".format(float(reference)),
            "processed": str(processed).lower(),
            "obstacle_count": (
                "" if obstacle_count is None else int(obstacle_count)
            ),
        }
        for field in (
            "configured_sync_wait_ms",
            "next_sync_wait_ms",
            "queue_wait_ms",
            "contention_wait_ms",
            "processing_ms",
            "input_preparation_ms",
            "bev_feature_ms",
            "adaptive_fusion_ms",
            "obstacle_extraction_ms",
            "candidate_fusion_ms",
            "output_ms",
            "pipeline_unattributed_ms",
            "adaptive_lidar_quality_ms",
            "adaptive_camera_total_ms",
            "adaptive_camera_quality_ms",
            "adaptive_camera_projection_ms",
            "adaptive_camera_ground_ipm_ms",
            "adaptive_radar_ms",
            "adaptive_weighting_fusion_ms",
            "total_compute_ms",
            "sensor_to_output_ms",
            "max_selected_sync_offset_ms",
            "deadline_budget_ms",
        ):
            row[field] = "{:.6f}".format(float(latency.get(field, 0.0)))
        prefilter = self._last_external_candidate_prefilter_stats
        for field in (
            "input_count",
            "kept_count",
            "dropped_count",
            "independent_input_count",
            "independent_kept_count",
            "independent_dropped_count",
        ):
            row["external_candidate_{}".format(field)] = int(
                prefilter.get(field, 0)
            )
        row["ros_time_valid"] = str(bool(latency["ros_time_valid"])).lower()
        row["sync_inputs_complete"] = str(
            bool(latency.get("sync_inputs_complete", False))
        ).lower()
        row["available_modalities"] = int(latency["available_modalities"])
        row["deadline_met"] = str(bool(latency["deadline_met"])).lower()
        self._fusion_latency_csv_writer.writerow(row)
        self._fusion_latency_csv_stream.flush()

    def _close_fusion_latency_csv(self):
        if self._fusion_latency_csv_stream is not None:
            self._fusion_latency_csv_stream.close()
            self._fusion_latency_csv_stream = None
        self._fusion_latency_csv_writer = None

    def _open_candidate_reliability_csv(self):
        if not self.candidate_reliability_csv_path:
            return
        directory = os.path.dirname(self.candidate_reliability_csv_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        write_header = (
            not os.path.exists(self.candidate_reliability_csv_path)
            or os.path.getsize(self.candidate_reliability_csv_path) == 0
        )
        self._candidate_reliability_csv_stream = open(
            self.candidate_reliability_csv_path,
            "a",
            newline="",
            encoding="utf-8",
        )
        fieldnames = [
            "timestamp",
            "obstacle_id",
            "accepted",
            "base_score",
            "reliability_score",
            "participating_modalities",
            "active_modality_count",
            "temporal_track_id",
            "temporal_confirmed",
            "temporal_ever_confirmed",
            "temporal_unconfirmed_streak",
            "temporal_score_scale",
            "temporal_coordinate_mode",
            "temporal_accepted",
            "temporal_failsafe",
        ]
        for modality in ("lidar", "camera", "radar"):
            fieldnames.extend(
                [
                    "{}_quality".format(modality),
                    "{}_global_weight".format(modality),
                    "{}_sync_offset_sec".format(modality),
                    "{}_sync_quality".format(modality),
                    "{}_support_ratio".format(modality),
                    "{}_local_weight".format(modality),
                ]
            )
        fieldnames.extend(["vins_quality", "vins_used"])
        fieldnames.extend(
            [
                "lidar_point_count",
                "lidar_point_support",
                "radar_point_count",
                "radar_point_support",
                "radar_intensity_mean",
                "radar_intensity_max",
                "radar_intensity_contrast",
                "radar_intensity_support",
                "camera_oriented_coverage",
                "camera_oriented_support",
                "camera_raw_image_quality",
                "hierarchical_source_kind",
                "hierarchical_supplement_reliability",
                "hierarchical_source_health_passed",
                "hierarchical_camera_degraded_mode",
                "hierarchical_lidar_degraded_mode",
                "hierarchical_independent_fallback",
                "hierarchical_independent_reliability_threshold_passed",
                "hierarchical_independent_single_fallback",
                "hierarchical_cross_source_consensus",
                "hierarchical_cross_source_iou",
                "hierarchical_accepted",
            ]
        )
        fieldnames.extend(
            [
                "learned_online_reliability",
                "learned_online_threshold_passed",
                "learned_online_accepted",
            ]
        )
        self._candidate_reliability_csv_writer = csv.DictWriter(
            self._candidate_reliability_csv_stream,
            fieldnames=fieldnames,
        )
        if write_header:
            self._candidate_reliability_csv_writer.writeheader()

    def _write_candidate_reliability(self, timestamp, diagnostics):
        if (
            self._candidate_reliability_csv_writer is None
            or self._candidate_reliability_csv_stream is None
            or self._candidate_reliability_csv_stream.closed
        ):
            return
        for diagnostic in diagnostics:
            row = {
                "timestamp": "{:.9f}".format(timestamp),
                "obstacle_id": diagnostic["obstacle_id"],
                "accepted": str(bool(diagnostic["accepted"])).lower(),
                "base_score": "{:.6f}".format(diagnostic["base_score"]),
                "reliability_score": "{:.6f}".format(
                    diagnostic["reliability_score"]
                ),
                "participating_modalities": diagnostic[
                    "participating_modalities"
                ],
                "active_modality_count": diagnostic[
                    "active_modality_count"
                ],
                "temporal_track_id": diagnostic.get(
                    "temporal_track_id",
                    -1,
                ),
                "temporal_confirmed": str(
                    bool(diagnostic.get("temporal_confirmed", False))
                ).lower(),
                "temporal_ever_confirmed": str(
                    bool(
                        diagnostic.get(
                            "temporal_ever_confirmed",
                            False,
                        )
                    )
                ).lower(),
                "temporal_unconfirmed_streak": diagnostic.get(
                    "temporal_unconfirmed_streak",
                    0,
                ),
                "temporal_score_scale": "{:.6f}".format(
                    float(diagnostic.get("temporal_score_scale", 1.0))
                ),
                "temporal_coordinate_mode": diagnostic.get(
                    "temporal_coordinate_mode",
                    "disabled",
                ),
                "temporal_accepted": str(
                    bool(
                        diagnostic.get(
                            "temporal_accepted",
                            diagnostic["accepted"],
                        )
                    )
                ).lower(),
                "temporal_failsafe": str(
                    bool(diagnostic.get("temporal_failsafe", False))
                ).lower(),
                "vins_quality": "{:.6f}".format(self._last_vins_quality),
                "vins_used": str(bool(self._last_vins_used)).lower(),
                "lidar_point_count": int(
                    diagnostic.get("lidar_point_count", 0)
                ),
                "lidar_point_support": "{:.6f}".format(
                    float(diagnostic.get("lidar_point_support", 0.0))
                ),
                "radar_point_count": int(
                    diagnostic.get("radar_point_count", 0)
                ),
                "radar_point_support": "{:.6f}".format(
                    float(diagnostic.get("radar_point_support", 0.0))
                ),
                "radar_intensity_mean": "{:.6f}".format(
                    float(diagnostic.get("radar_intensity_mean", 0.0))
                ),
                "radar_intensity_max": "{:.6f}".format(
                    float(diagnostic.get("radar_intensity_max", 0.0))
                ),
                "radar_intensity_contrast": "{:.6f}".format(
                    float(
                        diagnostic.get(
                            "radar_intensity_contrast",
                            0.0,
                        )
                    )
                ),
                "radar_intensity_support": "{:.6f}".format(
                    float(
                        diagnostic.get(
                            "radar_intensity_support",
                            0.0,
                        )
                    )
                ),
                "camera_oriented_coverage": "{:.6f}".format(
                    float(
                        diagnostic.get(
                            "camera_oriented_coverage",
                            0.0,
                        )
                    )
                ),
                "camera_oriented_support": "{:.6f}".format(
                    float(
                        diagnostic.get(
                            "camera_oriented_support",
                            0.0,
                        )
                    )
                ),
                "camera_raw_image_quality": "{:.6f}".format(
                    float(
                        diagnostic.get(
                            "camera_raw_image_quality",
                            self._last_camera_raw_image_quality,
                        )
                    )
                ),
                "hierarchical_source_kind": diagnostic.get(
                    "hierarchical_source_kind",
                    "",
                ),
                "hierarchical_supplement_reliability": "{:.6f}".format(
                    float(
                        diagnostic.get(
                            "hierarchical_supplement_reliability",
                            0.0,
                        )
                    )
                ),
                "hierarchical_source_health_passed": str(
                    bool(
                        diagnostic.get(
                            "hierarchical_source_health_passed",
                            False,
                        )
                    )
                ).lower(),
                "hierarchical_camera_degraded_mode": str(
                    bool(
                        diagnostic.get(
                            "hierarchical_camera_degraded_mode",
                            False,
                        )
                    )
                ).lower(),
                "hierarchical_lidar_degraded_mode": str(
                    bool(
                        diagnostic.get(
                            "hierarchical_lidar_degraded_mode",
                            False,
                        )
                    )
                ).lower(),
                "hierarchical_independent_fallback": str(
                    bool(
                        diagnostic.get(
                            "hierarchical_independent_fallback",
                            False,
                        )
                    )
                ).lower(),
                "hierarchical_independent_reliability_threshold_passed": str(
                    bool(
                        diagnostic.get(
                            "hierarchical_independent_reliability_threshold_passed",
                            False,
                        )
                    )
                ).lower(),
                "hierarchical_independent_single_fallback": str(
                    bool(
                        diagnostic.get(
                            "hierarchical_independent_single_fallback",
                            False,
                        )
                    )
                ).lower(),
                "hierarchical_cross_source_consensus": str(
                    bool(
                        diagnostic.get(
                            "hierarchical_cross_source_consensus",
                            False,
                        )
                    )
                ).lower(),
                "hierarchical_cross_source_iou": "{:.6f}".format(
                    float(
                        diagnostic.get(
                            "hierarchical_cross_source_iou",
                            0.0,
                        )
                    )
                ),
                "hierarchical_accepted": str(
                    bool(
                        diagnostic.get(
                            "hierarchical_accepted",
                            False,
                        )
                    )
                ).lower(),
                "learned_online_reliability": "{:.6f}".format(
                    float(
                        diagnostic.get(
                            "learned_online_reliability",
                            0.0,
                        )
                    )
                ),
                "learned_online_accepted": str(
                    bool(
                        diagnostic.get(
                            "learned_online_accepted",
                            False,
                        )
                    )
                ).lower(),
                "learned_online_threshold_passed": str(
                    bool(
                        diagnostic.get(
                            "learned_online_threshold_passed",
                            False,
                        )
                    )
                ).lower(),
            }
            for modality in ("lidar", "camera", "radar"):
                row.update(
                    {
                        "{}_quality".format(modality): "{:.6f}".format(
                            float(self._last_modality_qualities.get(modality, 0.0))
                        ),
                        "{}_global_weight".format(modality): "{:.6f}".format(
                            float(self._last_modality_weights.get(modality, 0.0))
                        ),
                        "{}_sync_offset_sec".format(modality): "{:.6f}".format(
                            float(
                                self._last_modality_sync_offsets.get(
                                    modality,
                                    -1.0,
                                )
                            )
                        ),
                        "{}_sync_quality".format(modality): "{:.6f}".format(
                            float(
                                self._last_modality_sync_qualities.get(
                                    modality,
                                    0.0,
                                )
                            )
                        ),
                        "{}_support_ratio".format(modality): "{:.6f}".format(
                            diagnostic["{}_support_ratio".format(modality)]
                        ),
                        "{}_local_weight".format(modality): "{:.6f}".format(
                            diagnostic["{}_local_weight".format(modality)]
                        ),
                    }
                )
            self._candidate_reliability_csv_writer.writerow(row)
        self._candidate_reliability_csv_stream.flush()

    def _close_candidate_reliability_csv(self):
        if self._candidate_reliability_csv_stream is not None:
            self._candidate_reliability_csv_stream.close()
            self._candidate_reliability_csv_stream = None
        self._candidate_reliability_csv_writer = None

    def _shutdown(self):
        self._shutting_down = True
        if hasattr(self, "timer"):
            self.timer.shutdown()
        self._close_detection_csv()
        self._close_reliability_csv()
        self._close_candidate_reliability_csv()
        self._close_sync_diagnostics_csv()
        self._close_fusion_latency_csv()


def main():
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: rosrun bev_obstacle_detection bev_obstacle_node.py")
        print("ROS params: ~camera_topic ~lidar_topic ~radar_topic ~odom_topic ~max_sensor_age")
        print(
            "Point cloud params: ~point_cloud_transform ~lidar_point_cloud_transform "
            "~radar_point_cloud_transform ~max_points_per_frame "
            "~min_occupancy_weight ~min_current_support_ratio ~current_support_dilation_cells "
            "~use_current_support_gate ~use_dual_obstacle_fusion ~use_support_component_split "
            "~dual_fusion_iou_threshold "
            "~min_point_z ~max_point_z ~use_range_median_filter ~range_filter_max_deviation "
            "~use_auto_range_median_filter ~auto_range_filter_threshold "
            "~use_adaptive_ground_filter ~adaptive_ground_clearance "
            "~min_obstacle_cells_by_distance ~max_obstacle_size_x ~max_obstacle_size_y "
            "~max_obstacle_size_x_by_distance ~max_obstacle_size_y_by_distance "
            "~max_obstacle_aspect_ratio ~max_obstacle_aspect_ratio_by_distance "
            "~use_adaptive_multimodal_fusion ~multimodal_fusion_threshold "
            "~process_on_lidar_callback ~lidar_processing_delay_sec ~sensor_queue_size "
            "~lidar_fusion_prior ~camera_fusion_prior ~radar_fusion_prior "
            "~camera_support_mode ~camera_support_edge_weight ~camera_support_min_score "
            "~semantic_mask_topic ~semantic_mask_min_score ~use_camera_semantic_gate"
        )
        print("ROS outputs: ~status ~occupancy_grid ~obstacle_markers")
        print("CSV params: ~detection_csv_path ~reliability_csv_path")
        print("Fusion params: ~use_vins_temporal_alignment ~temporal_max_frames ~temporal_max_age ~temporal_decay_tau")
        return

    rospy.init_node("bev_obstacle_detection")
    MultimodalBevNode()
    rospy.spin()


if __name__ == "__main__":
    main()
