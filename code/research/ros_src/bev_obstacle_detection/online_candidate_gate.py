"""Deployable learned reliability gate for online BEV candidates."""

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np


BOOLEAN_FEATURES = {
    "vins_used",
    "temporal_confirmed",
    "temporal_ever_confirmed",
    "temporal_failsafe",
}


def _as_float(value, default=0.0):
    if value in ("", None):
        return float(default)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and value.strip().lower() in (
        "true",
        "false",
    ):
        return 1.0 if value.strip().lower() == "true" else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def candidate_feature_mapping(obstacle, diagnostics):
    size_x = float(obstacle.size_x)
    size_y = float(obstacle.size_y)
    center_x = float(obstacle.center_x)
    center_y = float(obstacle.center_y)
    values = dict(diagnostics)
    source = str(getattr(obstacle, "source", "") or "").strip().lower()
    # Legacy feature names are retained for model compatibility. They now
    # identify the candidate's auxiliary modality, including independent
    # Camera-only and Radar-only experts.
    source_is_camera_lidar = float("camera" in source)
    source_is_lidar_radar = float("radar" in source)
    source_is_lidar = float(
        "lidar" in source
        and not source_is_camera_lidar
        and not source_is_lidar_radar
    )
    values.update(
        {
            "score": float(obstacle.score),
            "log_cell_count": math.log1p(float(obstacle.cell_count)),
            "distance": math.hypot(center_x, center_y),
            "size_x": size_x,
            "size_y": size_y,
            "area": size_x * size_y,
            "aspect_ratio": max(size_x, size_y)
            / max(min(size_x, size_y), 1.0e-6),
            "source_is_lidar": source_is_lidar,
            "source_is_camera_lidar": source_is_camera_lidar,
            "source_is_lidar_radar": source_is_lidar_radar,
            "source_lidar_quality": source_is_lidar
            * _as_float(diagnostics.get("lidar_quality")),
            "source_camera_quality": source_is_camera_lidar
            * _as_float(diagnostics.get("camera_quality")),
            "source_radar_quality": source_is_lidar_radar
            * _as_float(diagnostics.get("radar_quality")),
        }
    )
    for feature in BOOLEAN_FEATURES:
        values[feature] = _as_float(values.get(feature))
    return values


def axis_aligned_iou(first, second):
    first_x0 = first.center_x - first.size_x / 2.0
    first_x1 = first.center_x + first.size_x / 2.0
    first_y0 = first.center_y - first.size_y / 2.0
    first_y1 = first.center_y + first.size_y / 2.0
    second_x0 = second.center_x - second.size_x / 2.0
    second_x1 = second.center_x + second.size_x / 2.0
    second_y0 = second.center_y - second.size_y / 2.0
    second_y1 = second.center_y + second.size_y / 2.0
    width = max(0.0, min(first_x1, second_x1) - max(first_x0, second_x0))
    height = max(0.0, min(first_y1, second_y1) - max(first_y0, second_y0))
    intersection = width * height
    union = (
        first.size_x * first.size_y
        + second.size_x * second.size_y
        - intersection
    )
    return intersection / union if union > 0.0 else 0.0


def _box_corners(obstacle):
    half_x = float(obstacle.size_x) * 0.5
    half_y = float(obstacle.size_y) * 0.5
    yaw = float(getattr(obstacle, "yaw", 0.0))
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return [
        (
            obstacle.center_x + x * cos_yaw - y * sin_yaw,
            obstacle.center_y + x * sin_yaw + y * cos_yaw,
        )
        for x, y in (
            (half_x, half_y),
            (half_x, -half_y),
            (-half_x, -half_y),
            (-half_x, half_y),
        )
    ]


def _polygon_area(points):
    if len(points) < 3:
        return 0.0
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(
                points,
                points[1:] + points[:1],
            )
        )
    ) * 0.5


def _inside(point, edge_start, edge_end):
    x, y = point
    x1, y1 = edge_start
    x2, y2 = edge_end
    return (
        (x2 - x1) * (y - y1)
        - (y2 - y1) * (x - x1)
        <= 1.0e-9
    )


def _line_intersection(first, second, edge_start, edge_end):
    x1, y1 = first
    x2, y2 = second
    x3, y3 = edge_start
    x4, y4 = edge_end
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1.0e-9:
        return second
    determinant_first = x1 * y2 - y1 * x2
    determinant_edge = x3 * y4 - y3 * x4
    return (
        (
            determinant_first * (x3 - x4)
            - (x1 - x2) * determinant_edge
        )
        / denominator,
        (
            determinant_first * (y3 - y4)
            - (y1 - y2) * determinant_edge
        )
        / denominator,
    )


def _polygon_clip(subject, clip):
    output = list(subject)
    for edge_start, edge_end in zip(clip, clip[1:] + clip[:1]):
        input_points = output
        output = []
        if not input_points:
            break
        previous = input_points[-1]
        for current in input_points:
            if _inside(current, edge_start, edge_end):
                if not _inside(previous, edge_start, edge_end):
                    output.append(
                        _line_intersection(
                            previous,
                            current,
                            edge_start,
                            edge_end,
                        )
                    )
                output.append(current)
            elif _inside(previous, edge_start, edge_end):
                output.append(
                    _line_intersection(
                        previous,
                        current,
                        edge_start,
                        edge_end,
                    )
                )
            previous = current
    return output


def oriented_iou(first, second):
    first_corners = _box_corners(first)
    second_corners = _box_corners(second)
    area_first = _polygon_area(first_corners)
    area_second = _polygon_area(second_corners)
    intersection = _polygon_area(
        _polygon_clip(first_corners, second_corners)
    )
    union = area_first + area_second - intersection
    return intersection / union if union > 0.0 else 0.0


class OnlineCandidateReliabilityGate:
    def __init__(
        self,
        features,
        mean,
        std,
        weights,
        threshold,
        nms_iou,
        topk,
    ):
        self.features = tuple(features)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.threshold = float(threshold)
        self.nms_iou = float(nms_iou)
        self.topk = int(topk)
        if self.mean.shape != (len(self.features),):
            raise ValueError("online candidate model mean dimension mismatch")
        if self.std.shape != (len(self.features),):
            raise ValueError("online candidate model std dimension mismatch")
        if self.weights.shape != (len(self.features) + 1,):
            raise ValueError("online candidate model weight dimension mismatch")
        if np.any(self.std <= 0.0):
            raise ValueError("online candidate model std must be positive")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("online candidate threshold must be in [0, 1]")
        if not 0.0 <= self.nms_iou <= 1.0:
            raise ValueError("online candidate NMS IoU must be in [0, 1]")
        if self.topk <= 0:
            raise ValueError("online candidate top-k must be positive")

    @classmethod
    def from_json(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != 1:
            raise ValueError("unsupported online candidate model format")
        if (
            payload.get("model_type")
            != "standardized_logistic_online_candidate_gate"
        ):
            raise ValueError("unsupported online candidate model type")
        return cls(
            features=payload["features"],
            mean=payload["mean"],
            std=payload["std"],
            weights=payload["weights"],
            threshold=payload["candidate_threshold"],
            nms_iou=payload["nms_iou"],
            topk=payload["topk"],
        )

    def score(self, obstacle, diagnostics):
        values = candidate_feature_mapping(obstacle, diagnostics)
        numeric = np.asarray(
            [_as_float(values.get(feature)) for feature in self.features],
            dtype=np.float64,
        )
        normalized = (numeric - self.mean) / self.std
        logit = self.weights[0] + normalized @ self.weights[1:]
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))

    def apply(self, obstacles, diagnostics):
        if len(obstacles) != len(diagnostics):
            raise ValueError("obstacles and diagnostics must be aligned")
        ranked = []
        updated_diagnostics = []
        for obstacle, diagnostic in zip(obstacles, diagnostics):
            score = self.score(obstacle, diagnostic)
            updated = dict(diagnostic)
            updated["learned_online_reliability"] = score
            updated["learned_online_threshold_passed"] = (
                score >= self.threshold
            )
            updated["learned_online_accepted"] = False
            updated_diagnostics.append(updated)
            if score >= self.threshold:
                ranked.append(
                    (
                        score,
                        int(obstacle.obstacle_id),
                        replace(obstacle, score=score),
                        updated,
                    )
                )
        ranked.sort(key=lambda item: (-item[0], item[1]))
        kept = []
        for _, _, obstacle, diagnostic in ranked[: self.topk]:
            if all(
                oriented_iou(obstacle, previous[0]) < self.nms_iou
                for previous in kept
            ):
                diagnostic["learned_online_accepted"] = True
                kept.append((obstacle, diagnostic))
        return (
            tuple(item[0] for item in kept),
            tuple(updated_diagnostics),
        )
