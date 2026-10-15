"""Deployable hierarchical multi-source candidate fusion."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from bev_obstacle_detection.online_candidate_gate import (
    OnlineCandidateReliabilityGate,
    _as_float,
    candidate_feature_mapping,
    oriented_iou,
)


def candidate_source_kind(obstacle):
    source = str(getattr(obstacle, "source", "") or "").strip().lower()
    if "camera" in source:
        return "camera"
    if "radar" in source:
        return "radar"
    return "base"


def candidate_is_independent(obstacle):
    source = str(getattr(obstacle, "source", "") or "").strip().lower()
    return "only" in source and (
        "camera" in source or "radar" in source
    )


class HierarchicalMultisourceCandidateGate:
    def __init__(
        self,
        base_gate,
        features,
        mean,
        std,
        weights,
        camera_threshold,
        radar_threshold,
        camera_score_threshold,
        radar_score_threshold,
        max_base_iou,
        supplement_nms_iou,
        max_camera_per_frame,
        max_radar_per_frame,
        camera_degraded_quality_threshold=0.10,
        degraded_max_camera_per_frame=0,
        degraded_radar_threshold=0.75,
        degraded_max_radar_per_frame=1,
        radar_min_quality=0.05,
        radar_min_local_support=0.01,
        lidar_degraded_quality_threshold=0.0,
        degraded_consensus_iou=0.10,
        degraded_max_consensus_per_frame=0,
        independent_camera_score_threshold=None,
        independent_radar_score_threshold=None,
        independent_camera_reliability_threshold=0.0,
        independent_radar_reliability_threshold=0.0,
        degraded_independent_camera_threshold=1.1,
        degraded_independent_radar_threshold=1.1,
        degraded_max_independent_camera_per_frame=0,
        degraded_max_independent_radar_per_frame=0,
        allow_independent_radar_when_camera_healthy=False,
        lidar_only_independent_radar_threshold=None,
        lidar_only_max_independent_radar_per_frame=None,
        source_models=None,
        independent_source_models=None,
    ):
        self.base_gate = base_gate
        self.features = tuple(features)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.camera_threshold = float(camera_threshold)
        self.radar_threshold = float(radar_threshold)
        self.camera_score_threshold = float(camera_score_threshold)
        self.radar_score_threshold = float(radar_score_threshold)
        self.max_base_iou = float(max_base_iou)
        self.supplement_nms_iou = float(supplement_nms_iou)
        self.max_camera_per_frame = int(max_camera_per_frame)
        self.max_radar_per_frame = int(max_radar_per_frame)
        self.camera_degraded_quality_threshold = float(
            camera_degraded_quality_threshold
        )
        self.degraded_max_camera_per_frame = int(
            degraded_max_camera_per_frame
        )
        self.degraded_radar_threshold = float(degraded_radar_threshold)
        self.degraded_max_radar_per_frame = int(
            degraded_max_radar_per_frame
        )
        self.radar_min_quality = float(radar_min_quality)
        self.radar_min_local_support = float(radar_min_local_support)
        self.lidar_degraded_quality_threshold = float(
            lidar_degraded_quality_threshold
        )
        self.degraded_consensus_iou = float(degraded_consensus_iou)
        self.degraded_max_consensus_per_frame = int(
            degraded_max_consensus_per_frame
        )
        self.independent_camera_score_threshold = float(
            camera_score_threshold
            if independent_camera_score_threshold is None
            else independent_camera_score_threshold
        )
        self.independent_radar_score_threshold = float(
            radar_score_threshold
            if independent_radar_score_threshold is None
            else independent_radar_score_threshold
        )
        self.independent_camera_reliability_threshold = float(
            independent_camera_reliability_threshold
        )
        self.independent_radar_reliability_threshold = float(
            independent_radar_reliability_threshold
        )
        self.degraded_independent_camera_threshold = float(
            degraded_independent_camera_threshold
        )
        self.degraded_independent_radar_threshold = float(
            degraded_independent_radar_threshold
        )
        self.degraded_max_independent_camera_per_frame = int(
            degraded_max_independent_camera_per_frame
        )
        self.degraded_max_independent_radar_per_frame = int(
            degraded_max_independent_radar_per_frame
        )
        self.allow_independent_radar_when_camera_healthy = bool(
            allow_independent_radar_when_camera_healthy
        )
        self.lidar_only_independent_radar_threshold = float(
            degraded_independent_radar_threshold
            if lidar_only_independent_radar_threshold is None
            else lidar_only_independent_radar_threshold
        )
        self.lidar_only_max_independent_radar_per_frame = int(
            degraded_max_independent_radar_per_frame
            if lidar_only_max_independent_radar_per_frame is None
            else lidar_only_max_independent_radar_per_frame
        )
        self.source_models = {}
        for kind, model in (source_models or {}).items():
            if kind not in ("camera", "radar"):
                raise ValueError("unsupported hierarchical source model")
            source_mean = np.asarray(model["mean"], dtype=np.float64)
            source_std = np.asarray(model["std"], dtype=np.float64)
            source_weights = np.asarray(
                model["weights"], dtype=np.float64
            )
            self._validate_model_arrays(
                source_mean,
                source_std,
                source_weights,
                "hierarchical {} gate".format(kind),
            )
            self.source_models[kind] = (
                source_mean,
                source_std,
                source_weights,
            )
        self.independent_source_models = {}
        for kind, model in (independent_source_models or {}).items():
            if kind not in ("camera", "radar"):
                raise ValueError("unsupported independent source model")
            source_mean = np.asarray(model["mean"], dtype=np.float64)
            source_std = np.asarray(model["std"], dtype=np.float64)
            source_weights = np.asarray(
                model["weights"],
                dtype=np.float64,
            )
            self._validate_model_arrays(
                source_mean,
                source_std,
                source_weights,
                "independent {} gate".format(kind),
            )
            self.independent_source_models[kind] = (
                source_mean,
                source_std,
                source_weights,
            )
        self._validate_model_arrays(
            self.mean,
            self.std,
            self.weights,
            "hierarchical gate",
        )

    def _validate_model_arrays(self, mean, std, weights, label):
        if mean.shape != (len(self.features),):
            raise ValueError("{} mean dimension mismatch".format(label))
        if std.shape != (len(self.features),):
            raise ValueError("{} std dimension mismatch".format(label))
        if weights.shape != (len(self.features) + 1,):
            raise ValueError("{} weight dimension mismatch".format(label))
        if np.any(std <= 0.0):
            raise ValueError("{} std must be positive".format(label))

    @classmethod
    def from_json(cls, path):
        model_path = Path(path).expanduser().resolve()
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        if (
            payload.get("format_version") != 1
            or payload.get("model_type")
            != "hierarchical_multisource_online_candidate_gate"
        ):
            raise ValueError("unsupported hierarchical candidate model")
        base_path = Path(payload["base_model_path"]).expanduser()
        if not base_path.is_absolute():
            base_path = model_path.parent / base_path
        return cls(
            base_gate=OnlineCandidateReliabilityGate.from_json(base_path),
            features=payload["features"],
            mean=payload["mean"],
            std=payload["std"],
            weights=payload["weights"],
            camera_threshold=payload.get(
                "camera_supplement_threshold",
                payload.get("supplement_threshold"),
            ),
            radar_threshold=payload.get(
                "radar_supplement_threshold",
                payload.get("supplement_threshold"),
            ),
            camera_score_threshold=payload["camera_score_threshold"],
            radar_score_threshold=payload["radar_score_threshold"],
            max_base_iou=payload["max_base_iou"],
            supplement_nms_iou=payload["supplement_nms_iou"],
            max_camera_per_frame=payload["max_camera_per_frame"],
            max_radar_per_frame=payload["max_radar_per_frame"],
            camera_degraded_quality_threshold=payload.get(
                "camera_degraded_quality_threshold",
                0.10,
            ),
            degraded_max_camera_per_frame=payload.get(
                "degraded_max_camera_per_frame",
                0,
            ),
            degraded_radar_threshold=payload.get(
                "degraded_radar_threshold",
                0.75,
            ),
            degraded_max_radar_per_frame=payload.get(
                "degraded_max_radar_per_frame",
                1,
            ),
            radar_min_quality=payload.get("radar_min_quality", 0.05),
            radar_min_local_support=payload.get(
                "radar_min_local_support",
                0.01,
            ),
            lidar_degraded_quality_threshold=payload.get(
                "lidar_degraded_quality_threshold",
                0.0,
            ),
            degraded_consensus_iou=payload.get(
                "degraded_consensus_iou",
                0.10,
            ),
            degraded_max_consensus_per_frame=payload.get(
                "degraded_max_consensus_per_frame",
                0,
            ),
            independent_camera_score_threshold=payload.get(
                "independent_camera_score_threshold",
            ),
            independent_radar_score_threshold=payload.get(
                "independent_radar_score_threshold",
            ),
            independent_camera_reliability_threshold=payload.get(
                "independent_camera_reliability_threshold",
                0.0,
            ),
            independent_radar_reliability_threshold=payload.get(
                "independent_radar_reliability_threshold",
                0.0,
            ),
            degraded_independent_camera_threshold=payload.get(
                "degraded_independent_camera_threshold",
                1.1,
            ),
            degraded_independent_radar_threshold=payload.get(
                "degraded_independent_radar_threshold",
                1.1,
            ),
            degraded_max_independent_camera_per_frame=payload.get(
                "degraded_max_independent_camera_per_frame",
                0,
            ),
            degraded_max_independent_radar_per_frame=payload.get(
                "degraded_max_independent_radar_per_frame",
                0,
            ),
            allow_independent_radar_when_camera_healthy=payload.get(
                "allow_independent_radar_when_camera_healthy",
                False,
            ),
            lidar_only_independent_radar_threshold=payload.get(
                "lidar_only_independent_radar_threshold",
            ),
            lidar_only_max_independent_radar_per_frame=payload.get(
                "lidar_only_max_independent_radar_per_frame",
            ),
            source_models=payload.get("source_models"),
            independent_source_models=payload.get(
                "independent_source_models"
            ),
        )

    def supplement_score(self, obstacle, diagnostic):
        values = candidate_feature_mapping(obstacle, diagnostic)
        numeric = np.asarray(
            [_as_float(values.get(feature)) for feature in self.features],
            dtype=np.float64,
        )
        kind = candidate_source_kind(obstacle)
        if candidate_is_independent(obstacle):
            mean, std, weights = self.independent_source_models.get(
                kind,
                self.source_models.get(
                    kind,
                    (self.mean, self.std, self.weights),
                ),
            )
        else:
            mean, std, weights = self.source_models.get(
                kind,
                (self.mean, self.std, self.weights),
            )
        normalized = (numeric - mean) / std
        logit = weights[0] + normalized @ weights[1:]
        return float(
            1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0)))
        )

    def apply(self, obstacles, diagnostics):
        if len(obstacles) != len(diagnostics):
            raise ValueError("obstacles and diagnostics must be aligned")
        base_items = []
        supplement_items = []
        updated_by_id = {}
        for obstacle, diagnostic in zip(obstacles, diagnostics):
            updated = dict(diagnostic)
            kind = candidate_source_kind(obstacle)
            updated["hierarchical_source_kind"] = kind
            updated["hierarchical_accepted"] = False
            updated["hierarchical_supplement_reliability"] = 0.0
            updated["hierarchical_source_health_passed"] = kind == "base"
            updated_by_id[int(obstacle.obstacle_id)] = updated
            if kind == "base":
                base_items.append((obstacle, updated))
            else:
                supplement_items.append(
                    (
                        obstacle,
                        updated,
                        kind,
                        candidate_is_independent(obstacle),
                    )
                )

        base_kept, base_diagnostics = self.base_gate.apply(
            tuple(item[0] for item in base_items),
            tuple(item[1] for item in base_items),
        )
        for diagnostic in base_diagnostics:
            obstacle_id = int(diagnostic["obstacle_id"])
            diagnostic["hierarchical_accepted"] = bool(
                diagnostic.get("learned_online_accepted", False)
            )
            updated_by_id[obstacle_id] = diagnostic

        camera_quality = max(
            (
                _as_float(
                    row.get(
                        "camera_raw_image_quality",
                        row.get("camera_quality"),
                    )
                )
                for row in diagnostics
            ),
            default=0.0,
        )
        camera_degraded = (
            camera_quality < self.camera_degraded_quality_threshold
        )
        lidar_quality = max(
            (
                _as_float(row.get("lidar_quality"))
                for row in diagnostics
            ),
            default=0.0,
        )
        lidar_degraded = (
            (
                self.degraded_max_consensus_per_frame > 0
                or self.degraded_max_independent_camera_per_frame > 0
                or self.degraded_max_independent_radar_per_frame > 0
            )
            and lidar_quality < self.lidar_degraded_quality_threshold
        )
        thresholds = {
            "camera": self.camera_threshold,
            "radar": (
                self.degraded_radar_threshold
                if camera_degraded
                else self.radar_threshold
            ),
        }
        limits = {
            "camera": (
                self.degraded_max_camera_per_frame
                if camera_degraded
                else self.max_camera_per_frame
            ),
            "radar": (
                self.degraded_max_radar_per_frame
                if camera_degraded
                else self.max_radar_per_frame
            ),
        }
        source_score_thresholds = {
            "camera": self.camera_score_threshold,
            "radar": self.radar_score_threshold,
        }
        independent_source_score_thresholds = {
            "camera": self.independent_camera_score_threshold,
            "radar": self.independent_radar_score_threshold,
        }
        independent_reliability_thresholds = {
            "camera": self.independent_camera_reliability_threshold,
            "radar": self.independent_radar_reliability_threshold,
        }
        ranked = {"camera": [], "radar": []}
        eligible = {"camera": [], "radar": []}
        for obstacle, diagnostic, kind, independent in supplement_items:
            score = self.supplement_score(obstacle, diagnostic)
            diagnostic["hierarchical_supplement_reliability"] = score
            diagnostic["hierarchical_camera_degraded_mode"] = (
                camera_degraded
            )
            diagnostic["hierarchical_lidar_degraded_mode"] = (
                lidar_degraded
            )
            diagnostic["hierarchical_cross_source_consensus"] = False
            diagnostic["hierarchical_cross_source_iou"] = 0.0
            diagnostic["hierarchical_independent_fallback"] = independent
            diagnostic[
                "hierarchical_independent_single_fallback"
            ] = False
            diagnostic[
                "hierarchical_independent_reliability_threshold_passed"
            ] = (
                not independent
                or score >= independent_reliability_thresholds[kind]
            )
            source_health_passed = True
            if kind == "radar":
                source_health_passed = (
                    _as_float(diagnostic.get("radar_quality"))
                    >= self.radar_min_quality
                    and _as_float(
                        diagnostic.get(
                            "radar_point_support",
                            diagnostic.get("radar_support_ratio"),
                        )
                    )
                    >= self.radar_min_local_support
                )
            diagnostic["hierarchical_source_health_passed"] = (
                source_health_passed
            )
            eligible_for_consensus = (
                not independent or lidar_degraded
            )
            source_score_threshold = (
                independent_source_score_thresholds[kind]
                if independent
                else source_score_thresholds[kind]
            )
            if (
                eligible_for_consensus
                and source_health_passed
                and (
                    not independent
                    or score >= independent_reliability_thresholds[kind]
                )
                and _as_float(diagnostic.get("base_score"))
                >= source_score_threshold
                and not any(
                    oriented_iou(obstacle, base) >= self.max_base_iou
                    for base in base_kept
                )
            ):
                eligible[kind].append(
                    (obstacle, diagnostic, score, independent)
                )
            if independent:
                single_threshold = (
                    self.degraded_independent_camera_threshold
                    if kind == "camera"
                    else (
                        self.degraded_independent_radar_threshold
                        if camera_degraded
                        else self.lidar_only_independent_radar_threshold
                    )
                )
                single_limit = (
                    self.degraded_max_independent_camera_per_frame
                    if kind == "camera"
                    else (
                        self.degraded_max_independent_radar_per_frame
                        if camera_degraded
                        else self.lidar_only_max_independent_radar_per_frame
                    )
                )
                source_available = (
                    not camera_degraded
                    if kind == "camera"
                    else (
                        camera_degraded
                        or self.allow_independent_radar_when_camera_healthy
                    )
                )
                if (
                    lidar_degraded
                    and source_available
                    and source_health_passed
                    and single_limit > 0
                    and score >= single_threshold
                    and _as_float(diagnostic.get("base_score"))
                    >= source_score_threshold
                    and not any(
                        oriented_iou(obstacle, base)
                        >= self.max_base_iou
                        for base in base_kept
                    )
                ):
                    diagnostic[
                        "hierarchical_independent_single_fallback"
                    ] = True
                    ranked[kind].append(
                        (
                            score,
                            int(obstacle.obstacle_id),
                            replace(obstacle, score=score),
                            diagnostic,
                        )
                    )
                continue
            if (
                not source_health_passed
                or score < thresholds[kind]
                or _as_float(diagnostic.get("base_score"))
                < source_score_threshold
                or any(
                    oriented_iou(obstacle, base) >= self.max_base_iou
                    for base in base_kept
                )
            ):
                continue
            ranked[kind].append(
                (
                    score,
                    int(obstacle.obstacle_id),
                    replace(obstacle, score=score),
                    diagnostic,
                )
            )

        if lidar_degraded and not camera_degraded:
            consensus = []
            for (
                camera,
                camera_diagnostic,
                camera_score,
                camera_independent,
            ) in eligible["camera"]:
                best = None
                for (
                    radar,
                    radar_diagnostic,
                    radar_score,
                    radar_independent,
                ) in eligible["radar"]:
                    if (
                        camera_independent or radar_independent
                    ) and not (
                        camera_independent and radar_independent
                    ):
                        continue
                    overlap = oriented_iou(camera, radar)
                    if overlap < self.degraded_consensus_iou:
                        continue
                    rank = (
                        overlap,
                        min(
                            _as_float(camera_diagnostic.get("base_score")),
                            _as_float(radar_diagnostic.get("base_score")),
                        ),
                        min(camera_score, radar_score),
                    )
                    if best is None or rank > best[0]:
                        best = (rank, radar_diagnostic)
                if best is None:
                    continue
                overlap = best[0][0]
                consensus_score = max(
                    camera_score,
                    min(
                        _as_float(camera_diagnostic.get("base_score")),
                        _as_float(best[1].get("base_score")),
                    )
                    * overlap,
                )
                camera_diagnostic[
                    "hierarchical_cross_source_consensus"
                ] = True
                camera_diagnostic["hierarchical_cross_source_iou"] = overlap
                consensus.append(
                    (
                        1.0 + consensus_score,
                        int(camera.obstacle_id),
                        replace(camera, score=consensus_score),
                        camera_diagnostic,
                    )
                )
            consensus.sort(key=lambda item: (-item[0], item[1]))
            ranked["camera"] = (
                consensus[: self.degraded_max_consensus_per_frame]
                + ranked["camera"]
            )

        supplements = []
        regular_counts = {"camera": 0, "radar": 0}
        independent_counts = {"camera": 0, "radar": 0}
        independent_limits = {
            "camera": self.degraded_max_independent_camera_per_frame,
            "radar": self.degraded_max_independent_radar_per_frame,
        }
        consensus_count = 0
        for kind in ("camera", "radar"):
            ranked[kind].sort(key=lambda item: (-item[0], item[1]))
            for _, _, obstacle, diagnostic in ranked[kind]:
                is_consensus = bool(
                    diagnostic.get("hierarchical_cross_source_consensus")
                )
                if is_consensus:
                    if (
                        consensus_count
                        >= self.degraded_max_consensus_per_frame
                    ):
                        continue
                elif diagnostic.get(
                    "hierarchical_independent_single_fallback",
                    False,
                ):
                    if independent_counts[kind] >= independent_limits[kind]:
                        continue
                elif regular_counts[kind] >= limits[kind]:
                    continue
                if any(
                    oriented_iou(obstacle, previous[0])
                    >= self.supplement_nms_iou
                    for previous in supplements
                ):
                    continue
                diagnostic["hierarchical_accepted"] = True
                supplements.append((obstacle, diagnostic))
                if is_consensus:
                    consensus_count += 1
                elif diagnostic.get(
                    "hierarchical_independent_single_fallback",
                    False,
                ):
                    independent_counts[kind] += 1
                else:
                    regular_counts[kind] += 1

        kept = tuple(base_kept) + tuple(
            item[0] for item in supplements
        )
        updated = tuple(
            updated_by_id[int(obstacle.obstacle_id)]
            for obstacle in obstacles
        )
        return kept, updated
