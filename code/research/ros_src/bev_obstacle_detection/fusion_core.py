"""Core VINS-guided BEV fusion primitives."""

from dataclasses import dataclass

import numpy as np

from bev_obstacle_detection.geometry import inverse_transform, transform_points


@dataclass(frozen=True)
class FusionHealthConfig:
    max_age_sec: float = 0.25
    min_score: float = 0.0

    def validate(self):
        if self.max_age_sec <= 0.0:
            raise ValueError("max_age_sec must be positive")
        if self.min_score < 0.0 or self.min_score > 1.0:
            raise ValueError("min_score must be in [0, 1]")


def transform_from_source_to_target(source_world_from_body, target_world_from_body):
    source_world_from_body = np.asarray(source_world_from_body, dtype=float)
    target_world_from_body = np.asarray(target_world_from_body, dtype=float)
    if source_world_from_body.shape != (4, 4):
        raise ValueError("source_world_from_body must be 4x4")
    if target_world_from_body.shape != (4, 4):
        raise ValueError("target_world_from_body must be 4x4")
    return inverse_transform(target_world_from_body) @ source_world_from_body


def align_points_to_target_frame(points_xyz, source_world_from_body, target_world_from_body):
    transform = transform_from_source_to_target(source_world_from_body, target_world_from_body)
    return transform_points(points_xyz, transform)


def health_score(age_sec, dropout=False, config=None):
    config = config or FusionHealthConfig()
    config.validate()
    if dropout or age_sec is None:
        return 0.0
    age = float(age_sec)
    if age < 0.0:
        return 0.0
    score = 1.0 - min(age, config.max_age_sec) / config.max_age_sec
    return max(config.min_score, min(1.0, score))


def normalize_weights(scores):
    total = sum(max(0.0, float(value)) for value in scores.values())
    if total <= 0.0:
        return {name: 0.0 for name in scores}
    return {name: max(0.0, float(value)) / total for name, value in scores.items()}


def adaptive_fusion_weights(sensor_ages, dropout_sensors=(), priors=None, config=None):
    dropout_sensors = set(dropout_sensors or ())
    priors = priors or {}
    scores = {}
    for sensor, age in sensor_ages.items():
        score = health_score(age, dropout=sensor in dropout_sensors, config=config)
        scores[sensor] = score * float(priors.get(sensor, 1.0))
    return normalize_weights(scores)


def weighted_average_feature_maps(feature_maps_by_source, weights, layer_names=None):
    if not feature_maps_by_source:
        raise ValueError("feature_maps_by_source must be non-empty")

    layer_names = tuple(layer_names or next(iter(feature_maps_by_source.values())).keys())
    fused = {}
    for layer in layer_names:
        numerator = None
        denominator = 0.0
        for source, maps in feature_maps_by_source.items():
            if layer not in maps:
                continue
            weight = max(0.0, float(weights.get(source, 0.0)))
            array = np.asarray(maps[layer], dtype=np.float32)
            numerator = np.zeros_like(array, dtype=np.float32) if numerator is None else numerator
            numerator += weight * array
            denominator += weight
        if numerator is None:
            raise ValueError("missing feature layer in all sources: {}".format(layer))
        fused[layer] = numerator / denominator if denominator > 0.0 else numerator
    return fused
