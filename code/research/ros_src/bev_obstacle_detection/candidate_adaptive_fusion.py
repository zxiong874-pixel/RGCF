"""Candidate-level adaptive fusion of local BEV sensor evidence."""

from dataclasses import replace

import numpy as np


MODALITIES = ("lidar", "camera", "radar")


def _integral_image(mask):
    values = np.asarray(mask, dtype=np.int32)
    return np.pad(
        values.cumsum(axis=0).cumsum(axis=1),
        ((1, 0), (1, 0)),
        mode="constant",
    )


def _rectangle_sum(integral, row_min, row_max, col_min, col_max):
    return int(
        integral[row_max + 1, col_max + 1]
        - integral[row_min, col_max + 1]
        - integral[row_max + 1, col_min]
        + integral[row_min, col_min]
    )


def _candidate_support_ratio_from_integrals(
    obstacle,
    shape,
    candidate_integral,
    supported_integral,
):
    row_min = max(0, int(obstacle.row_min))
    row_max = min(shape[0] - 1, int(obstacle.row_max))
    col_min = max(0, int(obstacle.col_min))
    col_max = min(shape[1] - 1, int(obstacle.col_max))
    if row_max < row_min or col_max < col_min:
        return 0.0
    candidate_cells = _rectangle_sum(
        candidate_integral,
        row_min,
        row_max,
        col_min,
        col_max,
    )
    if candidate_cells <= 0:
        return 0.0
    supported_cells = _rectangle_sum(
        supported_integral,
        row_min,
        row_max,
        col_min,
        col_max,
    )
    return float(
        np.clip(
            supported_cells / float(candidate_cells),
            0.0,
            1.0,
        )
    )


def candidate_support_ratio(obstacle, candidate_occupancy, support_map):
    """Return the fraction of candidate cells supported by one modality."""
    candidate = np.asarray(candidate_occupancy) > 0
    support = np.asarray(support_map) > 0
    if candidate.ndim != 2:
        raise ValueError("candidate_occupancy must be a 2D array")
    if support.shape != candidate.shape:
        raise ValueError("support_map must have the same shape as candidate_occupancy")

    row_min = max(0, int(obstacle.row_min))
    row_max = min(candidate.shape[0] - 1, int(obstacle.row_max))
    col_min = max(0, int(obstacle.col_min))
    col_max = min(candidate.shape[1] - 1, int(obstacle.col_max))
    if row_max < row_min or col_max < col_min:
        return 0.0

    candidate_crop = candidate[row_min : row_max + 1, col_min : col_max + 1]
    candidate_cells = int(np.count_nonzero(candidate_crop))
    if candidate_cells <= 0:
        return 0.0
    support_crop = support[row_min : row_max + 1, col_min : col_max + 1]
    supported_cells = int(np.count_nonzero(candidate_crop & support_crop))
    return float(np.clip(supported_cells / float(candidate_cells), 0.0, 1.0))


def local_adaptive_weights(
    support_ratios,
    global_weights,
    support_floor=0.05,
    support_exponent=1.0,
    participation=None,
):
    """Redistribute frame-level weights according to local candidate evidence."""
    floor = float(np.clip(support_floor, 0.0, 1.0))
    exponent = float(support_exponent)
    if exponent <= 0.0:
        raise ValueError("support_exponent must be positive")

    scores = {}
    for modality in MODALITIES:
        if participation is not None and not bool(
            participation.get(modality, False)
        ):
            scores[modality] = 0.0
            continue
        global_weight = max(0.0, float(global_weights.get(modality, 0.0)))
        support = float(np.clip(support_ratios.get(modality, 0.0), 0.0, 1.0))
        local_attention = floor + (1.0 - floor) * support
        scores[modality] = global_weight * local_attention**exponent

    total = sum(scores.values())
    if total <= 1.0e-12:
        return {modality: 0.0 for modality in MODALITIES}
    return {modality: score / total for modality, score in scores.items()}


def candidate_reliability_score(
    base_score,
    support_ratios,
    local_weights,
    qualities,
    evidence_mix=0.35,
    quality_mix=0.0,
):
    """Confirm detector confidence without erasing its absolute uncertainty."""
    evidence_fraction = float(np.clip(evidence_mix, 0.0, 1.0))
    quality_fraction = float(np.clip(quality_mix, 0.0, 1.0))
    local_support = sum(
        float(local_weights.get(modality, 0.0))
        * float(np.clip(support_ratios.get(modality, 0.0), 0.0, 1.0))
        for modality in MODALITIES
    )
    local_quality = sum(
        float(local_weights.get(modality, 0.0))
        * float(np.clip(qualities.get(modality, 0.0), 0.0, 1.0))
        for modality in MODALITIES
    )
    base = float(np.clip(base_score, 0.0, 1.0))
    evidence_scale = (1.0 - evidence_fraction) + evidence_fraction * local_support
    quality_scale = (1.0 - quality_fraction) + quality_fraction * local_quality
    return float(np.clip(base * evidence_scale * quality_scale, 0.0, 1.0))


def adapt_obstacle_candidates(
    obstacles,
    candidate_occupancy,
    support_maps,
    qualities,
    global_weights,
    threshold=0.0,
    evidence_mix=0.35,
    quality_mix=0.0,
    support_floor=0.05,
    support_exponent=1.0,
    participation_min_quality=0.05,
    participation_min_support=0.05,
    support_ratios_by_obstacle=None,
):
    """Score and optionally filter obstacle candidates using local tri-modal evidence."""
    threshold = float(threshold)
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be in [0, 1]")
    min_quality = float(np.clip(participation_min_quality, 0.0, 1.0))
    min_support = float(np.clip(participation_min_support, 0.0, 1.0))
    candidate_binary = np.asarray(candidate_occupancy) > 0
    if candidate_binary.ndim != 2:
        raise ValueError("candidate_occupancy must be a 2D array")
    candidate_integral = _integral_image(candidate_binary)
    supported_integrals = {}
    for modality in MODALITIES:
        support = np.asarray(
            support_maps.get(
                modality,
                np.zeros_like(candidate_binary),
            )
        ) > 0
        if support.shape != candidate_binary.shape:
            raise ValueError(
                "support_map must have the same shape as candidate_occupancy"
            )
        supported_integrals[modality] = _integral_image(
            candidate_binary & support
        )

    accepted = []
    diagnostics = []
    for obstacle in obstacles:
        exact_ratios = (
            dict(support_ratios_by_obstacle or {}).get(
                int(obstacle.obstacle_id)
            )
        )
        support_ratios = (
            {
                modality: float(
                    np.clip(exact_ratios.get(modality, 0.0), 0.0, 1.0)
                )
                for modality in MODALITIES
            }
            if exact_ratios is not None
            else {
                modality: _candidate_support_ratio_from_integrals(
                    obstacle,
                    candidate_binary.shape,
                    candidate_integral,
                    supported_integrals[modality],
                )
                for modality in MODALITIES
            }
        )
        participation = {
            modality: (
                float(qualities.get(modality, 0.0)) >= min_quality
                and support_ratios[modality] >= min_support
            )
            for modality in MODALITIES
        }
        local_weights = local_adaptive_weights(
            support_ratios,
            global_weights,
            support_floor=support_floor,
            support_exponent=support_exponent,
            participation=participation,
        )
        reliability = candidate_reliability_score(
            obstacle.score,
            support_ratios,
            local_weights,
            qualities,
            evidence_mix=evidence_mix,
            quality_mix=quality_mix,
        )
        participating = tuple(
            modality
            for modality in MODALITIES
            if participation[modality]
        )
        keep = reliability >= threshold
        diagnostics.append(
            {
                "obstacle_id": int(obstacle.obstacle_id),
                "base_score": float(obstacle.score),
                "reliability_score": reliability,
                "accepted": keep,
                "participating_modalities": "+".join(participating) or "none",
                "active_modality_count": len(participating),
                **{
                    "{}_support_ratio".format(modality): support_ratios[modality]
                    for modality in MODALITIES
                },
                **{
                    "{}_local_weight".format(modality): local_weights[modality]
                    for modality in MODALITIES
                },
            }
        )
        if keep:
            accepted.append(replace(obstacle, score=reliability))
    return tuple(accepted), tuple(diagnostics)


def occupancy_for_candidates(candidate_occupancy, obstacles):
    """Keep occupied cells belonging to accepted candidate bounding regions."""
    occupancy = np.asarray(candidate_occupancy)
    if occupancy.ndim != 2:
        raise ValueError("candidate_occupancy must be a 2D array")
    kept = np.zeros_like(occupancy)
    for obstacle in obstacles:
        row_min = max(0, int(obstacle.row_min))
        row_max = min(occupancy.shape[0] - 1, int(obstacle.row_max))
        col_min = max(0, int(obstacle.col_min))
        col_max = min(occupancy.shape[1] - 1, int(obstacle.col_max))
        if row_max < row_min or col_max < col_min:
            continue
        kept[row_min : row_max + 1, col_min : col_max + 1] = occupancy[
            row_min : row_max + 1,
            col_min : col_max + 1,
        ]
    return kept
