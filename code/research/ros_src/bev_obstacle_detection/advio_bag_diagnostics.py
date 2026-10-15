"""Diagnostics for ADVIO-derived ROS bags."""

import csv
import json
import math
from pathlib import Path

import numpy as np


ADVIO_SUMMARY_FIELDS = (
    "sequence_id",
    "duration",
    "imu_messages",
    "gt_messages",
    "imu_mean_hz",
    "gt_mean_hz",
    "imu_gt_p95_sync_s",
    "matched_poses",
    "ate_rmse",
    "ate_mean",
    "ate_p95",
    "rpe_translation_rmse",
    "gt_path_length",
    "estimate_aligned_path_length",
)


def summarize_stamps(stamps):
    values = np.asarray(stamps, dtype=float)
    if values.size == 0:
        return {
            "count": 0,
            "start_time": None,
            "end_time": None,
            "duration": 0.0,
            "mean_hz": 0.0,
            "median_dt": None,
            "min_dt": None,
            "max_dt": None,
            "p95_dt": None,
        }
    values = np.sort(values)
    duration = float(values[-1] - values[0])
    intervals = np.diff(values)
    mean_hz = float((values.size - 1) / duration) if values.size > 1 and duration > 0.0 else 0.0
    if intervals.size == 0:
        return {
            "count": int(values.size),
            "start_time": float(values[0]),
            "end_time": float(values[-1]),
            "duration": duration,
            "mean_hz": mean_hz,
            "median_dt": None,
            "min_dt": None,
            "max_dt": None,
            "p95_dt": None,
        }
    return {
        "count": int(values.size),
        "start_time": float(values[0]),
        "end_time": float(values[-1]),
        "duration": duration,
        "mean_hz": mean_hz,
        "median_dt": float(np.median(intervals)),
        "min_dt": float(np.min(intervals)),
        "max_dt": float(np.max(intervals)),
        "p95_dt": float(np.percentile(intervals, 95)),
    }


def nearest_time_offsets(reference_stamps, target_stamps):
    reference = np.asarray(reference_stamps, dtype=float)
    target = np.asarray(target_stamps, dtype=float)
    if reference.size == 0 or target.size == 0:
        return np.asarray([], dtype=float)
    reference = np.sort(reference)
    target = np.sort(target)
    indices = np.searchsorted(target, reference, side="left")
    indices = np.clip(indices, 0, target.size - 1)
    previous = np.maximum(indices - 1, 0)
    choose_previous = np.abs(target[previous] - reference) < np.abs(target[indices] - reference)
    nearest = np.where(choose_previous, previous, indices)
    return np.abs(target[nearest] - reference)


def summarize_offsets(offsets):
    values = np.asarray(offsets, dtype=float)
    if values.size == 0:
        return {
            "count": 0,
            "mean_abs_offset": None,
            "median_abs_offset": None,
            "p95_abs_offset": None,
            "max_abs_offset": None,
        }
    return {
        "count": int(values.size),
        "mean_abs_offset": float(np.mean(values)),
        "median_abs_offset": float(np.median(values)),
        "p95_abs_offset": float(np.percentile(values, 95)),
        "max_abs_offset": float(np.max(values)),
    }


def summarize_positions(positions):
    points = np.asarray(positions, dtype=float)
    if points.size == 0:
        return {
            "count": 0,
            "path_length": 0.0,
            "displacement": 0.0,
            "min_z": None,
            "max_z": None,
        }
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("positions must have shape Nx3")
    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1) if diffs.size else np.asarray([], dtype=float)
    displacement = float(np.linalg.norm(points[-1] - points[0])) if points.shape[0] > 1 else 0.0
    return {
        "count": int(points.shape[0]),
        "path_length": float(np.sum(segment_lengths)) if segment_lengths.size else 0.0,
        "displacement": displacement,
        "min_z": float(np.min(points[:, 2])),
        "max_z": float(np.max(points[:, 2])),
    }


def match_pose_rows_by_timestamp(reference_rows, estimate_rows, max_offset=0.02):
    reference = np.asarray(reference_rows, dtype=float)
    estimate = np.asarray(estimate_rows, dtype=float)
    if reference.size == 0 or estimate.size == 0:
        return np.zeros((0, 8), dtype=float)
    if reference.ndim != 2 or reference.shape[1] < 4:
        raise ValueError("reference rows must contain timestamp,x,y,z")
    if estimate.ndim != 2 or estimate.shape[1] < 4:
        raise ValueError("estimate rows must contain timestamp,x,y,z")
    indices = np.searchsorted(estimate[:, 0], reference[:, 0], side="left")
    indices = np.clip(indices, 0, estimate.shape[0] - 1)
    previous = np.maximum(indices - 1, 0)
    choose_previous = np.abs(estimate[previous, 0] - reference[:, 0]) < np.abs(estimate[indices, 0] - reference[:, 0])
    nearest = np.where(choose_previous, previous, indices)
    offsets = estimate[nearest, 0] - reference[:, 0]
    keep = np.abs(offsets) <= float(max_offset)
    return np.column_stack((reference[keep, 0], reference[keep, 1:4], estimate[nearest[keep], 1:4], offsets[keep]))


def align_umeyama(reference_xyz, estimate_xyz, with_scale=True):
    reference = np.asarray(reference_xyz, dtype=float)
    estimate = np.asarray(estimate_xyz, dtype=float)
    if reference.shape != estimate.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("reference and estimate positions must both have shape Nx3")
    if reference.shape[0] < 3:
        raise ValueError("at least three matched poses are required")

    ref_mean = np.mean(reference, axis=0)
    est_mean = np.mean(estimate, axis=0)
    ref_centered = reference - ref_mean
    est_centered = estimate - est_mean
    covariance = ref_centered.T.dot(est_centered) / float(reference.shape[0])
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u.dot(vt)) < 0:
        correction[-1, -1] = -1.0
    rotation = u.dot(correction).dot(vt)
    if with_scale:
        variance = np.mean(np.sum(est_centered * est_centered, axis=1))
        scale = float(np.sum(singular_values * np.diag(correction)) / variance) if variance > 0.0 else 1.0
    else:
        scale = 1.0
    translation = ref_mean - scale * rotation.dot(est_mean)
    aligned = (scale * rotation.dot(estimate.T)).T + translation
    return aligned, {"scale": scale, "rotation": rotation, "translation": translation}


def trajectory_error_summary(reference_xyz, estimate_xyz, timestamps=None, with_scale=True):
    reference = np.asarray(reference_xyz, dtype=float)
    estimate = np.asarray(estimate_xyz, dtype=float)
    aligned, transform = align_umeyama(reference, estimate, with_scale=with_scale)
    errors = np.linalg.norm(aligned - reference, axis=1)
    ref_delta = np.diff(reference, axis=0)
    est_delta = np.diff(aligned, axis=0)
    rpe = np.linalg.norm(est_delta - ref_delta, axis=1) if ref_delta.size else np.asarray([], dtype=float)
    summary = {
        "matched_poses": int(reference.shape[0]),
        "alignment_scale": float(transform["scale"]),
        "ate_rmse": float(np.sqrt(np.mean(errors * errors))),
        "ate_mean": float(np.mean(errors)),
        "ate_median": float(np.median(errors)),
        "ate_p95": float(np.percentile(errors, 95)),
        "ate_max": float(np.max(errors)),
        "rpe_translation_rmse": float(np.sqrt(np.mean(rpe * rpe))) if rpe.size else 0.0,
        "rpe_translation_mean": float(np.mean(rpe)) if rpe.size else 0.0,
        "gt_path_length": summarize_positions(reference)["path_length"],
        "estimate_aligned_path_length": summarize_positions(aligned)["path_length"],
    }
    if timestamps is not None and len(timestamps) > 1:
        stamp_values = np.asarray(timestamps, dtype=float)
        summary["duration"] = float(stamp_values[-1] - stamp_values[0])
    return summary


def summarize_vins_output_timing(timestamps, reference_start=0.0, reference_end=None, expected_rate_hz=10.0):
    values = np.asarray(timestamps, dtype=float)
    reference_start = float(reference_start)
    reference_end = float(reference_end) if reference_end is not None else (float(np.max(values)) if values.size else reference_start)
    reference_duration = max(0.0, reference_end - reference_start)
    if values.size == 0:
        return {
            "output_rows": 0,
            "first_output_time_s": None,
            "last_output_time_s": None,
            "first_output_delay_s": None,
            "output_duration_s": 0.0,
            "coverage_ratio": 0.0,
            "mean_output_hz": 0.0,
            "median_output_gap_s": None,
            "p95_output_gap_s": None,
            "max_output_gap_s": None,
            "expected_rows": int(round(reference_duration * float(expected_rate_hz))) if expected_rate_hz > 0 else 0,
            "output_row_ratio": 0.0,
        }
    values = np.sort(values)
    output_duration = float(values[-1] - values[0]) if values.size > 1 else 0.0
    intervals = np.diff(values)
    expected_rows = int(round(reference_duration * float(expected_rate_hz))) if expected_rate_hz > 0 else 0
    return {
        "output_rows": int(values.size),
        "first_output_time_s": float(values[0]),
        "last_output_time_s": float(values[-1]),
        "first_output_delay_s": float(values[0] - reference_start),
        "output_duration_s": output_duration,
        "coverage_ratio": float(output_duration / reference_duration) if reference_duration > 0.0 else 0.0,
        "mean_output_hz": float((values.size - 1) / output_duration) if values.size > 1 and output_duration > 0.0 else 0.0,
        "median_output_gap_s": float(np.median(intervals)) if intervals.size else None,
        "p95_output_gap_s": float(np.percentile(intervals, 95)) if intervals.size else None,
        "max_output_gap_s": float(np.max(intervals)) if intervals.size else None,
        "expected_rows": expected_rows,
        "output_row_ratio": float(values.size / expected_rows) if expected_rows > 0 else 0.0,
    }


def finite_or_empty(value):
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_diagnostics(summary, output_dir, stem="advio_bag_diagnostics"):
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "{}.json".format(stem)
    csv_path = output / "{}.csv".format(stem)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["section", "metric", "value"])
        for section, values in sorted(summary.items()):
            if isinstance(values, dict):
                for key, value in sorted(values.items()):
                    writer.writerow([section, key, finite_or_empty(value)])
            else:
                writer.writerow([section, "value", finite_or_empty(values)])
    return json_path, csv_path


def load_json(path):
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def advio_sequence_summary(trajectory_summary, bag_diagnostics=None):
    trajectory = dict(trajectory_summary)
    bag = dict(bag_diagnostics or {})
    return {
        "sequence_id": trajectory.get("sequence_id", ""),
        "duration": trajectory.get("duration", ""),
        "imu_messages": (bag.get("imu") or {}).get("count", ""),
        "gt_messages": (bag.get("ground_truth") or {}).get("count", ""),
        "imu_mean_hz": (bag.get("imu") or {}).get("mean_hz", ""),
        "gt_mean_hz": (bag.get("ground_truth") or {}).get("mean_hz", ""),
        "imu_gt_p95_sync_s": (bag.get("imu_to_ground_truth_sync") or {}).get("p95_abs_offset", ""),
        "matched_poses": trajectory.get("matched_poses", ""),
        "ate_rmse": trajectory.get("ate_rmse", ""),
        "ate_mean": trajectory.get("ate_mean", ""),
        "ate_p95": trajectory.get("ate_p95", ""),
        "rpe_translation_rmse": trajectory.get("rpe_translation_rmse", ""),
        "gt_path_length": trajectory.get("gt_path_length", ""),
        "estimate_aligned_path_length": trajectory.get("estimate_aligned_path_length", ""),
    }


def write_advio_summary_csv(rows, output_path):
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ADVIO_SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: finite_or_empty(row.get(field, "")) for field in ADVIO_SUMMARY_FIELDS})
    return output
