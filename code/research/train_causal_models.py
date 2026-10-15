#!/usr/bin/env python3
"""Train fair simple and full causal reliability models on fixed Boreas splits."""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_boreas_detection_csv as evaluator  # noqa: E402
import fuse_boreas_trimodal_adaptive_gate as gate  # noqa: E402


SIMPLE_FEATURES = ["score", "distance", "area", "aspect_ratio"]
FULL_FEATURES = [
    "score", "distance", "area", "aspect_ratio",
    "temporal_support_all", "temporal_support_same_source",
    "vins_pose_available", "vins_temporal_alignment_used",
    "vins_temporal_support_all", "vins_temporal_support_same_source",
    "camera_sync_available", "radar_sync_available",
    "camera_semantic_support", "camera_support_iou", "camera_support_score",
    "radar_support", "radar_support_iou", "radar_support_score",
    "semantic_mask_available", "camera_semantic_mask_projected",
    "camera_semantic_mask_support", "camera_semantic_mask_quality",
    "camera_raw_projected", "camera_raw_support", "camera_raw_edge_support",
    "camera_raw_delta_us", "radar_raw_projected", "radar_raw_support",
    "radar_raw_intensity_support", "radar_raw_contrast_support",
    "radar_raw_delta_us",
]
ABSOLUTE_FEATURES = {"camera_raw_delta_us", "radar_raw_delta_us"}


def as_float(row, key, default=0.0):
    value = str(row.get(key, "")).strip()
    return default if not value else float(value)


def frame_token(row):
    return str(int(round(float(row["timestamp"]) * 1_000_000)))


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def matrix(rows, features, sources=None, mean=None, std=None):
    if sources is None:
        sources = sorted({row["source"] for row in rows})
    source_index = {value: index for index, value in enumerate(sources)}
    numeric = np.asarray(
        [
            [abs(as_float(row, key)) if key in ABSOLUTE_FEATURES else as_float(row, key) for key in features]
            for row in rows
        ],
        dtype=np.float64,
    )
    if mean is None:
        mean = numeric.mean(axis=0)
    if std is None:
        std = numeric.std(axis=0)
        std[std < 1e-6] = 1.0
    numeric = (numeric - mean) / std
    one_hot = np.zeros((len(rows), len(sources)), dtype=np.float64)
    for index, row in enumerate(rows):
        if row["source"] in source_index:
            one_hot[index, source_index[row["source"]]] = 1.0
    return np.hstack([np.ones((len(rows), 1)), numeric, one_hot]), sources, mean, std


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def fit(rows, features, label_key, epochs, learning_rate, l2):
    x, sources, mean, std = matrix(rows, features)
    y = np.asarray([as_float(row, label_key) for row in rows], dtype=np.float64)
    positive = y.sum()
    sample_weight = np.where(y > 0.5, (len(y) - positive) / max(positive, 1.0), 1.0)
    sample_weight /= sample_weight.mean()
    weights = np.zeros(x.shape[1], dtype=np.float64)
    for _ in range(epochs):
        predicted = sigmoid(x @ weights)
        gradient = x.T @ ((predicted - y) * sample_weight) / len(y)
        gradient[1:] += l2 * weights[1:]
        weights -= learning_rate * gradient
    return {"features": features, "sources": sources, "mean": mean, "std": std, "weights": weights}


def score(rows, model):
    x, _, _, _ = matrix(
        rows, model["features"], model["sources"], model["mean"], model["std"]
    )
    values = sigmoid(x @ model["weights"])
    output = []
    for row, value in zip(rows, values):
        item = dict(row)
        item["learned_reliability_score"] = f"{float(value):.10f}"
        output.append(item)
    return output


def eligible(rows, camera_min_score, radar_min_score, disabled_source=""):
    output = []
    for row in rows:
        source = row.get("source", "")
        if source == disabled_source:
            continue
        if source == "camera_lidar":
            valid = as_float(row, "camera_sync_available") > 0 and as_float(row, "score") >= camera_min_score
        elif source == "radar_lirafusion":
            valid = as_float(row, "radar_sync_available") > 0 and as_float(row, "score") >= radar_min_score
        else:
            valid = False
        if valid:
            output.append(row)
    return output


def select(scored_rows, base_frames, threshold, max_camera, max_radar, max_base_iou, nms_iou):
    limits = {"camera_lidar": max_camera, "radar_lirafusion": max_radar}
    counts = defaultdict(Counter)
    accepted = defaultdict(list, {key: list(value) for key, value in base_frames.items()})
    candidates = [row for row in scored_rows if as_float(row, "learned_reliability_score") >= threshold]
    candidates.sort(
        key=lambda row: (
            gate.row_timestamp(row),
            -as_float(row, "learned_reliability_score"),
            row["source"],
            row.get("candidate_id", ""),
        )
    )
    selected = []
    for row in candidates:
        timestamp = gate.row_timestamp(row)
        source = row["source"]
        if counts[timestamp][source] >= limits[source]:
            continue
        box = gate.row_box(row, "learned_reliability_score")
        if gate.max_overlap(box, accepted[timestamp]) > max_base_iou:
            continue
        if any(evaluator.bev_iou(box, other) >= nms_iou for other in accepted[timestamp]):
            continue
        selected.append(row)
        counts[timestamp][source] += 1
        accepted[timestamp].append(box)
    return selected


def selected_frames(base_frames, selected):
    frames = defaultdict(list, {key: list(value) for key, value in base_frames.items()})
    for row in selected:
        frames[gate.row_timestamp(row)].append(gate.row_box(row, "learned_reliability_score"))
    return frames


def subset(frames, tokens):
    timestamps = {f"{int(token) / 1_000_000:.6f}" for token in tokens}
    return defaultdict(list, {key: value for key, value in frames.items() if key in timestamps})


def evaluate(gt_frames, base_frames, selected, tokens, iou):
    return evaluator.evaluate(subset(gt_frames, tokens), subset(selected_frames(base_frames, selected), tokens), iou)


def threshold_grid(rows):
    values = np.asarray([as_float(row, "learned_reliability_score") for row in rows])
    candidates = set(np.linspace(0.05, 0.95, 19))
    for quantile in np.linspace(0.02, 0.98, 49):
        candidates.add(float(np.quantile(values, quantile)))
    return sorted(candidates)


def choose_threshold(scored, gt_frames, base_frames, calibration_tokens, args):
    sweep = []
    for threshold in threshold_grid(scored):
        chosen = select(scored, base_frames, threshold, args.max_camera_per_frame, args.max_radar_per_frame, args.max_base_iou, args.nms_iou)
        metrics = evaluate(gt_frames, base_frames, chosen, calibration_tokens, 0.10)
        sweep.append({"candidate_threshold": threshold, **metrics, "precision_floor_ok": int(metrics["precision"] >= args.precision_floor)})
    valid = [row for row in sweep if row["precision_floor_ok"]] or sweep
    best_f1 = max(row["f1"] for row in valid)
    eligible_rows = [row for row in valid if row["f1"] >= best_f1 - args.f1_tolerance]
    best = max(eligible_rows, key=lambda row: (row["recall"], row["precision"], -row["candidate_threshold"]))
    return float(best["candidate_threshold"]), sweep


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_prediction(path, base_fields, base_rows, selected, method):
    fields = list(base_fields)
    for key in ("timestamp", "method", "frame_id", "obstacle_id", "center_x", "center_y", "size_x", "size_y", "cell_count", "score", "yaw", "source", "learned_reliability_score"):
        if key not in fields:
            fields.append(key)
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in base_rows:
            writer.writerow({key: row.get(key, "") for key in fields})
        for index, row in enumerate(selected):
            output = {key: "" for key in fields}
            output.update({
                "timestamp": row.get("raw_timestamp") or row["timestamp"],
                "method": method,
                "frame_id": row.get("frame_id", "boreas_lidar"),
                "obstacle_id": f"{row['source']}_causal_{index}",
                "center_x": row["center_x"], "center_y": row["center_y"],
                "size_x": row["size_x"], "size_y": row["size_y"],
                "cell_count": row.get("cell_count", ""),
                "score": row["learned_reliability_score"],
                "yaw": row.get("yaw", "0"), "source": row["source"],
                "learned_reliability_score": row["learned_reliability_score"],
            })
            writer.writerow(output)


def serialize_model(model, threshold, metadata):
    return {
        "format_version": 2,
        "model_type": "standardized_logistic_candidate_reliability_causal",
        "features": model["features"], "sources": model["sources"],
        "mean": model["mean"].tolist(), "std": model["std"].tolist(),
        "weights": model["weights"].tolist(), "candidate_threshold": threshold,
        **metadata,
    }


def run_model(name, features, feature_rows, base_fields, base_rows, gt_frames, splits, args, out_dir, disabled_source=""):
    raw_pool = eligible(feature_rows, args.camera_min_score, args.radar_min_score, disabled_source)
    train_rows = [row for row in raw_pool if frame_token(row) in splits["model_development"]]
    if any(frame_token(row) not in splits["model_development"] for row in train_rows):
        raise AssertionError("non-training row reached model fitting")
    model = fit(train_rows, features, args.label_key, args.epochs, args.learning_rate, args.l2)
    scored = score(raw_pool, model)
    threshold, sweep = choose_threshold(scored, gt_frames, gate.rows_to_frames(base_rows), splits["calibration"], args)
    selected = select(scored, gate.rows_to_frames(base_rows), threshold, args.max_camera_per_frame, args.max_radar_per_frame, args.max_base_iou, args.nms_iou)
    prediction_path = out_dir / f"{name}_detections.csv"
    write_prediction(prediction_path, base_fields, base_rows, selected, name)
    write_csv(out_dir / f"{name}_threshold_sweep.csv", sweep)
    write_csv(out_dir / f"{name}_selected_candidates.csv", selected)
    summary = []
    for split_name, tokens in {**splits, "all": set().union(*splits.values())}.items():
        for iou in (0.10, 0.25, 0.50):
            metrics = evaluate(gt_frames, gate.rows_to_frames(base_rows), selected, tokens, iou)
            summary.append({"method": name, "split": split_name, "iou_threshold": iou, **metrics})
    write_csv(out_dir / f"{name}_multi_iou_summary.csv", summary)
    fit_token_hash = hashlib.sha256("\n".join(sorted(splits["model_development"], key=int)).encode()).hexdigest()
    payload = serialize_model(model, threshold, {
        "fit_split": "model_development", "fit_frames": len(splits["model_development"]),
        "calibration_split": "calibration", "calibration_frames": len(splits["calibration"]),
        "holdout_access_during_fit_or_selection": False,
        "fit_frame_keys_sha256": fit_token_hash,
        "eligible_candidates": len(raw_pool), "fit_candidates": len(train_rows),
        "selected_candidates": len(selected), "disabled_source": disabled_source,
        "common_candidate_policy": {
            "camera_min_score": args.camera_min_score, "radar_min_score": args.radar_min_score,
            "max_camera_per_frame": args.max_camera_per_frame, "max_radar_per_frame": args.max_radar_per_frame,
            "max_base_iou": args.max_base_iou, "nms_iou": args.nms_iou,
        },
    })
    (out_dir / f"{name}_model.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"name": name, "prediction": str(prediction_path), "threshold": threshold, "selected": len(selected), "summary": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-csv", required=True)
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--feature-csv", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--label-key", default="label_iou10")
    parser.add_argument("--camera-min-score", type=float, default=0.25)
    parser.add_argument("--radar-min-score", type=float, default=0.30)
    parser.add_argument("--max-camera-per-frame", type=int, default=1)
    parser.add_argument("--max-radar-per-frame", type=int, default=4)
    parser.add_argument("--max-base-iou", type=float, default=0.03)
    parser.add_argument("--nms-iou", type=float, default=0.10)
    parser.add_argument("--precision-floor", type=float, default=0.86)
    parser.add_argument("--f1-tolerance", type=float, default=0.00025)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.01)
    args = parser.parse_args()

    base_fields, base_rows = read_rows(args.base_csv)
    _, feature_rows = read_rows(args.feature_csv)
    gt_frames = evaluator.read_boxes(args.gt_csv, is_gt=True)
    with Path(args.split_manifest).open(encoding="utf-8", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    splits = defaultdict(set)
    for row in manifest:
        splits[row["split"]].add(row["frame_key"])
    expected = {"model_development": 250, "calibration": 100, "holdout_test": 150}
    if {key: len(splits[key]) for key in expected} != expected:
        raise ValueError("unexpected fixed split sizes")
    if any(splits[left] & splits[right] for left in expected for right in expected if left < right):
        raise AssertionError("split overlap")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        run_model("SimpleGeometryCausal", SIMPLE_FEATURES, feature_rows, base_fields, base_rows, gt_frames, splits, args, out_dir),
        run_model("RGCF_B2_Causal", FULL_FEATURES, feature_rows, base_fields, base_rows, gt_frames, splits, args, out_dir),
        run_model("RGCF_B2_NoTemporal", [f for f in FULL_FEATURES if not (f.startswith("temporal_") or f.startswith("vins_"))], feature_rows, base_fields, base_rows, gt_frames, splits, args, out_dir),
        run_model("RGCF_B2_NoCamera", FULL_FEATURES, feature_rows, base_fields, base_rows, gt_frames, splits, args, out_dir, "camera_lidar"),
        run_model("RGCF_B2_NoRadar", FULL_FEATURES, feature_rows, base_fields, base_rows, gt_frames, splits, args, out_dir, "radar_lirafusion"),
    ]
    (out_dir / "run_summary.json").write_text(json.dumps(runs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{key: row[key] for key in ("name", "threshold", "selected")} for row in runs], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
