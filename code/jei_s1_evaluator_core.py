#!/usr/bin/env python3
"""JEI S1: audit and uniformly evaluate frozen Boreas detection CSV files."""

import csv
import hashlib
import json
import math
import os
import platform
import sys
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
TASK = Path(__file__).resolve().parents[1]
RESULTS = TASK / "results"
FIGURES = TASK / "figures"
IOU_THRESHOLDS = (0.10, 0.25, 0.50)
DISTANCE_EDGES = (0.0, 3.0, 6.0, 10.0, 15.0, 20.0, 30.0, 50.0, float("inf"))
RANDOM_SEED = 20260902
TIMESTAMP_TOLERANCE_US = 1


@dataclass
class Box:
    frame_key: str
    row_id: int
    center_x: float
    center_y: float
    size_x: float
    size_y: float
    yaw: float
    score: float
    object_id: str
    source: str
    mapping: str

    @property
    def distance(self):
        return math.hypot(self.center_x, self.center_y)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_float(row, name, default=None):
    value = str(row.get(name, "")).strip()
    if not value and default is not None:
        return float(default)
    return float(value)


def token_from_gt_row(row):
    stem = Path(str(row.get("source_path", ""))).stem
    if stem.isdigit():
        return stem
    timestamp = Decimal(str(row["timestamp"]))
    return str(int((timestamp * Decimal(1000000)).to_integral_value(rounding=ROUND_HALF_UP)))


def timestamp_to_microseconds(value):
    try:
        return int((Decimal(str(value)) * Decimal(1000000)).to_integral_value(rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError):
        return None


def load_gt(path):
    frames = defaultdict(list)
    tokens = set()
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row_id, row in enumerate(csv.DictReader(stream)):
            token = token_from_gt_row(row)
            tokens.add(token)
            frames[token].append(
                Box(
                    frame_key=token,
                    row_id=row_id,
                    center_x=parse_float(row, "center_x"),
                    center_y=parse_float(row, "center_y"),
                    size_x=parse_float(row, "size_x"),
                    size_y=parse_float(row, "size_y"),
                    yaw=parse_float(row, "yaw", 0.0),
                    score=1.0,
                    object_id=str(row.get("object_id", row_id)),
                    source=str(row.get("label", "obstacle")),
                    mapping="source_path_integer_microseconds",
                )
            )
    return dict(frames), sorted(tokens, key=int)


def nearest_token(timestamp_us, sorted_tokens):
    values = [int(item) for item in sorted_tokens]
    index = bisect_left(values, timestamp_us)
    candidates = []
    if index < len(values):
        candidates.append(values[index])
    if index:
        candidates.append(values[index - 1])
    if not candidates:
        return None, None
    nearest = min(candidates, key=lambda item: abs(item - timestamp_us))
    return str(nearest), abs(nearest - timestamp_us)


def load_detections(path, gt_tokens, score_threshold=None):
    frames = defaultdict(list)
    token_set = set(gt_tokens)
    unmapped = []
    map_counts = defaultdict(int)
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row_id, row in enumerate(csv.DictReader(stream)):
            score = parse_float(row, "score", 1.0)
            if score_threshold is not None and score < score_threshold:
                continue
            raw_token = str(row.get("sample_token", "")).strip()
            if raw_token.endswith(".0") and raw_token[:-2].isdigit():
                raw_token = raw_token[:-2]
            if raw_token in token_set:
                frame_key = raw_token
                mapping = "sample_token_exact"
            else:
                timestamp_us = timestamp_to_microseconds(row.get("timestamp", ""))
                frame_key, delta = nearest_token(timestamp_us, gt_tokens) if timestamp_us is not None else (None, None)
                if frame_key is None or delta > TIMESTAMP_TOLERANCE_US:
                    unmapped.append({"row_id": row_id, "sample_token": raw_token, "timestamp": row.get("timestamp", ""), "nearest_delta_us": delta})
                    continue
                mapping = "timestamp_nearest_{}us".format(delta)
            map_counts[mapping] += 1
            frames[frame_key].append(
                Box(
                    frame_key=frame_key,
                    row_id=row_id,
                    center_x=parse_float(row, "center_x"),
                    center_y=parse_float(row, "center_y"),
                    size_x=parse_float(row, "size_x"),
                    size_y=parse_float(row, "size_y"),
                    yaw=parse_float(row, "yaw", 0.0),
                    score=score,
                    object_id=str(row.get("obstacle_id", row_id)),
                    source=str(row.get("source", "") or row.get("method", "")),
                    mapping=mapping,
                )
            )
    return dict(frames), unmapped, dict(map_counts)


def rect_corners(box):
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    points = []
    for x, y in ((box.size_x / 2, box.size_y / 2), (box.size_x / 2, -box.size_y / 2), (-box.size_x / 2, -box.size_y / 2), (-box.size_x / 2, box.size_y / 2)):
        points.append((box.center_x + x * c - y * s, box.center_y + x * s + y * c))
    return points


def polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    return abs(sum(x1 * poly[(index + 1) % len(poly)][1] - poly[(index + 1) % len(poly)][0] * y1 for index, (x1, y1) in enumerate(poly))) * 0.5


def inside(point, start, end):
    return (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0]) <= 1e-10


def line_intersection(p1, p2, e1, e2):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = e1
    x4, y4 = e2
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-12:
        return p2
    return (
        ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator,
        ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator,
    )


def polygon_clip(subject, clip):
    output = subject
    for index, edge_start in enumerate(clip):
        edge_end = clip[(index + 1) % len(clip)]
        input_poly, output = output, []
        if not input_poly:
            break
        previous = input_poly[-1]
        for current in input_poly:
            if inside(current, edge_start, edge_end):
                if not inside(previous, edge_start, edge_end):
                    output.append(line_intersection(previous, current, edge_start, edge_end))
                output.append(current)
            elif inside(previous, edge_start, edge_end):
                output.append(line_intersection(previous, current, edge_start, edge_end))
            previous = current
    return output


def bev_iou(left, right):
    poly_left, poly_right = rect_corners(left), rect_corners(right)
    area_left, area_right = polygon_area(poly_left), polygon_area(poly_right)
    intersection = polygon_area(polygon_clip(poly_left, poly_right))
    union = area_left + area_right - intersection
    return intersection / union if union > 0 else 0.0


def metrics(tp, fp, fn, ious, frame_count):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "frames": frame_count,
        "gt_count": tp + fn,
        "prediction_count": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_mean_iou": float(np.mean(ious)) if ious else np.nan,
        "fp_per_frame": fp / frame_count if frame_count else np.nan,
    }


def match_method(method, gt_frames, det_frames, frame_domain, threshold):
    pred_rows, gt_rows, frame_rows = [], [], []
    for frame_order, frame_key in enumerate(frame_domain):
        gt = gt_frames.get(frame_key, [])
        detections = sorted(det_frames.get(frame_key, []), key=lambda box: (-box.score, box.row_id))
        candidates = []
        all_ious = {}
        for det_index, detection in enumerate(detections):
            for gt_index, gt_box in enumerate(gt):
                overlap = bev_iou(detection, gt_box)
                all_ious[(det_index, gt_index)] = overlap
                if overlap >= threshold:
                    candidates.append((detection.score, -detection.row_id, overlap, det_index, gt_index))
        candidates.sort(reverse=True)
        matched_det, matched_gt, pair_iou = set(), set(), {}
        for _, _, overlap, det_index, gt_index in candidates:
            if det_index in matched_det or gt_index in matched_gt:
                continue
            matched_det.add(det_index)
            matched_gt.add(gt_index)
            pair_iou[det_index] = (gt_index, overlap)
        frame_ious = []
        for det_index, detection in enumerate(detections):
            overlaps = [all_ious.get((det_index, index), 0.0) for index in range(len(gt))]
            best_index = int(np.argmax(overlaps)) if overlaps else -1
            best_iou = overlaps[best_index] if best_index >= 0 else 0.0
            if det_index in matched_det:
                matched_index, matched_iou = pair_iou[det_index]
                status, category = "TP", "matched"
                matched_object = gt[matched_index].object_id
                assignment_distance = gt[matched_index].distance
                frame_ious.append(matched_iou)
            else:
                matched_iou, matched_object = np.nan, ""
                assignment_distance = detection.distance
                if best_iou >= threshold:
                    category = "duplicate_prediction"
                elif best_iou > 0:
                    category = "localization_insufficient"
                else:
                    category = "unassociated_false_positive"
                status = "FP"
            pred_rows.append(
                {
                    "method": method,
                    "iou_threshold": threshold,
                    "frame_order": frame_order,
                    "frame_key": frame_key,
                    "prediction_row_id": detection.row_id,
                    "score": detection.score,
                    "source": detection.source,
                    "frame_mapping": detection.mapping,
                    "status": status,
                    "error_category": category,
                    "matched_gt_object_id": matched_object,
                    "matched_iou": matched_iou,
                    "best_gt_object_id": gt[best_index].object_id if best_index >= 0 else "",
                    "best_iou": best_iou,
                    "prediction_distance_m": detection.distance,
                    "assignment_distance_m": assignment_distance,
                }
            )
        for gt_index, gt_box in enumerate(gt):
            gt_rows.append(
                {
                    "method": method,
                    "iou_threshold": threshold,
                    "frame_order": frame_order,
                    "frame_key": frame_key,
                    "gt_row_id": gt_box.row_id,
                    "gt_object_id": gt_box.object_id,
                    "gt_distance_m": gt_box.distance,
                    "status": "matched" if gt_index in matched_gt else "missed",
                }
            )
        tp, fp, fn = len(matched_det), len(detections) - len(matched_det), len(gt) - len(matched_gt)
        frame_metric = metrics(tp, fp, fn, frame_ious, 1)
        frame_rows.append({"method": method, "iou_threshold": threshold, "frame_order": frame_order, "frame_key": frame_key, **frame_metric})
    return pd.DataFrame(pred_rows), pd.DataFrame(gt_rows), pd.DataFrame(frame_rows)


def aggregate_frame_metrics(frame):
    tp, fp, fn = int(frame.tp.sum()), int(frame.fp.sum()), int(frame.fn.sum())
    # Recomputed separately from prediction details by caller when exact matched mean is needed.
    return metrics(tp, fp, fn, [], len(frame))


def distance_bin(value):
    for left, right in zip(DISTANCE_EDGES[:-1], DISTANCE_EDGES[1:]):
        if value >= left and value < right:
            return "{}-{}m".format(f"{left:g}", "inf" if math.isinf(right) else f"{right:g}")
    return "outside"


def distance_summary(method, threshold, pred_detail, gt_detail, frame_count):
    pred = pred_detail[(pred_detail.method == method) & (pred_detail.iou_threshold == threshold)].copy()
    gt = gt_detail[(gt_detail.method == method) & (gt_detail.iou_threshold == threshold)].copy()
    pred["distance_bin"] = pred.assignment_distance_m.map(distance_bin)
    gt["distance_bin"] = gt.gt_distance_m.map(distance_bin)
    rows = []
    labels = [distance_bin(left) for left in DISTANCE_EDGES[:-1]]
    for label in labels:
        selected_pred, selected_gt = pred[pred.distance_bin == label], gt[gt.distance_bin == label]
        tp = int((selected_pred.status == "TP").sum())
        fp = int((selected_pred.status == "FP").sum())
        fn = int((selected_gt.status == "missed").sum())
        ious = selected_pred.loc[selected_pred.status == "TP", "matched_iou"].dropna().tolist()
        rows.append({"method": method, "iou_threshold": threshold, "distance_bin": label, **metrics(tp, fp, fn, ious, frame_count)})
    return rows


def block_bootstrap(frame_metrics, comparisons, block_length=10, repetitions=1000):
    rng = np.random.default_rng(RANDOM_SEED)
    pivot = frame_metrics.pivot_table(index="frame_order", columns="method", values=["tp", "fp", "fn"], aggfunc="first")
    n = len(pivot)
    blocks_needed = int(math.ceil(n / block_length))
    rows = []
    for left, right in comparisons:
        samples = []
        for repetition in range(repetitions):
            starts = rng.integers(0, n - block_length + 1, size=blocks_needed)
            indices = np.concatenate([np.arange(start, start + block_length) for start in starts])[:n]
            values = {}
            for method in (left, right):
                tp = pivot[("tp", method)].to_numpy()[indices].sum()
                fp = pivot[("fp", method)].to_numpy()[indices].sum()
                fn = pivot[("fn", method)].to_numpy()[indices].sum()
                p = tp / (tp + fp) if tp + fp else 0.0
                r = tp / (tp + fn) if tp + fn else 0.0
                values[method] = 2 * p * r / (p + r) if p + r else 0.0
            samples.append(values[left] - values[right])
        rows.append(
            {
                "method_left": left,
                "method_right": right,
                "block_length": block_length,
                "repetitions": repetitions,
                "random_seed": RANDOM_SEED,
                "mean_delta_f1": float(np.mean(samples)),
                "ci95_low": float(np.quantile(samples, 0.025)),
                "ci95_high": float(np.quantile(samples, 0.975)),
            }
        )
    return rows


def file_manifest_row(role, path, frames=None, mapping=None, limitation=""):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        rows = list(reader)
    count = len(rows)
    score_nonempty = sum(bool(str(row.get("score", "")).strip()) for row in rows)
    if role == "ground_truth":
        score_complete = "not_applicable"
    else:
        score_complete = str("score" in fields and score_nonempty == count).lower()
    return {
        "role": role,
        "source_path": str(Path(path).resolve()),
        "relative_path": str(Path(path).resolve().relative_to(ROOT)),
        "sha256": sha256_file(path),
        "size_bytes": Path(path).stat().st_size,
        "records": count,
        "actual_frames": frames if frames is not None else "",
        "score_complete": score_complete,
        "score_nonempty_records": score_nonempty,
        "threshold_or_nms_already_applied": role != "ground_truth",
        "frame_mapping": json.dumps(mapping or {}, ensure_ascii=False, sort_keys=True),
        "known_limitation": limitation,
    }


def plot_metrics(overall):
    plt.rcParams.update({"font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"], "axes.unicode_minus": False})
    subset = overall[overall.method.isin(["RGCF", "BEVFusion Camera+LiDAR exact", "BEVFusion LiDAR", "LiRaFusion raw", "LiRaFusion calibrated"])].copy()
    methods = list(dict.fromkeys(subset.method))
    colors = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]
    x = np.arange(len(IOU_THRESHOLDS))
    width = 0.15
    fig, axis = plt.subplots(figsize=(10.5, 5.2))
    for index, (method, color) in enumerate(zip(methods, colors)):
        values = subset[subset.method == method].set_index("iou_threshold").loc[list(IOU_THRESHOLDS)].f1
        bars = axis.bar(x + (index - 2) * width, values, width, label=method, color=color)
        axis.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    axis.set_xticks(x, [f"IoU={value:.2f}" for value in IOU_THRESHOLDS])
    axis.set_ylabel("F1")
    axis.set_title("Boreas同序列时间留出150帧统一评测")
    axis.set_ylim(0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "S1_150帧多IoU统一对照.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_publication_figures(overall, distance, ablation):
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})
    methods = [
        "RGCF",
        "BEVFusion Camera+LiDAR exact",
        "BEVFusion LiDAR",
        "LiRaFusion calibrated",
    ]
    labels = ["RGCF", "BEVFusion C+L", "BEVFusion L", "LiRaFusion L+R"]
    colors = ["#0072B2", "#009E73", "#E69F00", "#CC79A7"]
    selected = overall[overall.method.isin(methods)].copy()
    metrics_to_plot = [
        ("precision", "Precision", (0.0, 1.0)),
        ("recall", "Recall", (0.0, 1.0)),
        ("f1", "F1 score", (0.0, 1.0)),
        ("fp_per_frame", "False positives per frame", (0.0, 4.0)),
    ]
    x = np.arange(len(IOU_THRESHOLDS))
    width = 0.19
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.0))
    for axis, (column, ylabel, ylim) in zip(axes.flat, metrics_to_plot):
        for index, (method, label, color) in enumerate(zip(methods, labels, colors)):
            values = selected[selected.method == method].set_index("iou_threshold").loc[list(IOU_THRESHOLDS), column]
            bars = axis.bar(x + (index - 1.5) * width, values, width, label=label, color=color)
            axis.bar_label(bars, fmt="%.3f", fontsize=6.5, padding=2, rotation=90)
        axis.set_xticks(x, [f"{value:.2f}" for value in IOU_THRESHOLDS])
        axis.set_xlabel("BEV IoU threshold")
        axis.set_ylabel(ylabel)
        axis.set_ylim(*ylim)
        axis.grid(axis="y", alpha=0.22)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=4, frameon=False)
    fig.suptitle("Boreas temporal holdout (150 frames)", y=1.005)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(FIGURES / "S1_holdout150_multi_metric_en.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    bins = [distance_bin(left) for left in DISTANCE_EDGES[:-1]]
    x = np.arange(len(bins))
    fig, axis = plt.subplots(figsize=(12.5, 5.8))
    distance_selected = distance[(distance.iou_threshold == 0.10) & distance.method.isin(methods)]
    for method_index, (method, label, color) in enumerate(zip(methods, labels, colors)):
        values = distance_selected[distance_selected.method == method].set_index("distance_bin").loc[bins, "f1"]
        bars = axis.bar(x + (method_index - 1.5) * 0.2, values, 0.2, label=label, color=color)
        axis.bar_label(bars, fmt="%.3f", fontsize=6.0, padding=2, rotation=90)
    axis.set_xticks(x, [label.replace("inf", "∞") for label in bins])
    axis.set_xlabel("Radial distance")
    axis.set_ylabel("F1 score at BEV IoU=0.10")
    axis.set_ylim(0, 1.12)
    axis.grid(alpha=0.22)
    handles, legend_labels = axis.get_legend_handles_labels()
    fig.suptitle("Distance-stratified performance on the 150-frame temporal holdout", y=0.995)
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(FIGURES / "S1_holdout150_distance_f1_en.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    ablation_labels = {
        "RGCF minus Base LiDAR geometry+score": "Base LiDAR geometry + score",
        "RGCF minus LiDAR+Camera reliability": "LiDAR + Camera reliability",
        "RGCF minus LiDAR+Radar reliability": "LiDAR + Radar reliability",
        "RGCF minus RGCF no semantic": "Without semantic evidence",
        "RGCF minus RGCF no raw radar": "Without raw Radar evidence",
        "RGCF minus RGCF no VINS": "Without VINS evidence",
    }
    abl = ablation[ablation.iou_threshold == 0.10].copy()
    abl["label"] = abl.comparison.map(ablation_labels)
    abl["delta_f1_pp"] = abl.delta_f1 * 100
    fig, axis = plt.subplots(figsize=(8.7, 4.8))
    bars = axis.barh(abl.label, abl.delta_f1_pp, color="#0072B2")
    axis.bar_label(bars, fmt="%+.3f pp", padding=4, fontsize=8)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("F1 difference: full RGCF minus ablation (percentage points)")
    axis.set_title("Controlled ablation on the 150-frame holdout, BEV IoU=0.10")
    axis.grid(axis="x", alpha=0.22)
    axis.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIGURES / "S1_holdout150_ablation_delta_en.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    paths = {
        "ground_truth": ROOT / "datasets/public/boreas/metadata/boreas_objects_obj_train_public_gt_500.csv",
        "RGCF": ROOT / "results/my_method_boreas_train500_navtech_robust_raw_v3/learned_trimodal_semantic_vins_reliability_fixedmodel_v5/boreas_train500_learned_trimodal_reliability_gate.csv",
        "BEVFusion Camera+LiDAR exact": ROOT / "results/supplementary_experiments_20260812/inputs/bevfusion_boreas_train500_camera_lidar_detections.csv",
        "BEVFusion Camera+LiDAR archive": ROOT / "handoff/received_stage28_complete_work_transfer_20260731_124856/results/stage27_gpu_degraded_candidates/clean/bevfusion_camera_lidar.csv",
        "BEVFusion LiDAR": ROOT / "handoff/received_stage28_complete_work_transfer_20260731_124856/results/stage27_gpu_degraded_candidates/clean/bevfusion_lidar.csv",
        "LiRaFusion": ROOT / "handoff/received_stage28_complete_work_transfer_20260731_124856/results/stage27_gpu_degraded_candidates/clean/lirafusion_lidar_radar.csv",
        "Base LiDAR geometry+score": ROOT / "results/my_method_boreas_train500_iou0p10/class_adaptive_confidence_gate_v1/boreas_train500_class_adaptive_detections.csv",
        "LiDAR+Camera reliability": ROOT / "results/my_method_boreas_train500_navtech_robust_raw_v3/ablation_lidar_camera/boreas_train500_learned_trimodal_reliability_gate.csv",
        "LiDAR+Radar reliability": ROOT / "results/my_method_boreas_train500_navtech_robust_raw_v3/ablation_lidar_radar/boreas_train500_learned_trimodal_reliability_gate.csv",
        "RGCF no semantic": ROOT / "results/my_method_boreas_train500_navtech_robust_raw_v3/ablation_no_semantic_mask/boreas_train500_learned_trimodal_reliability_gate.csv",
        "RGCF no raw radar": ROOT / "results/my_method_boreas_train500_navtech_robust_raw_v3/ablation_no_raw_radar/boreas_train500_learned_trimodal_reliability_gate.csv",
        "RGCF no VINS": ROOT / "results/my_method_boreas_train500_navtech_robust_raw_v3/ablation_no_vins/boreas_train500_learned_trimodal_reliability_gate.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))

    gt_frames, all_tokens = load_gt(paths["ground_truth"])
    if len(all_tokens) != 500:
        raise AssertionError(f"expected 500 GT frames, got {len(all_tokens)}")
    split_names = ["model_development"] * 250 + ["calibration"] * 100 + ["holdout_test"] * 150
    split = pd.DataFrame(
        {
            "sequence_id": "boreas-objects-v1_train_partition__capture_sequence_unknown",
            "frame_key": all_tokens,
            "timestamp_raw": all_tokens,
            "split": split_names,
            "frame_order": np.arange(500),
            "gt_count": [len(gt_frames[token]) for token in all_tokens],
        }
    )
    split.to_csv(TASK / "split_manifest.csv", index=False)
    holdout = all_tokens[350:500]

    method_specs = {
        "RGCF": (paths["RGCF"], None),
        "BEVFusion Camera+LiDAR exact": (paths["BEVFusion Camera+LiDAR exact"], None),
        "BEVFusion Camera+LiDAR archive": (paths["BEVFusion Camera+LiDAR archive"], None),
        "BEVFusion LiDAR": (paths["BEVFusion LiDAR"], None),
        "LiRaFusion raw": (paths["LiRaFusion"], 0.10),
        "LiRaFusion calibrated": (paths["LiRaFusion"], 0.359346956),
        "Base LiDAR geometry+score": (paths["Base LiDAR geometry+score"], None),
        "LiDAR+Camera reliability": (paths["LiDAR+Camera reliability"], None),
        "LiDAR+Radar reliability": (paths["LiDAR+Radar reliability"], None),
        "RGCF no semantic": (paths["RGCF no semantic"], None),
        "RGCF no raw radar": (paths["RGCF no raw radar"], None),
        "RGCF no VINS": (paths["RGCF no VINS"], None),
    }
    loaded, map_metadata, unmapped_rows = {}, {}, []
    for method, (path, score_threshold) in method_specs.items():
        frames, unmapped, mapping = load_detections(path, all_tokens, score_threshold)
        loaded[method] = frames
        map_metadata[method] = mapping
        for row in unmapped:
            unmapped_rows.append({"method": method, **row})
    pd.DataFrame(
        unmapped_rows,
        columns=["method", "row_id", "sample_token", "timestamp", "nearest_delta_us"],
    ).to_csv(RESULTS / "unmapped_predictions.csv", index=False)

    manifest = [file_manifest_row("ground_truth", paths["ground_truth"], len(gt_frames), limitation="GT frame list reconstructed from label source filename; capture sequence ID is absent")]
    for method, (path, threshold) in method_specs.items():
        limitation = "Candidate file is already score-thresholded/post-NMS; PR/AP would be candidate-limited"
        if method == "LiRaFusion calibrated":
            limitation += "; threshold 0.359346956 selected on calibration100 only"
        manifest.append(file_manifest_row(method, path, len(loaded[method]), map_metadata[method], limitation))
    pd.DataFrame(manifest).to_csv(TASK / "input_manifest.csv", index=False)

    all_pred, all_gt, all_frame, overall_rows, distance_rows, error_rows = [], [], [], [], [], []
    for method, frames in loaded.items():
        for threshold in IOU_THRESHOLDS:
            pred_detail, gt_detail, frame_detail = match_method(method, gt_frames, frames, holdout, threshold)
            all_pred.append(pred_detail)
            all_gt.append(gt_detail)
            all_frame.append(frame_detail)
            tp, fp, fn = int(frame_detail.tp.sum()), int(frame_detail.fp.sum()), int(frame_detail.fn.sum())
            ious = pred_detail.loc[pred_detail.status == "TP", "matched_iou"].dropna().tolist()
            overall_rows.append({"method": method, "split": "holdout150", "iou_threshold": threshold, **metrics(tp, fp, fn, ious, len(holdout))})
            distance_rows.extend(distance_summary(method, threshold, pred_detail, gt_detail, len(holdout)))
            counts = pred_detail.error_category.value_counts()
            for category in ("matched", "duplicate_prediction", "localization_insufficient", "unassociated_false_positive"):
                error_rows.append({"method": method, "iou_threshold": threshold, "record_type": "prediction", "category": category, "count": int(counts.get(category, 0))})
            error_rows.append({"method": method, "iou_threshold": threshold, "record_type": "ground_truth", "category": "missed", "count": int((gt_detail.status == "missed").sum())})
    pred_detail = pd.concat(all_pred, ignore_index=True)
    gt_detail = pd.concat(all_gt, ignore_index=True)
    frame_detail = pd.concat(all_frame, ignore_index=True)
    overall = pd.DataFrame(overall_rows)
    pred_detail.to_csv(RESULTS / "per_prediction_detail.csv", index=False)
    gt_detail.to_csv(RESULTS / "per_gt_detail.csv", index=False)
    frame_detail.to_csv(RESULTS / "per_frame_metrics.csv", index=False)
    overall.to_csv(RESULTS / "holdout150_multi_iou_metrics.csv", index=False)
    distance = pd.DataFrame(distance_rows)
    distance.to_csv(RESULTS / "holdout150_distance_metrics.csv", index=False)
    pd.DataFrame(error_rows).to_csv(RESULTS / "holdout150_error_decomposition.csv", index=False)

    full = overall[overall.method == "RGCF"].set_index("iou_threshold")
    delta_rows = []
    ablations = ["Base LiDAR geometry+score", "LiDAR+Camera reliability", "LiDAR+Radar reliability", "RGCF no semantic", "RGCF no raw radar", "RGCF no VINS"]
    for method in ablations:
        other = overall[overall.method == method].set_index("iou_threshold")
        for threshold in IOU_THRESHOLDS:
            delta_rows.append(
                {
                    "comparison": f"RGCF minus {method}",
                    "iou_threshold": threshold,
                    **{f"delta_{name}": float(full.loc[threshold, name] - other.loc[threshold, name]) for name in ("tp", "fp", "fn", "precision", "recall", "f1")},
                }
            )
    ablation = pd.DataFrame(delta_rows)
    ablation.to_csv(RESULTS / "controlled_ablation_deltas.csv", index=False)

    bootstrap_rows = []
    for threshold in (0.10, 0.50):
        selected = frame_detail[frame_detail.iou_threshold == threshold]
        comparisons = [("RGCF", "BEVFusion Camera+LiDAR exact"), ("RGCF", "BEVFusion LiDAR"), ("RGCF", "RGCF no VINS")]
        rows = block_bootstrap(selected, comparisons)
        for row in rows:
            row["iou_threshold"] = threshold
        bootstrap_rows.extend(rows)
    pd.DataFrame(bootstrap_rows).to_csv(RESULTS / "paired_block_bootstrap_f1.csv", index=False)

    exact = overall[overall.method == "BEVFusion Camera+LiDAR exact"]
    archive = overall[overall.method == "BEVFusion Camera+LiDAR archive"]
    discrepancy = exact.merge(archive, on=["split", "iou_threshold"], suffixes=("_exact", "_archive"))
    discrepancy.to_csv(RESULTS / "bevfusion_cl_version_comparison.csv", index=False)
    plot_metrics(overall)
    plot_publication_figures(overall, distance, ablation)

    protocol = {
        "protocol_version": "JEI-S1-20260902-v1",
        "task": "class-agnostic obstacle detection in BEV",
        "frame_domain": "fixed last 150 of 500 GT label frames ordered by exact integer-microsecond label filename",
        "sequence_boundary": "same Boreas Objects train partition; capture sequence ID unavailable; not cross-sequence",
        "frame_identifier": "integer microsecond token; exact sample_token preferred, otherwise nearest timestamp within 1 us",
        "coordinates": {"frame": "boreas_lidar", "units": "metres/radians", "box": "center_x, center_y, size_x, size_y, yaw"},
        "iou": {"type": "rotated BEV rectangle IoU", "thresholds": list(IOU_THRESHOLDS)},
        "matching": "global score-descending one-to-one greedy; stable original row ID tie break",
        "empty_frames": "all 150 predefined frames retained; no-prediction frames contribute FN; zero-GT frames retain FP",
        "fp_per_frame_denominator": 150,
        "mean_iou": "mean over matched TP only; undefined (NaN) if TP=0",
        "distance_bins_m": ["0-3", "3-6", "6-10", "10-15", "15-20", "20-30", "30-50", "50-inf"],
        "distance_assignment": "matching performed globally first; TP/FN assigned by GT radial distance, FP by prediction radial distance",
        "workpoint": "same frozen confidence/NMS candidate set across IoU thresholds; no test tuning",
        "liRaFusion_calibration": {"raw_threshold": 0.10, "calibrated_threshold": 0.359346956, "selected_on": "frames 250-349 calibration100"},
        "random_seed": RANDOM_SEED,
    }
    atomic_json(TASK / "evaluation_protocol.json", protocol)
    protocol_hash = sha256_file(TASK / "evaluation_protocol.json")
    frozen = {
        "frozen_at": "2026-09-02T17:57:17+08:00",
        "protocol_sha256": protocol_hash,
        "random_seed": RANDOM_SEED,
        "methods": {
            method: {"prediction_sha256": sha256_file(path), "score_threshold": threshold, "candidate_generation": "archived frozen CSV; full raw-sensor inference availability must be checked before S2"}
            for method, (path, threshold) in method_specs.items()
        },
        "RGCF_model": {
            "path": "results/my_method_boreas_train500_navtech_robust_raw_v3/learned_trimodal_semantic_vins_reliability_fixedmodel_v5/trimodal_reliability_model.json",
            "sha256": sha256_file(ROOT / "results/my_method_boreas_train500_navtech_robust_raw_v3/learned_trimodal_semantic_vins_reliability_fixedmodel_v5/trimodal_reliability_model.json"),
            "fixed_model_inference_script": "scripts/apply_boreas_trimodal_reliability_model.py",
            "base_candidate_path": "results/my_method_boreas_train500_iou0p10/class_adaptive_confidence_gate_v1/boreas_train500_class_adaptive_detections.csv",
            "base_candidate_sha256": sha256_file(ROOT / "results/my_method_boreas_train500_iou0p10/class_adaptive_confidence_gate_v1/boreas_train500_class_adaptive_detections.csv"),
            "feature_path": "results/my_method_boreas_train500_navtech_robust_raw_v3/features/boreas_train500_trimodal_candidates_with_navtech_semantic_vins_support.csv",
            "feature_sha256": sha256_file(ROOT / "results/my_method_boreas_train500_navtech_robust_raw_v3/features/boreas_train500_trimodal_candidates_with_navtech_semantic_vins_support.csv"),
            "semantic_model_path": "yolov8n-seg.pt",
            "semantic_model_sha256": sha256_file(ROOT / "yolov8n-seg.pt"),
            "pose_source": "Boreas applanix/lidar_poses.csv reference poses; proxy evidence, not an actual VINS run",
            "feature_file_is_dataset_specific": True,
            "raw_sensor_end_to_end_ready_for_S2": False,
            "blocking_reasons": [
                "fixed inference begins from base and auxiliary candidate CSV files rather than raw sensors",
                "public baseline raw-sensor inference environments are not frozen on this CPU host",
                "no eligible unseen sequence with complete identity, labels and all required sensors is local",
            ],
        },
        "code_sha256": {
            "jei_s1_audit.py": sha256_file(Path(__file__)),
            "test_jei_evaluator.py": sha256_file(TASK / "tests/test_jei_evaluator.py"),
            "verify_outputs.py": sha256_file(TASK / "tests/verify_outputs.py"),
            "cross_validate_original_evaluator.py": sha256_file(TASK / "tests/cross_validate_original_evaluator.py"),
            "finalize_sha256_manifest.py": sha256_file(TASK / "scripts/finalize_sha256_manifest.py"),
            "evaluate_boreas_detection_csv.py": sha256_file(ROOT / "scripts/evaluate_boreas_detection_csv.py"),
            "apply_boreas_trimodal_reliability_model.py": sha256_file(ROOT / "scripts/apply_boreas_trimodal_reliability_model.py"),
            "run_boreas_trimodal_fusion_pipeline.py": sha256_file(ROOT / "scripts/run_boreas_trimodal_fusion_pipeline.py"),
            "add_boreas_raw_sensor_support_features.py": sha256_file(ROOT / "scripts/add_boreas_raw_sensor_support_features.py"),
            "add_boreas_semantic_mask_support_features.py": sha256_file(ROOT / "scripts/add_boreas_semantic_mask_support_features.py"),
            "add_boreas_vins_temporal_support_features.py": sha256_file(ROOT / "scripts/add_boreas_vins_temporal_support_features.py"),
        },
    }
    atomic_json(TASK / "frozen_run_manifest.json", frozen)

    output_files = sorted(path for path in TASK.rglob("*") if path.is_file())
    pd.DataFrame([{"relative_path": str(path.relative_to(TASK)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in output_files]).to_csv(TASK / "input_output_sha256.csv", index=False)

    audit = {
        "environment": {"platform": platform.platform(), "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
        "holdout_frames": len(holdout),
        "holdout_gt": sum(len(gt_frames[token]) for token in holdout),
        "unmapped_prediction_rows": len(unmapped_rows),
        "protocol_sha256": protocol_hash,
        "count_invariants": "passed" if all((row.tp + row.fn == row.gt_count and row.tp + row.fp == row.prediction_count) for row in overall.itertuples()) else "failed",
    }
    atomic_json(RESULTS / "audit_summary.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
