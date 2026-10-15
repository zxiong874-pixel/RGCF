#!/usr/bin/env python3
"""Unified candidate-level adaptive gate for Boreas LiDAR+Camera+Radar fusion."""

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import evaluate_boreas_detection_csv as eval_csv


def as_float(row, key, default=0.0):
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except ValueError:
        return default


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def row_timestamp(row):
    return f"{float(row['timestamp']):.6f}"


def row_box(row, score_key="score"):
    return {
        "center_x": as_float(row, "center_x"),
        "center_y": as_float(row, "center_y"),
        "size_x": as_float(row, "size_x"),
        "size_y": as_float(row, "size_y"),
        "yaw": as_float(row, "yaw"),
        "score": as_float(row, score_key, as_float(row, "score")),
    }


def rows_to_frames(rows):
    frames = defaultdict(list)
    for row in rows:
        try:
            frames[row_timestamp(row)].append(row_box(row))
        except (KeyError, ValueError):
            continue
    return frames


def temporal_support(row, same_source=False):
    vins_key = (
        "vins_temporal_support_same_source"
        if same_source
        else "vins_temporal_support_all"
    )
    if vins_key in row:
        if as_float(row, "vins_pose_available") <= 0.0:
            return 0.0
        return as_float(row, vins_key)
    legacy_key = "temporal_support_same_source" if same_source else "temporal_support_all"
    return as_float(row, legacy_key)


def split_timestamps(gt_frames, calibration_start, calibration_end):
    timestamps = sorted(gt_frames)
    return {
        "model_train": set(timestamps[:calibration_start]),
        "calibration": set(timestamps[calibration_start:calibration_end]),
        "holdout": set(timestamps[calibration_end:]),
        "all": set(timestamps),
    }


def subset_frames(frames, timestamps):
    return defaultdict(list, {timestamp: boxes for timestamp, boxes in frames.items() if timestamp in timestamps})


def max_overlap(box, accepted_boxes):
    overlap = 0.0
    for accepted in accepted_boxes:
        center = math.hypot(box["center_x"] - accepted["center_x"], box["center_y"] - accepted["center_y"])
        if center <= max(box["size_x"], box["size_y"], accepted["size_x"], accepted["size_y"], 4.0):
            overlap = max(overlap, eval_csv.bev_iou(box, accepted))
    return overlap


def adaptive_score(row):
    source = row["source"]
    score = as_float(row, "score")
    camera_support = as_float(row, "camera_semantic_support")
    radar_support = as_float(row, "radar_support")
    temporal = math.log1p(max(temporal_support(row), 0.0)) / 4.0
    if source == "camera_lidar":
        return 0.75 * score + 0.15 * radar_support + 0.10 * temporal
    if source == "radar_lirafusion":
        return 0.80 * score + 0.10 * camera_support + 0.10 * temporal
    return score


def candidate_passes(row, args):
    source = row["source"]
    score = as_float(row, "score")
    if source == "camera_lidar":
        if "camera_sync_available" in row and as_float(row, "camera_sync_available") <= 0.0:
            return False
        return (
            score >= args.camera_score_threshold
            and as_float(row, "camera_semantic_support") >= args.min_camera_support
            and as_float(row, "radar_support") >= args.min_camera_cross_radar_support
        )
    if source == "radar_lirafusion":
        if "radar_sync_available" in row and as_float(row, "radar_sync_available") <= 0.0:
            return False
        return (
            score >= args.radar_score_threshold
            and as_float(row, "radar_support") >= args.min_radar_support
            and as_float(row, "camera_semantic_support") >= args.min_radar_cross_camera_support
        )
    return False


def select_candidates(feature_rows, base_frames, args):
    priority = {"camera_lidar": 0, "radar_lirafusion": 1}
    max_per_frame = {
        "camera_lidar": args.max_camera_per_frame,
        "radar_lirafusion": args.max_radar_per_frame,
    }
    selected = []
    accepted_frames = defaultdict(list, {timestamp: list(boxes) for timestamp, boxes in base_frames.items()})
    source_counts_by_frame = defaultdict(lambda: Counter())
    rows = []
    for row in feature_rows:
        if not candidate_passes(row, args):
            continue
        row = dict(row)
        row["adaptive_gate_score"] = f"{adaptive_score(row):.8f}"
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row_timestamp(row),
            priority.get(row["source"], 99),
            -as_float(row, "adaptive_gate_score"),
        )
    )
    for row in rows:
        timestamp = row_timestamp(row)
        source = row["source"]
        if source_counts_by_frame[timestamp][source] >= max_per_frame.get(source, 0):
            continue
        box = row_box(row, "adaptive_gate_score")
        if max_overlap(box, accepted_frames.get(timestamp, [])) > args.max_base_iou:
            continue
        if any(eval_csv.bev_iou(box, kept) >= args.nms_iou for kept in accepted_frames.get(timestamp, [])):
            continue
        selected.append(row)
        accepted_frames[timestamp].append(box)
        source_counts_by_frame[timestamp][source] += 1
    return selected


def write_fused_csv(path, base_fieldnames, base_rows, selected, method_name):
    fieldnames = list(base_fieldnames)
    for field in [
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
        "yaw",
        "source",
        "adaptive_gate_score",
    ]:
        if field not in fieldnames:
            fieldnames.append(field)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in base_rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
        for idx, row in enumerate(selected):
            out = {field: "" for field in fieldnames}
            out.update(
                {
                    "timestamp": row.get("raw_timestamp") or row["timestamp"],
                    "method": method_name,
                    "frame_id": row.get("frame_id", "boreas_lidar"),
                    "obstacle_id": f"{row['source']}_adaptive_{idx}",
                    "center_x": row["center_x"],
                    "center_y": row["center_y"],
                    "size_x": row["size_x"],
                    "size_y": row["size_y"],
                    "cell_count": row.get("cell_count", ""),
                    "score": row["adaptive_gate_score"],
                    "yaw": row.get("yaw", "0.0"),
                    "source": row["source"],
                    "adaptive_gate_score": row["adaptive_gate_score"],
                }
            )
            writer.writerow(out)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-csv", required=True)
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--feature-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--method-name", default="MyMethod_trimodal_adaptive_gate_local")
    parser.add_argument("--camera-score-threshold", type=float, default=0.25)
    parser.add_argument("--radar-score-threshold", type=float, default=0.60)
    parser.add_argument("--min-camera-support", type=float, default=0.0)
    parser.add_argument("--min-radar-support", type=float, default=0.0)
    parser.add_argument("--min-camera-cross-radar-support", type=float, default=0.0)
    parser.add_argument("--min-radar-cross-camera-support", type=float, default=0.0)
    parser.add_argument("--max-base-iou", type=float, default=0.03)
    parser.add_argument("--nms-iou", type=float, default=0.10)
    parser.add_argument("--max-camera-per-frame", type=int, default=1)
    parser.add_argument("--max-radar-per-frame", type=int, default=4)
    parser.add_argument("--calibration-start-frame", type=int, default=250)
    parser.add_argument("--calibration-end-frame", type=int, default=350)
    parser.add_argument("--evaluation-ious", default="0.10,0.25,0.50")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_frames = eval_csv.read_boxes(args.gt_csv, is_gt=True)
    base_fieldnames, base_rows = read_rows(args.base_csv)
    base_frames = rows_to_frames(base_rows)
    _, feature_rows = read_rows(args.feature_csv)
    selected = select_candidates(feature_rows, base_frames, args)
    out_csv = out_dir / "boreas_train500_trimodal_adaptive_gate.csv"
    write_fused_csv(out_csv, base_fieldnames, base_rows, selected, args.method_name)

    fused_frames = eval_csv.read_boxes(out_csv, is_gt=False)
    split_sets = split_timestamps(gt_frames, args.calibration_start_frame, args.calibration_end_frame)
    evaluation_ious = [float(item) for item in args.evaluation_ious.split(",") if item.strip()]
    summary_rows = []
    for split, timestamps in split_sets.items():
        split_selected = [row for row in selected if row_timestamp(row) in timestamps]
        source_counts = Counter(row["source"] for row in split_selected)
        for threshold in evaluation_ious:
            metrics = eval_csv.evaluate(
                subset_frames(gt_frames, timestamps),
                subset_frames(fused_frames, timestamps),
                threshold,
            )
            summary_rows.append(
                {
                    "method": args.method_name,
                    "split": split,
                    "iou_threshold": f"{threshold:.2f}",
                    "selected_supplements": len(split_selected),
                    "camera_supplements": source_counts.get("camera_lidar", 0),
                    "radar_supplements": source_counts.get("radar_lirafusion", 0),
                    **{
                        key: f"{value:.4f}" if isinstance(value, float) else value
                        for key, value in metrics.items()
                    },
                    "detection_csv_path": str(out_csv),
                }
            )
    selected_summary = []
    for source, count in Counter(row["source"] for row in selected).items():
        selected_summary.append({"source": source, "selected": count})
    write_csv(out_dir / "trimodal_adaptive_gate_multi_iou_summary.csv", summary_rows)
    write_csv(out_dir / "trimodal_adaptive_gate_selected_sources.csv", selected_summary)
    config_lines = [
        f"gt_csv={args.gt_csv}",
        f"base_csv={args.base_csv}",
        f"feature_csv={args.feature_csv}",
        f"output={out_csv}",
        f"method_name={args.method_name}",
        f"camera_score_threshold={args.camera_score_threshold}",
        f"radar_score_threshold={args.radar_score_threshold}",
        f"min_camera_support={args.min_camera_support}",
        f"min_radar_support={args.min_radar_support}",
        f"min_camera_cross_radar_support={args.min_camera_cross_radar_support}",
        f"min_radar_cross_camera_support={args.min_radar_cross_camera_support}",
        f"max_base_iou={args.max_base_iou}",
        f"max_camera_per_frame={args.max_camera_per_frame}",
        f"max_radar_per_frame={args.max_radar_per_frame}",
        f"selected_supplements={len(selected)}",
    ]
    (out_dir / "trimodal_adaptive_gate_config.txt").write_text("\n".join(config_lines) + "\n", encoding="utf-8")
    print(out_csv)
    print(out_dir / "trimodal_adaptive_gate_multi_iou_summary.csv")
    print(f"selected_supplements={len(selected)}")


if __name__ == "__main__":
    main()
