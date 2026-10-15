#!/usr/bin/env python3
"""Evaluate unified Boreas detection CSV files with BEV IoU matching."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def read_boxes(path, is_gt=False):
    frames = defaultdict(list)
    with Path(path).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                timestamp = f"{float(row['timestamp']):.6f}"
                score = 1.0 if is_gt or row.get("score", "") == "" else float(row["score"])
                frames[timestamp].append(
                    {
                        "center_x": float(row["center_x"]),
                        "center_y": float(row["center_y"]),
                        "size_x": float(row["size_x"]),
                        "size_y": float(row["size_y"]),
                        "yaw": float(row.get("yaw", 0.0) or 0.0),
                        "score": score,
                    }
                )
            except (KeyError, ValueError):
                continue
    return frames


def rect_corners(box):
    cx, cy = box["center_x"], box["center_y"]
    dx, dy = box["size_x"], box["size_y"]
    yaw = box["yaw"]
    c, s = math.cos(yaw), math.sin(yaw)
    corners = []
    for x, y in [(dx / 2, dy / 2), (dx / 2, -dy / 2), (-dx / 2, -dy / 2), (-dx / 2, dy / 2)]:
        corners.append((cx + x * c - y * s, cy + x * s + y * c))
    return corners


def polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for i, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(i + 1) % len(poly)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def inside(point, edge_start, edge_end):
    x, y = point
    x1, y1 = edge_start
    x2, y2 = edge_end
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) <= 1e-9


def intersection(p1, p2, e1, e2):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = e1
    x4, y4 = e2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return p2
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return px, py


def polygon_clip(subject, clip):
    output = subject
    for i, edge_start in enumerate(clip):
        edge_end = clip[(i + 1) % len(clip)]
        input_poly = output
        output = []
        if not input_poly:
            break
        prev = input_poly[-1]
        for cur in input_poly:
            if inside(cur, edge_start, edge_end):
                if not inside(prev, edge_start, edge_end):
                    output.append(intersection(prev, cur, edge_start, edge_end))
                output.append(cur)
            elif inside(prev, edge_start, edge_end):
                output.append(intersection(prev, cur, edge_start, edge_end))
            prev = cur
    return output


def bev_iou(a, b):
    poly_a = rect_corners(a)
    poly_b = rect_corners(b)
    area_a = polygon_area(poly_a)
    area_b = polygon_area(poly_b)
    inter = polygon_area(polygon_clip(poly_a, poly_b))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate(gt_frames, det_frames, iou_threshold):
    tp = fp = fn = 0
    ious = []
    frame_ids = sorted(set(gt_frames) | set(det_frames))
    for frame_id in frame_ids:
        gt = gt_frames.get(frame_id, [])
        det = sorted(det_frames.get(frame_id, []), key=lambda x: x["score"], reverse=True)
        matched = set()
        for det_box in det:
            best_idx = -1
            best_iou = 0.0
            for idx, gt_box in enumerate(gt):
                if idx in matched:
                    continue
                iou = bev_iou(det_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0 and best_iou >= iou_threshold:
                tp += 1
                matched.add(best_idx)
                ious.append(best_iou)
            else:
                fp += 1
        fn += len(gt) - len(matched)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "gt_frames": len(gt_frames),
        "detection_frames": len(det_frames),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "miss_rate": fn / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (tp + fp) if tp + fp else 0.0,
    }


def parse_distance_bins(text):
    edges = []
    for item in str(text).split(","):
        value = item.strip().lower()
        if not value:
            continue
        edges.append(float("inf") if value in ("inf", "infinity") else float(value))
    if len(edges) < 2:
        raise ValueError("distance bins require at least two edges")
    if any(right <= left for left, right in zip(edges, edges[1:])):
        raise ValueError("distance bin edges must be strictly increasing")
    return list(zip(edges, edges[1:]))


def filter_frames_by_distance(frames, min_distance, max_distance):
    filtered = defaultdict(list)
    for timestamp, boxes in frames.items():
        selected = [
            box
            for box in boxes
            if min_distance
            <= math.hypot(box["center_x"], box["center_y"])
            < max_distance
        ]
        if selected:
            filtered[timestamp] = selected
    return filtered


def distance_label(min_distance, max_distance):
    right = "inf" if math.isinf(max_distance) else f"{max_distance:g}"
    return f"{min_distance:g}-{right}m"


def evaluate_by_distance(gt_frames, det_frames, iou_threshold, distance_bins):
    rows = []
    for min_distance, max_distance in distance_bins:
        metrics = evaluate(
            filter_frames_by_distance(gt_frames, min_distance, max_distance),
            filter_frames_by_distance(det_frames, min_distance, max_distance),
            iou_threshold,
        )
        rows.append(
            {
                "distance_bin": distance_label(min_distance, max_distance),
                "distance_min_m": min_distance,
                "distance_max_m": max_distance,
                **metrics,
            }
        )
    return rows


def write_distance_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "distance_bin",
        "distance_min_m",
        "distance_max_m",
        "gt_frames",
        "detection_frames",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "mean_iou",
        "miss_rate",
        "false_positive_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.4f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def write_summary(path, metrics, detection_csv, status):
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()}
    row["detection_csv_path"] = str(detection_csv)
    row["status"] = status
    fields = [
        "gt_frames",
        "detection_frames",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "mean_iou",
        "miss_rate",
        "false_positive_rate",
        "detection_csv_path",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-csv", default="boreas_objects_obj_train_public_gt_500.csv")
    parser.add_argument("--det-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.10)
    parser.add_argument(
        "--distance-bins",
        default="0,3,6,10,15,20,30,50,inf",
        help="Comma-separated radial distance edges.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    gt = read_boxes(args.gt_csv, is_gt=True)
    det = read_boxes(args.det_csv, is_gt=False)
    metrics = evaluate(gt, det, args.iou_threshold)
    out_dir = Path(args.out_dir)
    write_summary(out_dir / "public_detection_eval_summary.csv", metrics, args.det_csv, "ok")
    distance_path = out_dir / "public_detection_eval_by_distance.csv"
    write_distance_summary(
        distance_path,
        evaluate_by_distance(
            gt,
            det,
            args.iou_threshold,
            parse_distance_bins(args.distance_bins),
        ),
    )
    print(out_dir / "public_detection_eval_summary.csv")
    print(distance_path)
    print(metrics)


if __name__ == "__main__":
    main()
