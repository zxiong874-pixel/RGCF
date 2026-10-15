"""Public-dataset ground-truth adapters for BEV obstacle evaluation."""

import csv
import math
from pathlib import Path

from bev_obstacle_detection.detection_metrics import Box2D


PUBLIC_GT_COLUMNS = [
    "timestamp",
    "frame_id",
    "label",
    "object_id",
    "center_x",
    "center_y",
    "center_z",
    "size_x",
    "size_y",
    "size_z",
    "yaw",
    "score",
    "source_path",
]


def normalize_public_timestamp(value):
    """Normalize seconds, microseconds, or nanoseconds into ROS-style seconds."""
    timestamp = float(value)
    magnitude = abs(timestamp)
    if magnitude > 1.0e17:
        return timestamp / 1.0e9
    if magnitude > 1.0e12:
        return timestamp / 1.0e6
    return timestamp


def yaw_aligned_bev_size(size_x, size_y, yaw):
    cos_yaw = abs(math.cos(yaw))
    sin_yaw = abs(math.sin(yaw))
    return cos_yaw * size_x + sin_yaw * size_y, sin_yaw * size_x + cos_yaw * size_y


def _row_to_box(row):
    score_text = row.get("score", "")
    score = float(score_text) if str(score_text).strip() else 1.0
    yaw_text = row.get("yaw", "")
    yaw = float(yaw_text) if str(yaw_text).strip() else 0.0
    size_x, size_y = yaw_aligned_bev_size(float(row["size_x"]), float(row["size_y"]), yaw)
    return Box2D(
        timestamp=normalize_public_timestamp(row["timestamp"]),
        center_x=float(row["center_x"]),
        center_y=float(row["center_y"]),
        size_x=size_x,
        size_y=size_y,
        label=str(row.get("label", "")),
        score=score,
    )


def read_public_gt_csv(path, start_time=None, end_time=None, allowed_labels=None):
    frames = {}
    labels = set(allowed_labels or [])
    with Path(path).expanduser().open("r", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if labels and row.get("label", "") not in labels:
                continue
            try:
                box = _row_to_box(row)
            except (KeyError, ValueError):
                continue
            if start_time is not None and box.timestamp < start_time:
                continue
            if end_time is not None and box.timestamp > end_time:
                continue
            frames.setdefault(box.timestamp, []).append(box)
    return frames


def boreas_label_row_from_line(line, timestamp, source_path):
    parts = line.split()
    if len(parts) < 10:
        return None
    score = "" if len(parts) < 11 or parts[10] == "None" else parts[10]
    return {
        "timestamp": timestamp,
        "frame_id": "boreas_lidar",
        "label": parts[1],
        "object_id": parts[0],
        "size_x": parts[2],
        "size_y": parts[3],
        "size_z": parts[4],
        "center_x": parts[5],
        "center_y": parts[6],
        "center_z": parts[7],
        "yaw": parts[8],
        "score": score,
        "source_path": str(source_path),
    }


def read_boreas_label_file(path):
    label_path = Path(path).expanduser()
    timestamp = normalize_public_timestamp(label_path.stem)
    rows = []
    with label_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = boreas_label_row_from_line(line, timestamp, label_path)
            if row is not None:
                rows.append(row)
    return rows


def iter_boreas_label_files(path):
    root = Path(path).expanduser()
    if root.is_file():
        yield root
        return
    for candidate in sorted(root.rglob("*.txt")):
        if candidate.parent.name in ("labels", "labels_detection") or "label" in candidate.parent.name.lower():
            yield candidate


def convert_boreas_labels_to_public_csv(input_path, output_path):
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    files = 0
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PUBLIC_GT_COLUMNS)
        writer.writeheader()
        for label_file in iter_boreas_label_files(input_path):
            files += 1
            for row in read_boreas_label_file(label_file):
                writer.writerow({column: row.get(column, "") for column in PUBLIC_GT_COLUMNS})
                count += 1
    return {"output_csv": str(output), "label_files": files, "boxes": count}


def load_public_gt_frames(path, fmt="auto", start_time=None, end_time=None, allowed_labels=None):
    source = Path(path).expanduser()
    inferred_format = fmt
    if inferred_format == "auto":
        inferred_format = "csv" if source.suffix.lower() == ".csv" else "boreas"
    if inferred_format == "csv":
        return read_public_gt_csv(source, start_time=start_time, end_time=end_time, allowed_labels=allowed_labels)
    if inferred_format == "boreas":
        rows = []
        for label_file in iter_boreas_label_files(source):
            rows.extend(read_boreas_label_file(label_file))
        frames = {}
        labels = set(allowed_labels or [])
        for row in rows:
            if labels and row.get("label", "") not in labels:
                continue
            try:
                box = _row_to_box(row)
            except (KeyError, ValueError):
                continue
            if start_time is not None and box.timestamp < start_time:
                continue
            if end_time is not None and box.timestamp > end_time:
                continue
            frames.setdefault(box.timestamp, []).append(box)
        return frames
    raise ValueError("unsupported public GT format: {}".format(fmt))
