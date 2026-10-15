"""Detection-vs-Gazebo-GT metrics for BEV obstacle smoke experiments."""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from bev_obstacle_detection.smoke_evaluation import bag_path_for_run, detection_csv_for_method, format_number


DEFAULT_MODEL_STATES_TOPIC = "/gazebo/model_states"

GT_COLUMNS = [
    "method",
    "timestamp",
    "model_name",
    "frame_id",
    "center_x",
    "center_y",
    "size_x",
    "size_y",
]

EVAL_COLUMNS = [
    "experiment_id",
    "dataset",
    "environment",
    "method",
    "bag_path",
    "detection_csv_path",
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
    "status",
]

DISTANCE_EVAL_COLUMNS = [
    "experiment_id",
    "dataset",
    "environment",
    "method",
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
    "status",
]


@dataclass(frozen=True)
class ObstacleSpec:
    model_name: str
    size_x: float
    size_y: float


@dataclass(frozen=True)
class Box2D:
    timestamp: float
    center_x: float
    center_y: float
    size_x: float
    size_y: float
    label: str = ""
    score: float = 1.0

    @property
    def x_min(self):
        return self.center_x - self.size_x / 2.0

    @property
    def x_max(self):
        return self.center_x + self.size_x / 2.0

    @property
    def y_min(self):
        return self.center_y - self.size_y / 2.0

    @property
    def y_max(self):
        return self.center_y + self.size_y / 2.0

    def distance_xy(self):
        return math.hypot(self.center_x, self.center_y)


def quaternion_to_yaw(orientation):
    x = float(orientation.x)
    y = float(orientation.y)
    z = float(orientation.z)
    w = float(orientation.w)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def world_delta_to_body_xy(delta_x, delta_y, ego_yaw):
    cos_yaw = math.cos(ego_yaw)
    sin_yaw = math.sin(ego_yaw)
    body_x = cos_yaw * delta_x + sin_yaw * delta_y
    body_y = -sin_yaw * delta_x + cos_yaw * delta_y
    return body_x, body_y


def box_iou(a, b):
    inter_x_min = max(a.x_min, b.x_min)
    inter_x_max = min(a.x_max, b.x_max)
    inter_y_min = max(a.y_min, b.y_min)
    inter_y_max = min(a.y_max, b.y_max)
    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, a.size_x) * max(0.0, a.size_y)
    area_b = max(0.0, b.size_x) * max(0.0, b.size_y)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def load_gt_config(path):
    with Path(path).expanduser().open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    specs = []
    for item in raw.get("obstacles", []):
        specs.append(
            ObstacleSpec(
                model_name=str(item.get("model_name", "")),
                size_x=float(item.get("size_x", 0.0)),
                size_y=float(item.get("size_y", 0.0)),
            )
        )
    return {
        "ego_model": str(raw.get("ego_model", "iris_0")),
        "model_states_topic": str(raw.get("model_states_topic", DEFAULT_MODEL_STATES_TOPIC)),
        "frame_id": str(raw.get("frame_id", "iris_0/base_link")),
        "iou_threshold": float(raw.get("iou_threshold", 0.1)),
        "time_tolerance_sec": float(raw.get("time_tolerance_sec", 0.05)),
        "obstacles": tuple(spec for spec in specs if spec.model_name and spec.size_x > 0 and spec.size_y > 0),
    }


def read_detection_csv(path, start_time=None, end_time=None):
    csv_path = Path(path) if path else None
    if not csv_path or not csv_path.exists():
        return {}
    frames = {}
    with csv_path.open("r", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                timestamp = float(row.get("timestamp", ""))
                center_x = float(row.get("center_x", ""))
                center_y = float(row.get("center_y", ""))
                size_x = float(row.get("size_x", ""))
                size_y = float(row.get("size_y", ""))
                score = float(row.get("score", "1.0"))
            except ValueError:
                continue
            if start_time is not None and timestamp < start_time:
                continue
            if end_time is not None and timestamp > end_time:
                continue
            frames.setdefault(timestamp, []).append(
                Box2D(
                    timestamp=timestamp,
                    center_x=center_x,
                    center_y=center_y,
                    size_x=size_x,
                    size_y=size_y,
                    score=score,
                    label=str(row.get("obstacle_id", "")),
                )
            )
    return frames


def _nearest_time(sorted_times, target):
    if not sorted_times:
        return None
    import bisect

    index = bisect.bisect_left(sorted_times, target)
    candidates = []
    if index < len(sorted_times):
        candidates.append(sorted_times[index])
    if index > 0:
        candidates.append(sorted_times[index - 1])
    return min(candidates, key=lambda item: abs(item - target)) if candidates else None


def _gt_boxes_from_model_state(timestamp, message, obstacle_specs, ego_model, frame_id):
    name_to_index = {name: index for index, name in enumerate(message.name)}
    ego_index = name_to_index.get(ego_model)
    if ego_index is None:
        return []
    ego_pose = message.pose[ego_index]
    ego_yaw = quaternion_to_yaw(ego_pose.orientation)
    ego_x = ego_pose.position.x
    ego_y = ego_pose.position.y
    boxes = []
    for spec in obstacle_specs:
        obstacle_index = name_to_index.get(spec.model_name)
        if obstacle_index is None:
            continue
        obstacle_pose = message.pose[obstacle_index]
        center_x, center_y = world_delta_to_body_xy(
            obstacle_pose.position.x - ego_x,
            obstacle_pose.position.y - ego_y,
            ego_yaw,
        )
        boxes.append(
            Box2D(
                timestamp=timestamp,
                center_x=center_x,
                center_y=center_y,
                size_x=spec.size_x,
                size_y=spec.size_y,
                label=spec.model_name,
            )
        )
    return boxes


def read_gt_frames_from_bag(bag_path, gt_config, target_times):
    try:
        import rosbag
    except ImportError as exc:
        raise RuntimeError("rosbag Python module is unavailable; source ROS setup before running") from exc

    path = Path(bag_path)
    if not path.exists() or not target_times:
        return {}, None, None

    topic = gt_config["model_states_topic"]
    model_states = []
    with rosbag.Bag(str(path), "r") as bag:
        start_time = bag.get_start_time()
        end_time = bag.get_end_time()
        for _, message, stamp in bag.read_messages(topics=[topic]):
            model_states.append((stamp.to_sec(), message))

    times = [item[0] for item in model_states]
    gt_frames = {}
    for target in sorted(target_times):
        nearest = _nearest_time(times, target)
        if nearest is None or abs(nearest - target) > gt_config["time_tolerance_sec"]:
            gt_frames[target] = []
            continue
        message = model_states[times.index(nearest)][1]
        gt_frames[target] = _gt_boxes_from_model_state(
            target,
            message,
            gt_config["obstacles"],
            gt_config["ego_model"],
            gt_config["frame_id"],
        )
    return gt_frames, start_time, end_time


def greedy_match_frame(detections, gt_boxes, iou_threshold):
    candidates = []
    for det_index, detection in enumerate(detections):
        for gt_index, gt_box in enumerate(gt_boxes):
            iou = box_iou(detection, gt_box)
            if iou >= iou_threshold:
                candidates.append((iou, det_index, gt_index))
    candidates.sort(reverse=True)
    matched_detections = set()
    matched_gt = set()
    matched_ious = []
    for iou, det_index, gt_index in candidates:
        if det_index in matched_detections or gt_index in matched_gt:
            continue
        matched_detections.add(det_index)
        matched_gt.add(gt_index)
        matched_ious.append(iou)
    tp = len(matched_ious)
    fp = max(0, len(detections) - tp)
    fn = max(0, len(gt_boxes) - tp)
    return tp, fp, fn, matched_ious


def metrics_from_counts(tp, fp, fn, ious):
    precision = tp / float(tp + fp) if (tp + fp) else None
    recall = tp / float(tp + fn) if (tp + fn) else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    mean_iou = sum(ious) / len(ious) if ious else None
    miss_rate = fn / float(tp + fn) if (tp + fn) else None
    false_positive_rate = fp / float(tp + fp) if (tp + fp) else None
    return precision, recall, f1, mean_iou, miss_rate, false_positive_rate


def evaluate_detection_frames(detection_frames, gt_frames, iou_threshold):
    all_times = sorted(set(detection_frames) | set(gt_frames))
    tp = fp = fn = 0
    ious = []
    for timestamp in all_times:
        frame_tp, frame_fp, frame_fn, frame_ious = greedy_match_frame(
            detection_frames.get(timestamp, []),
            gt_frames.get(timestamp, []),
            iou_threshold,
        )
        tp += frame_tp
        fp += frame_fp
        fn += frame_fn
        ious.extend(frame_ious)
    precision, recall, f1, mean_iou, miss_rate, false_positive_rate = metrics_from_counts(tp, fp, fn, ious)
    return {
        "gt_frames": sum(1 for boxes in gt_frames.values() if boxes),
        "detection_frames": sum(1 for boxes in detection_frames.values() if boxes),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "miss_rate": miss_rate,
        "false_positive_rate": false_positive_rate,
    }


def parse_distance_bins(text):
    raw_edges = []
    for item in str(text).split(","):
        stripped = item.strip().lower()
        if not stripped:
            continue
        if stripped in ("inf", "+inf", "infinity"):
            raw_edges.append(float("inf"))
        else:
            raw_edges.append(float(stripped))
    if len(raw_edges) < 2:
        raise ValueError("distance bins require at least two edges")
    if raw_edges[0] < 0.0:
        raise ValueError("distance bins must be non-negative")
    for previous, current in zip(raw_edges, raw_edges[1:]):
        if current <= previous:
            raise ValueError("distance bins must be strictly increasing")
    return tuple((raw_edges[index], raw_edges[index + 1]) for index in range(len(raw_edges) - 1))


def distance_bin_label(min_distance, max_distance):
    min_text = format_number(min_distance, digits=1)
    max_text = "inf" if math.isinf(max_distance) else format_number(max_distance, digits=1)
    return "{}-{}m".format(min_text, max_text)


def _filter_frames_by_distance(frames, min_distance, max_distance):
    filtered = {}
    for timestamp, boxes in frames.items():
        kept = []
        for box in boxes:
            distance = box.distance_xy()
            if distance >= min_distance and distance < max_distance:
                kept.append(box)
        if kept:
            filtered[timestamp] = kept
    return filtered


def evaluate_detection_distance_bins(detection_frames, gt_frames, iou_threshold, distance_bins):
    rows = []
    for min_distance, max_distance in distance_bins:
        filtered_detections = _filter_frames_by_distance(detection_frames, min_distance, max_distance)
        filtered_gt = _filter_frames_by_distance(gt_frames, min_distance, max_distance)
        row = {
            "distance_bin": distance_bin_label(min_distance, max_distance),
            "distance_min_m": min_distance,
            "distance_max_m": max_distance,
        }
        row.update(evaluate_detection_frames(filtered_detections, filtered_gt, iou_threshold))
        rows.append(row)
    return rows


def _bag_time_bounds(bag_path):
    try:
        import rosbag
    except ImportError as exc:
        raise RuntimeError("rosbag Python module is unavailable; source ROS setup before running") from exc

    with rosbag.Bag(str(bag_path), "r") as bag:
        return bag.get_start_time(), bag.get_end_time()


def evaluate_method_run(config, dataset, environment, method, result_dir, gt_config, bag_root=None):
    bag_path = bag_path_for_run(config, dataset, environment, method, bag_root=bag_root)
    detection_path = detection_csv_for_method(result_dir, method)
    base_row = {
        "experiment_id": config.experiment_id,
        "dataset": dataset.name,
        "environment": environment,
        "method": method,
        "bag_path": str(bag_path),
        "detection_csv_path": str(detection_path) if detection_path else "",
    }
    if method == "vins_only":
        return {**base_row, "status": "not_detection_method"}, {}
    if not bag_path.exists():
        return {**base_row, "status": "missing_bag"}, {}
    if not detection_path or not detection_path.exists():
        return {**base_row, "status": "missing_detection_csv"}, {}
    if not gt_config["obstacles"]:
        return {**base_row, "status": "missing_gt_config"}, {}

    start_time, end_time = _bag_time_bounds(bag_path)
    detection_frames = read_detection_csv(detection_path, start_time=start_time, end_time=end_time)
    target_times = sorted(detection_frames)
    gt_frames, _, _ = read_gt_frames_from_bag(bag_path, gt_config, target_times)
    if not any(gt_frames.values()):
        row = {**base_row, "status": "no_gt_models_in_bag"}
        row.update(evaluate_detection_frames(detection_frames, gt_frames, gt_config["iou_threshold"]))
        return row, gt_frames

    row = {**base_row, "status": "ok"}
    row.update(evaluate_detection_frames(detection_frames, gt_frames, gt_config["iou_threshold"]))
    return row, gt_frames


def evaluate_method_distance_bins(config, dataset, environment, method, result_dir, gt_config, distance_bins, bag_root=None):
    bag_path = bag_path_for_run(config, dataset, environment, method, bag_root=bag_root)
    detection_path = detection_csv_for_method(result_dir, method)
    base_row = {
        "experiment_id": config.experiment_id,
        "dataset": dataset.name,
        "environment": environment,
        "method": method,
    }
    if method == "vins_only":
        return [{**base_row, "status": "not_detection_method", **_distance_bin_metadata(item)} for item in distance_bins]
    if not bag_path.exists():
        return [{**base_row, "status": "missing_bag", **_distance_bin_metadata(item)} for item in distance_bins]
    if not detection_path or not detection_path.exists():
        return [{**base_row, "status": "missing_detection_csv", **_distance_bin_metadata(item)} for item in distance_bins]
    if not gt_config["obstacles"]:
        return [{**base_row, "status": "missing_gt_config", **_distance_bin_metadata(item)} for item in distance_bins]

    start_time, end_time = _bag_time_bounds(bag_path)
    detection_frames = read_detection_csv(detection_path, start_time=start_time, end_time=end_time)
    target_times = sorted(detection_frames)
    gt_frames, _, _ = read_gt_frames_from_bag(bag_path, gt_config, target_times)
    rows = []
    for row in evaluate_detection_distance_bins(
        detection_frames,
        gt_frames,
        gt_config["iou_threshold"],
        distance_bins,
    ):
        status = "ok" if any(gt_frames.values()) else "no_gt_models_in_bag"
        rows.append({**base_row, **row, "status": status})
    return rows


def _distance_bin_metadata(distance_bin):
    min_distance, max_distance = distance_bin
    return {
        "distance_bin": distance_bin_label(min_distance, max_distance),
        "distance_min_m": min_distance,
        "distance_max_m": max_distance,
    }


def evaluate_experiment(config, result_dir, gt_config, bag_root=None):
    rows = []
    gt_frames_by_method = {}
    for dataset in config.datasets:
        for environment in config.environments:
            for method in config.methods:
                row, gt_frames = evaluate_method_run(
                    config,
                    dataset,
                    environment,
                    method,
                    result_dir,
                    gt_config,
                    bag_root=bag_root,
                )
                rows.append(row)
                if gt_frames:
                    gt_frames_by_method[method] = gt_frames
    return rows, gt_frames_by_method


def evaluate_experiment_distance_bins(config, result_dir, gt_config, distance_bins, bag_root=None):
    rows = []
    for dataset in config.datasets:
        for environment in config.environments:
            for method in config.methods:
                rows.extend(
                    evaluate_method_distance_bins(
                        config,
                        dataset,
                        environment,
                        method,
                        result_dir,
                        gt_config,
                        distance_bins,
                        bag_root=bag_root,
                    )
                )
    return rows


def write_eval_summary_csv(path, rows):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=EVAL_COLUMNS)
        writer.writeheader()
        for row in rows:
            output = {}
            for column in EVAL_COLUMNS:
                value = row.get(column, "")
                if isinstance(value, float):
                    value = format_number(value)
                output[column] = value
            writer.writerow(output)
    return output_path


def write_distance_eval_summary_csv(path, rows):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=DISTANCE_EVAL_COLUMNS)
        writer.writeheader()
        for row in rows:
            output = {}
            for column in DISTANCE_EVAL_COLUMNS:
                value = row.get(column, "")
                if isinstance(value, float):
                    value = "inf" if math.isinf(value) else format_number(value)
                output[column] = value
            writer.writerow(output)
    return output_path


def write_gt_csv(path, gt_frames_by_method, frame_id="iris_0/base_link"):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=GT_COLUMNS)
        writer.writeheader()
        for method, frames in sorted(gt_frames_by_method.items()):
            for timestamp, boxes in sorted(frames.items()):
                for box in boxes:
                    writer.writerow(
                        {
                            "method": method,
                            "timestamp": format_number(timestamp, digits=9),
                            "model_name": box.label,
                            "frame_id": frame_id,
                            "center_x": format_number(box.center_x),
                            "center_y": format_number(box.center_y),
                            "size_x": format_number(box.size_x),
                            "size_y": format_number(box.size_y),
                        }
                    )
    return output_path


def update_metrics_csv_with_detection_metrics(metrics_path, eval_rows):
    path = Path(metrics_path)
    with path.open("r", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    by_key = {
        (row.get("experiment_id", ""), row.get("dataset", ""), row.get("environment", ""), row.get("method", "")): row
        for row in eval_rows
        if row.get("status") == "ok"
    }
    metric_columns = ("precision", "recall", "f1", "mean_iou", "miss_rate", "false_positive_rate")
    for row in rows:
        key = (row.get("experiment_id", ""), row.get("dataset", ""), row.get("environment", ""), row.get("method", ""))
        eval_row = by_key.get(key)
        if not eval_row:
            continue
        for metric in metric_columns:
            if metric in fieldnames and eval_row.get(metric) is not None:
                row[metric] = format_number(eval_row[metric])

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
