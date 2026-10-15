"""Smoke-run evaluation helpers for recorded BEV + VINS rosbag samples."""

import csv
import os
from dataclasses import dataclass
from pathlib import Path


BEV_STATUS_TOPIC = "/bev_obstacle_detection/status"
BEV_OCCUPANCY_TOPIC = "/bev_obstacle_detection/occupancy_grid"
BEV_MARKERS_TOPIC = "/bev_obstacle_detection/obstacle_markers"

SUMMARY_COLUMNS = [
    "experiment_id",
    "dataset",
    "environment",
    "method",
    "bag_path",
    "bag_exists",
    "bag_duration_sec",
    "bag_size_mb",
    "compression",
    "total_messages",
    "camera_rate_hz",
    "camera_right_rate_hz",
    "lidar_rate_hz",
    "imu_rate_hz",
    "vins_odom_rate_hz",
    "vins_camera_pose_rate_hz",
    "bev_grid_rate_hz",
    "bev_marker_rate_hz",
    "status_rate_hz",
    "status_ready_ratio",
    "detection_csv_path",
    "detection_rows",
    "detection_frames",
    "mean_obstacles_per_detection_frame",
    "detection_frame_rate_hz",
    "primary_output_fps",
    "vins_gap_failure_rate",
]


@dataclass(frozen=True)
class TopicStats:
    topic: str
    message_count: int = 0
    first_time: float = None
    last_time: float = None
    max_gap_sec: float = None

    @property
    def span_sec(self):
        if self.first_time is None or self.last_time is None:
            return None
        return max(0.0, self.last_time - self.first_time)

    def rate_hz(self, bag_duration_sec=None):
        duration = bag_duration_sec if bag_duration_sec and bag_duration_sec > 0 else self.span_sec
        if not duration:
            return None
        return self.message_count / duration


@dataclass(frozen=True)
class DetectionCsvStats:
    rows: int = 0
    frames: int = 0
    mean_obstacles_per_frame: float = None
    frame_rate_hz: float = None


def unique_nonempty(items):
    result = []
    seen = set()
    for item in items:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


def topic_stats_from_times(topic, times):
    clean_times = sorted(float(item) for item in times)
    if not clean_times:
        return TopicStats(topic=topic, message_count=0)
    gaps = [right - left for left, right in zip(clean_times, clean_times[1:])]
    max_gap = max(gaps) if gaps else 0.0
    return TopicStats(
        topic=topic,
        message_count=len(clean_times),
        first_time=clean_times[0],
        last_time=clean_times[-1],
        max_gap_sec=max_gap,
    )


def gap_failure_rate(times, allowed_gap_sec):
    clean_times = sorted(float(item) for item in times)
    if len(clean_times) < 2:
        return 1.0 if clean_times else None
    duration = clean_times[-1] - clean_times[0]
    if duration <= 0:
        return 0.0
    excessive_gap = 0.0
    for left, right in zip(clean_times, clean_times[1:]):
        excessive_gap += max(0.0, (right - left) - allowed_gap_sec)
    return min(1.0, excessive_gap / duration)


def detection_csv_stats(path, bag_duration_sec=None, start_time=None, end_time=None):
    if not path:
        return DetectionCsvStats()
    csv_path = Path(path)
    if not csv_path.exists() or not csv_path.is_file():
        return DetectionCsvStats()

    timestamps = []
    rows = 0
    with csv_path.open("r", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows += 1
            try:
                timestamp = float(row.get("timestamp", ""))
            except ValueError:
                continue
            if start_time is not None and timestamp < start_time:
                continue
            if end_time is not None and timestamp > end_time:
                continue
            timestamps.append(timestamp)

    unique_timestamps = sorted(set(timestamps))
    frame_count = len(unique_timestamps)
    filtered_rows = len(timestamps)
    mean_obstacles = (float(filtered_rows) / frame_count) if frame_count else None
    frame_rate = None
    if frame_count:
        if bag_duration_sec and bag_duration_sec > 0:
            frame_rate = frame_count / bag_duration_sec
        elif frame_count > 1:
            span = unique_timestamps[-1] - unique_timestamps[0]
            frame_rate = frame_count / span if span > 0 else None

    return DetectionCsvStats(
        rows=filtered_rows,
        frames=frame_count,
        mean_obstacles_per_frame=mean_obstacles,
        frame_rate_hz=frame_rate,
    )


def compression_info(bag):
    try:
        info = list(bag.get_compression_info())
    except Exception:
        return "", None, None
    if not info:
        return "", None, None
    algorithm = str(info[0])
    uncompressed = int(info[1]) if len(info) > 1 else None
    compressed = int(info[2]) if len(info) > 2 else None
    return algorithm, uncompressed, compressed


def read_bag_topic_stats(bag_path, topics):
    try:
        import rosbag
    except ImportError as exc:
        raise RuntimeError("rosbag Python module is unavailable; source ROS setup before running") from exc

    path = Path(bag_path)
    if not path.exists():
        return {
            "bag_exists": False,
            "bag_start_time": "",
            "bag_end_time": "",
            "bag_duration_sec": "",
            "bag_size_mb": "",
            "compression": "",
            "total_messages": "",
            "topic_stats": {},
            "status_ready_ratio": None,
            "topic_times": {},
        }

    requested_topics = unique_nonempty(topics)
    topic_times = {topic: [] for topic in requested_topics}
    status_ready = []

    with rosbag.Bag(str(path), "r") as bag:
        start_time = bag.get_start_time()
        end_time = bag.get_end_time()
        duration = max(0.0, end_time - start_time)
        algorithm, _, _ = compression_info(bag)
        topic_info = bag.get_type_and_topic_info()[1]
        total_messages = sum(item.message_count for item in topic_info.values())
        available_requested = [topic for topic in requested_topics if topic in topic_info]
        for topic, message, stamp in bag.read_messages(topics=available_requested):
            topic_times.setdefault(topic, []).append(stamp.to_sec())
            if topic == BEV_STATUS_TOPIC and hasattr(message, "data"):
                status_ready.append("ready=True" in message.data)

    ready_ratio = None
    if status_ready:
        ready_ratio = sum(1 for item in status_ready if item) / float(len(status_ready))

    return {
        "bag_exists": True,
        "bag_start_time": start_time,
        "bag_end_time": end_time,
        "bag_duration_sec": duration,
        "bag_size_mb": os.path.getsize(str(path)) / (1024.0 * 1024.0),
        "compression": algorithm,
        "total_messages": total_messages,
        "topic_stats": {topic: topic_stats_from_times(topic, times) for topic, times in topic_times.items()},
        "status_ready_ratio": ready_ratio,
        "topic_times": topic_times,
    }


def detection_csv_for_method(result_dir, method):
    if method == "bev_only":
        return Path(result_dir) / "bev_only_detections.csv"
    if method == "bev_vins_fusion":
        return Path(result_dir) / "fusion_detections.csv"
    return None


def bag_path_for_run(config, dataset, environment, method, bag_root=None):
    root = Path(bag_root) if bag_root else Path(dataset.root) / "rosbags"
    return root / dataset.name / environment / "{}.bag".format(method)


def topics_for_summary(config):
    topics = [
        config.topics.get("camera", ""),
        config.topics.get("camera_right", ""),
        config.topics.get("lidar", ""),
        config.topics.get("imu", ""),
        config.topics.get("vins_odom", ""),
        config.topics.get("vins_camera_pose", ""),
        BEV_STATUS_TOPIC,
        BEV_OCCUPANCY_TOPIC,
        BEV_MARKERS_TOPIC,
    ]
    return unique_nonempty(topics)


def _rate(topic_stats, topic, duration):
    stats = topic_stats.get(topic)
    return stats.rate_hz(duration) if stats else None


def format_number(value, digits=4):
    if value is None or value == "":
        return ""
    return ("{:.%df}" % digits).format(float(value))


def summarize_method_run(config, dataset, environment, method, result_dir, bag_root=None):
    bag_path = bag_path_for_run(config, dataset, environment, method, bag_root=bag_root)
    bag_summary = read_bag_topic_stats(bag_path, topics_for_summary(config))
    duration = bag_summary["bag_duration_sec"] if bag_summary["bag_exists"] else None
    start_time = bag_summary["bag_start_time"] if bag_summary["bag_exists"] else None
    end_time = bag_summary["bag_end_time"] if bag_summary["bag_exists"] else None
    topic_stats = bag_summary["topic_stats"]
    topic_times = bag_summary["topic_times"]

    detection_path = detection_csv_for_method(result_dir, method)
    detection_stats = detection_csv_stats(
        detection_path,
        bag_duration_sec=duration,
        start_time=start_time,
        end_time=end_time,
    )

    vins_odom_topic = config.topics.get("vins_odom", "")
    vins_rate = _rate(topic_stats, vins_odom_topic, duration)
    bev_grid_rate = _rate(topic_stats, BEV_OCCUPANCY_TOPIC, duration)
    primary_output_fps = bev_grid_rate if method in ("bev_only", "bev_vins_fusion") else vins_rate

    vins_failure = None
    if method in ("vins_only", "bev_vins_fusion"):
        vins_failure = gap_failure_rate(topic_times.get(vins_odom_topic, []), allowed_gap_sec=0.5)

    row = {
        "experiment_id": config.experiment_id,
        "dataset": dataset.name,
        "environment": environment,
        "method": method,
        "bag_path": str(bag_path),
        "bag_exists": str(bool(bag_summary["bag_exists"])).lower(),
        "bag_duration_sec": format_number(duration),
        "bag_size_mb": format_number(bag_summary["bag_size_mb"], digits=2),
        "compression": bag_summary["compression"],
        "total_messages": bag_summary["total_messages"],
        "camera_rate_hz": format_number(_rate(topic_stats, config.topics.get("camera", ""), duration)),
        "camera_right_rate_hz": format_number(_rate(topic_stats, config.topics.get("camera_right", ""), duration)),
        "lidar_rate_hz": format_number(_rate(topic_stats, config.topics.get("lidar", ""), duration)),
        "imu_rate_hz": format_number(_rate(topic_stats, config.topics.get("imu", ""), duration)),
        "vins_odom_rate_hz": format_number(vins_rate),
        "vins_camera_pose_rate_hz": format_number(
            _rate(topic_stats, config.topics.get("vins_camera_pose", ""), duration)
        ),
        "bev_grid_rate_hz": format_number(bev_grid_rate),
        "bev_marker_rate_hz": format_number(_rate(topic_stats, BEV_MARKERS_TOPIC, duration)),
        "status_rate_hz": format_number(_rate(topic_stats, BEV_STATUS_TOPIC, duration)),
        "status_ready_ratio": format_number(bag_summary["status_ready_ratio"]),
        "detection_csv_path": str(detection_path) if detection_path else "",
        "detection_rows": detection_stats.rows,
        "detection_frames": detection_stats.frames,
        "mean_obstacles_per_detection_frame": format_number(detection_stats.mean_obstacles_per_frame),
        "detection_frame_rate_hz": format_number(detection_stats.frame_rate_hz),
        "primary_output_fps": format_number(primary_output_fps),
        "vins_gap_failure_rate": format_number(vins_failure),
    }
    return row


def summarize_experiment(config, result_dir, bag_root=None):
    rows = []
    for dataset in config.datasets:
        for environment in config.environments:
            for method in config.methods:
                rows.append(
                    summarize_method_run(
                        config,
                        dataset,
                        environment,
                        method,
                        result_dir=result_dir,
                        bag_root=bag_root,
                    )
                )
    return rows


def write_summary_csv(path, rows):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})
    return output_path


def update_metrics_csv(metrics_path, summary_rows):
    path = Path(metrics_path)
    with path.open("r", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        metrics_rows = list(reader)

    summary_by_key = {
        (row["experiment_id"], row["dataset"], row["environment"], row["method"]): row for row in summary_rows
    }

    for row in metrics_rows:
        key = (row.get("experiment_id", ""), row.get("dataset", ""), row.get("environment", ""), row.get("method", ""))
        summary = summary_by_key.get(key)
        if not summary:
            continue
        if "fps" in fieldnames:
            row["fps"] = summary.get("primary_output_fps", "")
        if "tracking_failure_rate" in fieldnames and row.get("method") in ("vins_only", "bev_vins_fusion"):
            row["tracking_failure_rate"] = summary.get("vins_gap_failure_rate", "")

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)
    return path
