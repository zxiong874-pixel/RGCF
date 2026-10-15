"""Helpers for selecting and downloading Boreas Objects labels."""

import json
import subprocess
from pathlib import Path

from bev_obstacle_detection.boreas_adapter import BOREAS_BUCKET_HTTP, s3_list_all_contents


BOREAS_OBJECT_SEQUENCE = "boreas-objects-v1"
BOREAS_OBJECT_LABEL_PREFIX = "{}/labels_detection/".format(BOREAS_OBJECT_SEQUENCE)
BOREAS_OBJECT_SENSOR_EXTENSIONS = {
    "camera": "png",
    "lidar": "bin",
    "radar": "png",
}


def object_split_interval(split_name):
    from pyboreas.data import splits

    split = getattr(splits, split_name)
    if not split:
        raise ValueError("empty Boreas object split: {}".format(split_name))
    intervals = []
    for item in split:
        if len(item) < 3:
            continue
        intervals.append((int(item[1]), int(item[2])))
    if not intervals:
        raise ValueError("Boreas object split has no timestamp interval: {}".format(split_name))
    return min(start for start, _ in intervals), max(end for _, end in intervals)


def label_timestamp_from_key(key):
    return int(Path(key).stem)


def list_object_label_files(start_us=None, end_us=None, max_pages=None):
    contents = s3_list_all_contents(
        prefix=BOREAS_OBJECT_LABEL_PREFIX,
        delimiter=None,
        timeout=60,
        max_pages=max_pages,
    )
    selected = []
    for item in contents:
        timestamp_us = label_timestamp_from_key(item["key"])
        if start_us is not None and timestamp_us < int(start_us):
            continue
        if end_us is not None and timestamp_us > int(end_us):
            continue
        selected.append(
            {
                "key": item["key"],
                "relative_path": item["key"][len(BOREAS_OBJECT_SEQUENCE) + 1 :],
                "timestamp_us": timestamp_us,
                "size": int(item["size"]),
            }
        )
    return sorted(selected, key=lambda item: item["timestamp_us"])


def object_sensor_key(sensor, timestamp_us):
    extension = BOREAS_OBJECT_SENSOR_EXTENSIONS[sensor]
    return "{}/{}/{}.{}".format(BOREAS_OBJECT_SEQUENCE, sensor, int(timestamp_us), extension)


def object_sensor_item(sensor, timestamp_us):
    key = object_sensor_key(sensor, timestamp_us)
    listing = s3_list_all_contents(prefix=key, delimiter=None, timeout=60, max_pages=1)
    if not listing:
        return None
    item = listing[0]
    if item["key"] != key:
        return None
    return {
        "sensor": sensor,
        "key": key,
        "relative_path": key[len(BOREAS_OBJECT_SEQUENCE) + 1 :],
        "timestamp_us": int(timestamp_us),
        "gps_time_us": int(timestamp_us),
        "size": int(item["size"]),
    }


def matching_sensor_items(label_items, sensors):
    sensor_items = []
    for label in label_items:
        for sensor in sensors:
            item = object_sensor_item(sensor, label["timestamp_us"])
            if item is not None:
                sensor_items.append(item)
    return sensor_items


def download_s3_item(item, root, max_time=0, retry_count=5):
    output = Path(root).expanduser() / BOREAS_OBJECT_SEQUENCE / item["relative_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(item["size"])
    local_size = output.stat().st_size if output.exists() else 0
    if local_size == expected_size:
        return {"path": str(output), "status": "complete", "bytes": local_size}
    if local_size > expected_size:
        output.unlink()
        local_size = 0
    cmd = [
        "curl",
        "-L",
        "-C",
        "-",
        "--retry",
        str(retry_count),
        "--retry-delay",
        "2",
        "--connect-timeout",
        "20",
        "-o",
        str(output),
        "{}/{}".format(BOREAS_BUCKET_HTTP, item["key"]),
    ]
    if max_time > 0:
        cmd[1:1] = ["--max-time", str(max_time)]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as exc:
        local_size = output.stat().st_size if output.exists() else 0
        return {"path": str(output), "status": "failed", "bytes": local_size, "returncode": exc.returncode}
    local_size = output.stat().st_size if output.exists() else 0
    status = "complete" if local_size == expected_size else "partial"
    return {"path": str(output), "status": status, "bytes": local_size}


def write_object_label_manifest(
    path,
    selected,
    downloaded=None,
    split_name=None,
    start_us=None,
    end_us=None,
    selected_sensor_files=None,
):
    manifest_path = Path(path).expanduser()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "dataset": "boreas_objects",
        "sequence": BOREAS_OBJECT_SEQUENCE,
        "split": split_name,
        "start_us": start_us,
        "end_us": end_us,
        "label_files": len(selected),
        "expected_bytes": sum(int(item["size"]) for item in selected),
        "selected": selected,
        "selected_sensor_files": selected_sensor_files or [],
        "expected_sensor_bytes": sum(int(item["size"]) for item in selected_sensor_files or []),
        "downloaded": downloaded or [],
    }
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path
