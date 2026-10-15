"""ADVIO public dataset inspection helpers."""

import csv
import json
from pathlib import Path

import numpy as np


ADVIO_SEQUENCE_FILES = {
    "ground_truth_poses": ("ground-truth/pose.csv", "ground-truth/poses.csv"),
    "iphone_frames_video": ("iphone/frames.mov",),
    "iphone_frame_timestamps": ("iphone/frames.csv",),
    "iphone_accelerometer": ("iphone/accelerometer.csv",),
    "iphone_gyroscope": ("iphone/gyro.csv", "iphone/gyroscope.csv"),
    "iphone_arkit": ("iphone/arkit.csv",),
}


def find_advio_sequences(root):
    root_path = Path(root).expanduser().resolve()
    candidates = []
    for path in sorted(root_path.rglob("advio-*")):
        if path.is_dir():
            candidates.append(path)
    return candidates


def count_csv_rows(path):
    csv_path = Path(path)
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8", errors="replace") as stream:
        return sum(1 for row in csv.reader(stream) if row)


def sample_csv_row(path):
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", errors="replace") as stream:
        for row in csv.reader(stream):
            if row:
                return row
    return []


def inspect_advio_sequence(sequence_root):
    root = Path(sequence_root).expanduser().resolve()
    files = {}
    for key, candidates in ADVIO_SEQUENCE_FILES.items():
        relative = next((candidate for candidate in candidates if (root / candidate).exists()), candidates[0])
        path = root / relative
        entry = {
            "relative_path": relative,
            "candidate_paths": list(candidates),
            "exists": path.exists(),
            "path": str(path),
        }
        if path.suffix.lower() == ".csv":
            entry["rows"] = count_csv_rows(path)
            entry["sample"] = sample_csv_row(path)
        elif path.exists():
            entry["size_bytes"] = path.stat().st_size
        files[key] = entry
    required_ok = all(files[key]["exists"] for key in ("ground_truth_poses", "iphone_accelerometer", "iphone_gyroscope"))
    camera_ok = files["iphone_frames_video"]["exists"] and files["iphone_frame_timestamps"]["exists"]
    return {
        "sequence_id": root.name,
        "root": str(root),
        "required_ok": required_ok,
        "camera_ok": camera_ok,
        "files": files,
    }


def inspect_advio_root(root):
    sequences = [inspect_advio_sequence(path) for path in find_advio_sequences(root)]
    return {
        "dataset": "advio",
        "root": str(Path(root).expanduser().resolve()),
        "sequence_count": len(sequences),
        "sequences": sequences,
    }


def write_advio_manifest(manifest, output_path):
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return output


def read_numeric_csv(path, expected_columns=None, max_rows=0):
    rows = []
    with Path(path).expanduser().open("r", encoding="utf-8", errors="replace") as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            values = [float(item) for item in row]
            if expected_columns is not None and len(values) != int(expected_columns):
                raise ValueError("{} expected {} columns, got {}".format(path, expected_columns, len(values)))
            rows.append(values)
            if max_rows > 0 and len(rows) >= int(max_rows):
                break
    return np.asarray(rows, dtype=float)


def nearest_rows_by_timestamp(primary_rows, secondary_rows):
    primary = np.asarray(primary_rows, dtype=float)
    secondary = np.asarray(secondary_rows, dtype=float)
    if primary.size == 0:
        return primary
    if secondary.size == 0:
        return np.hstack((primary, np.zeros((primary.shape[0], max(0, secondary.shape[1] - 1)))))
    indices = np.searchsorted(secondary[:, 0], primary[:, 0], side="left")
    indices = np.clip(indices, 0, secondary.shape[0] - 1)
    previous = np.maximum(indices - 1, 0)
    choose_previous = np.abs(secondary[previous, 0] - primary[:, 0]) < np.abs(secondary[indices, 0] - primary[:, 0])
    nearest = np.where(choose_previous, previous, indices)
    return np.hstack((primary, secondary[nearest, 1:]))
