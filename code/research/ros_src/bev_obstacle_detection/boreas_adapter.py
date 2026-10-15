"""Lightweight Boreas public dataset onboarding helpers."""

import csv
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


BOREAS_BUCKET_HTTP = "http://boreas.s3.amazonaws.com"
BOREAS_BUCKET_HTTPS = "https://boreas.s3.amazonaws.com"
S3_NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}

BOREAS_METADATA_FILES = (
    "calib/P_camera.txt",
    "calib/T_applanix_lidar.txt",
    "calib/T_camera_lidar.txt",
    "calib/T_radar_lidar.txt",
    "calib/camera0_intrinsics.yaml",
    "applanix/camera_poses.csv",
    "applanix/lidar_poses.csv",
    "applanix/radar_poses.csv",
    "applanix/ros_and_gps_time.csv",
)

BOREAS_SENSOR_DIRS = ("camera", "lidar", "radar")


def s3_list(prefix="", delimiter="/", timeout=30, use_https=True, retries=3, continuation_token=None, start_after=None):
    query = {"list-type": "2"}
    if prefix:
        query["prefix"] = prefix
    if delimiter is not None:
        query["delimiter"] = delimiter
    if continuation_token:
        query["continuation-token"] = continuation_token
    if start_after:
        query["start-after"] = start_after
    last_error = None
    schemes = (BOREAS_BUCKET_HTTPS, BOREAS_BUCKET_HTTP) if use_https else (BOREAS_BUCKET_HTTP,)
    for _attempt in range(int(retries)):
        for base in schemes:
            url = "{}?{}".format(base, urlencode(query))
            try:
                root = ET.fromstring(urlopen(url, timeout=timeout).read())
                break
            except Exception as exc:
                last_error = exc
        else:
            continue
        break
    else:
        raise last_error
    prefixes = [item.find("s:Prefix", S3_NS).text for item in root.findall("s:CommonPrefixes", S3_NS)]
    contents = []
    for item in root.findall("s:Contents", S3_NS):
        contents.append(
            {
                "key": item.find("s:Key", S3_NS).text,
                "size": int(item.find("s:Size", S3_NS).text),
            }
        )
    next_token = root.find("s:NextContinuationToken", S3_NS)
    is_truncated = root.find("s:IsTruncated", S3_NS)
    return {
        "prefixes": prefixes,
        "contents": contents,
        "is_truncated": is_truncated is not None and is_truncated.text == "true",
        "next_continuation_token": next_token.text if next_token is not None else None,
    }


def s3_list_all_contents(prefix="", delimiter=None, timeout=30, use_https=True, retries=3, max_pages=None):
    contents = []
    continuation_token = None
    pages = 0
    while True:
        listing = s3_list(
            prefix=prefix,
            delimiter=delimiter,
            timeout=timeout,
            use_https=use_https,
            retries=retries,
            continuation_token=continuation_token,
        )
        pages += 1
        contents.extend(listing["contents"])
        continuation_token = listing.get("next_continuation_token")
        if not listing.get("is_truncated") or not continuation_token:
            break
        if max_pages is not None and pages >= int(max_pages):
            break
    return contents


def list_boreas_sequences(timeout=30):
    listing = s3_list(delimiter="/", timeout=timeout)
    return [prefix.strip("/") for prefix in listing["prefixes"] if prefix.startswith("boreas-")]


def list_boreas_sequence_metadata(sequence_id, timeout=30):
    keys = {}
    for subdir in ("calib/", "applanix/"):
        prefix = "{}/{}".format(sequence_id, subdir)
        listing = s3_list(prefix=prefix, delimiter=None, timeout=timeout)
        for item in listing["contents"]:
            relative = item["key"][len(sequence_id) + 1 :]
            keys[relative] = item["size"]
    return keys


def list_boreas_sensor_files(sequence_id, sensor, timeout=30):
    if sensor not in BOREAS_SENSOR_DIRS:
        raise ValueError("unsupported Boreas sensor: {}".format(sensor))
    listing = s3_list(prefix="{}/{}/".format(sequence_id, sensor), delimiter=None, timeout=timeout)
    files = []
    for item in listing["contents"]:
        key = item["key"]
        name = Path(key).name
        stem = Path(name).stem
        if not stem.isdigit():
            continue
        files.append(
            {
                "sensor": sensor,
                "key": key,
                "relative_path": key[len(sequence_id) + 1 :],
                "timestamp_us": float(stem),
                "size": item["size"],
            }
        )
    return sorted(files, key=lambda item: item["timestamp_us"])


def nearest_sensor_file(files, gps_time_us):
    if not files:
        return None
    return min(files, key=lambda item: abs(item["timestamp_us"] - gps_time_us))


def count_csv_rows(path):
    csv_path = Path(path)
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8", errors="replace") as stream:
        return sum(1 for row in csv.reader(stream) if row)


def read_matrix_txt(path):
    matrix_path = Path(path)
    rows = []
    with matrix_path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped:
                rows.append([float(value) for value in stripped.split()])
    return rows


def radar_image_to_points(
    image,
    range_resolution_m=0.2,
    min_range_m=1.0,
    max_range_m=None,
    intensity_percentile=99.5,
    max_points=20000,
    azimuth_offset_rad=-math.pi,
    azimuth_span_rad=2.0 * math.pi,
):
    """Convert a Boreas/Navtech-style polar radar image into approximate xyzi points.

    Boreas radar PNGs are used here as a practical BEV smoke input. The conversion
    treats image rows as range bins and columns as uniformly spaced azimuth bins.
    It is intentionally configurable because exact radar projection parameters can
    differ by export format.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for radar image conversion") from exc

    array = np.asarray(image)
    if array.ndim == 3:
        array = array.mean(axis=2)
    if array.ndim != 2:
        raise ValueError("radar image must be a 2D or 3D array")
    if array.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    values = array.astype(np.float32, copy=False)
    height, width = values.shape
    if height <= 0 or width <= 0:
        return np.zeros((0, 4), dtype=np.float32)

    percentile = min(100.0, max(0.0, float(intensity_percentile)))
    threshold = float(np.percentile(values, percentile))
    candidates = np.argwhere(values >= threshold)
    if candidates.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    if max_range_m is not None:
        ranges = (candidates[:, 0].astype(np.float32) + 0.5) * float(range_resolution_m)
        candidates = candidates[(ranges >= float(min_range_m)) & (ranges <= float(max_range_m))]
    else:
        ranges = (candidates[:, 0].astype(np.float32) + 0.5) * float(range_resolution_m)
        candidates = candidates[ranges >= float(min_range_m)]
    if candidates.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    intensities = values[candidates[:, 0], candidates[:, 1]]
    if max_points and len(candidates) > int(max_points):
        keep_count = int(max_points)
        keep = np.argpartition(intensities, -keep_count)[-keep_count:]
        candidates = candidates[keep]
        intensities = intensities[keep]

    range_m = (candidates[:, 0].astype(np.float32) + 0.5) * float(range_resolution_m)
    azimuth = float(azimuth_offset_rad) + (
        (candidates[:, 1].astype(np.float32) + 0.5) / float(width)
    ) * float(azimuth_span_rad)
    x = range_m * np.cos(azimuth)
    y = range_m * np.sin(azimuth)
    z = np.zeros_like(x)
    max_intensity = float(values.max())
    if max_intensity > 0.0:
        intensity = intensities.astype(np.float32) / max_intensity
    else:
        intensity = intensities.astype(np.float32)
    return np.column_stack((x, y, z, intensity)).astype(np.float32)


def boreas_radar_resolution(timestamp_sec=None):
    """Return the native Boreas Navtech range resolution for a scan."""
    upgrade_time_sec = 1632182400.0
    if timestamp_sec is not None and float(timestamp_sec) > upgrade_time_sec:
        return 0.04381
    return 0.0596


def navtech_radar_target_index(
    image,
    range_m,
    azimuth_rad,
    timestamp_sec=None,
    range_resolution_m=None,
    encoder_size=5600,
    metadata_columns=11,
):
    """Map a Radar-frame polar target to its nearest native Navtech image cell."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for radar image conversion") from exc

    array = np.asarray(image)
    if array.ndim == 3:
        array = array[:, :, 0]
    if array.ndim != 2 or array.shape[1] <= int(metadata_columns):
        raise ValueError("radar image must contain native Navtech metadata and FFT data")
    if int(metadata_columns) < 11:
        raise ValueError("Navtech metadata requires at least 11 columns")
    if float(range_m) < 0.0:
        return None

    resolution = (
        boreas_radar_resolution(timestamp_sec)
        if range_resolution_m is None or float(range_resolution_m) <= 0.0
        else float(range_resolution_m)
    )
    range_bin = int(round((float(range_m) - resolution / 2.0) / resolution))
    fft_width = array.shape[1] - int(metadata_columns)
    if range_bin < 0 or range_bin >= fft_width:
        return None

    encoder_counts = (
        array[:, 8].astype(np.uint16)
        | (array[:, 9].astype(np.uint16) << np.uint16(8))
    )
    azimuths = (
        encoder_counts.astype(np.float64)
        / float(encoder_size)
        * 2.0
        * np.pi
    )
    valid_rows = np.flatnonzero(array[:, 10] == 255)
    if valid_rows.size == 0:
        return None
    target = float(azimuth_rad) % (2.0 * np.pi)
    delta = np.arctan2(
        np.sin(azimuths[valid_rows] - target),
        np.cos(azimuths[valid_rows] - target),
    )
    row_index = int(valid_rows[int(np.argmin(np.abs(delta)))])
    return row_index, int(metadata_columns) + range_bin, range_bin, resolution


def navtech_radar_patch_support(
    image,
    azimuth_row,
    range_column,
    range_half_window,
    azimuth_half_window,
    frame_reference,
    local_percentile=95.0,
    contrast_scale=1.5,
    intensity_weight=0.75,
    metadata_columns=11,
):
    """Compute robust local intensity and contrast support in a Navtech FFT image."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for radar image conversion") from exc

    array = np.asarray(image)
    if array.ndim == 3:
        array = array[:, :, 0]
    if array.ndim != 2 or array.shape[1] <= int(metadata_columns):
        raise ValueError("radar image must contain native Navtech FFT data")
    height, width = array.shape
    rows = [
        (int(azimuth_row) + offset) % height
        for offset in range(
            -max(0, int(azimuth_half_window)),
            max(0, int(azimuth_half_window)) + 1,
        )
    ]
    c0 = max(int(metadata_columns), int(range_column) - max(0, int(range_half_window)))
    c1 = min(width, int(range_column) + max(0, int(range_half_window)) + 1)
    if c1 <= c0:
        return 0.0, 0.0, 0.0
    patch = array[rows, c0:c1].astype(np.float32) / 255.0
    if patch.size == 0:
        return 0.0, 0.0, 0.0

    percentile = float(np.clip(local_percentile, 0.0, 100.0))
    local_level = float(np.percentile(patch, percentile))
    local_median = float(np.median(patch))
    reference = max(float(frame_reference), 1.0e-6)
    intensity = float(np.clip(local_level / reference, 0.0, 1.0))
    contrast = float(
        np.clip(
            (local_level - local_median)
            / reference
            * max(0.0, float(contrast_scale)),
            0.0,
            1.0,
        )
    )
    weight = float(np.clip(intensity_weight, 0.0, 1.0))
    support = float(np.clip(weight * intensity + (1.0 - weight) * contrast, 0.0, 1.0))
    return intensity, contrast, support


def navtech_radar_image_to_points(
    image,
    timestamp_sec=None,
    range_resolution_m=None,
    min_range_m=2.5,
    max_range_m=None,
    intensity_percentile=99.5,
    max_points=20000,
    encoder_size=5600,
    metadata_columns=11,
):
    """Decode a native Boreas Navtech polar PNG into Radar-frame xyzi points.

    Each image row is one azimuth scan. Columns 0:11 store the scan timestamp,
    encoder count, and validity flag; the remaining columns are range bins.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for radar image conversion") from exc

    array = np.asarray(image)
    if array.ndim == 3:
        array = array[:, :, 0]
    if array.ndim != 2:
        raise ValueError("radar image must be a 2D or 3D array")
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)
    if array.shape[1] <= int(metadata_columns):
        raise ValueError("radar image does not contain Navtech FFT columns")
    if int(metadata_columns) < 11:
        raise ValueError("Navtech metadata requires at least 11 columns")
    if int(encoder_size) <= 0:
        raise ValueError("encoder_size must be positive")

    encoder_counts = (
        array[:, 8].astype(np.uint16)
        | (array[:, 9].astype(np.uint16) << np.uint16(8))
    )
    azimuths = (
        encoder_counts.astype(np.float32)
        / float(encoder_size)
        * 2.0
        * np.pi
    )
    valid = array[:, 10] == 255
    fft_data = array[:, int(metadata_columns):].astype(np.float32)
    if not np.any(valid) or fft_data.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    resolution = (
        boreas_radar_resolution(timestamp_sec)
        if range_resolution_m is None or float(range_resolution_m) <= 0.0
        else float(range_resolution_m)
    )
    min_bin = max(0, int(round(float(min_range_m) / resolution)))
    max_bin = fft_data.shape[1]
    if max_range_m is not None:
        max_bin = min(max_bin, int(np.floor(float(max_range_m) / resolution)) + 1)
    if max_bin <= min_bin:
        return np.zeros((0, 4), dtype=np.float32)

    search = fft_data[valid, min_bin:max_bin]
    if search.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    percentile = min(100.0, max(0.0, float(intensity_percentile)))
    threshold = float(np.percentile(search, percentile))
    valid_rows = np.flatnonzero(valid)
    local_candidates = np.argwhere((search >= threshold) & (search > 0.0))
    if local_candidates.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    row_indices = valid_rows[local_candidates[:, 0]]
    range_bins = local_candidates[:, 1] + min_bin
    intensities = fft_data[row_indices, range_bins]
    if max_points and len(row_indices) > int(max_points):
        keep_count = int(max_points)
        keep = np.argpartition(intensities, -keep_count)[-keep_count:]
        row_indices = row_indices[keep]
        range_bins = range_bins[keep]
        intensities = intensities[keep]

    range_m = (range_bins.astype(np.float32) + 0.5) * resolution
    azimuth = azimuths[row_indices]
    x = range_m * np.cos(azimuth)
    y = range_m * np.sin(azimuth)
    z = np.zeros_like(x)
    intensity = intensities.astype(np.float32) / 255.0
    return np.column_stack((x, y, z, intensity)).astype(np.float32)


def summarize_pose_csv(path):
    pose_path = Path(path)
    if not pose_path.exists() or pose_path.stat().st_size == 0:
        return {"path": str(pose_path), "exists": pose_path.exists(), "rows": 0, "available": False}
    rows = 0
    skipped_rows = 0
    first_time = None
    last_time = None
    mins = {"easting": None, "northing": None, "altitude": None}
    maxs = {"easting": None, "northing": None, "altitude": None}
    with pose_path.open("r", encoding="utf-8", errors="replace") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if not row:
                continue
            required = ("GPSTime", "easting", "northing", "altitude")
            if any(row.get(key) in (None, "") for key in required):
                skipped_rows += 1
                continue
            rows += 1
            timestamp = float(row["GPSTime"])
            first_time = timestamp if first_time is None else first_time
            last_time = timestamp
            for key in mins:
                value = float(row[key])
                mins[key] = value if mins[key] is None else min(mins[key], value)
                maxs[key] = value if maxs[key] is None else max(maxs[key], value)
    duration_s = (last_time - first_time) / 1e6 if first_time is not None and last_time is not None else 0.0
    return {
        "path": str(pose_path),
        "exists": True,
        "rows": rows,
        "skipped_rows": skipped_rows,
        "available": rows > 0,
        "start_gps_time_us": first_time,
        "end_gps_time_us": last_time,
        "duration_s": duration_s,
        "easting_range_m": [mins["easting"], maxs["easting"]],
        "northing_range_m": [mins["northing"], maxs["northing"]],
        "altitude_range_m": [mins["altitude"], maxs["altitude"]],
    }


def read_pose_rows(path, start_gps_time_us=None, end_gps_time_us=None, max_rows=None):
    pose_path = Path(path)
    if not pose_path.exists() or pose_path.stat().st_size == 0:
        return []
    rows = []
    with pose_path.open("r", encoding="utf-8", errors="replace") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            required = ("GPSTime", "easting", "northing", "altitude")
            if not row or any(row.get(key) in (None, "") for key in required):
                continue
            gps_time_us = float(row["GPSTime"])
            if start_gps_time_us is not None and gps_time_us < start_gps_time_us:
                continue
            if end_gps_time_us is not None and gps_time_us > end_gps_time_us:
                continue
            rows.append(
                {
                    "gps_time_us": gps_time_us,
                    "easting": float(row["easting"]),
                    "northing": float(row["northing"]),
                    "altitude": float(row["altitude"]),
                    "roll": float(row.get("roll") or 0.0),
                    "pitch": float(row.get("pitch") or 0.0),
                    "heading": float(row.get("heading") or 0.0),
                }
            )
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def nearest_pose(rows, gps_time_us):
    if not rows:
        return None
    best = min(rows, key=lambda row: abs(row["gps_time_us"] - gps_time_us))
    return best


def pose_position_error_m(reference, candidate):
    if reference is None or candidate is None:
        return None
    dx = candidate["easting"] - reference["easting"]
    dy = candidate["northing"] - reference["northing"]
    dz = candidate["altitude"] - reference["altitude"]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def boreas_pose_sync_rows(sequence_root, step=10, max_samples=200, max_offset_s=0.05):
    root = Path(sequence_root).expanduser().resolve()
    smoke = summarize_boreas_smoke(root)
    overlap = smoke["pose_time_overlap"]
    if not overlap["available"]:
        return [], {
            "sequence_id": root.name,
            "available": False,
            "reason": "no_pose_time_overlap",
            "rows": 0,
        }
    start_us = overlap["start_gps_time_us"]
    end_us = overlap["end_gps_time_us"]
    camera_rows = read_pose_rows(root / "applanix/camera_poses.csv", start_us, end_us)
    lidar_rows = read_pose_rows(root / "applanix/lidar_poses.csv", start_us, end_us)
    radar_rows = read_pose_rows(root / "applanix/radar_poses.csv", start_us, end_us)
    reference_rows = lidar_rows or camera_rows or radar_rows
    sampled_reference = reference_rows[:: max(1, int(step))]
    if max_samples is not None:
        sampled_reference = sampled_reference[: int(max_samples)]
    rows = []
    max_offset_us = float(max_offset_s) * 1e6
    for reference in sampled_reference:
        gps_time_us = reference["gps_time_us"]
        camera = nearest_pose(camera_rows, gps_time_us)
        lidar = nearest_pose(lidar_rows, gps_time_us)
        radar = nearest_pose(radar_rows, gps_time_us)
        camera_offset_us = abs(camera["gps_time_us"] - gps_time_us) if camera else None
        lidar_offset_us = abs(lidar["gps_time_us"] - gps_time_us) if lidar else None
        radar_offset_us = abs(radar["gps_time_us"] - gps_time_us) if radar else None
        if any(offset is None or offset > max_offset_us for offset in (camera_offset_us, lidar_offset_us, radar_offset_us)):
            continue
        rows.append(
            {
                "gps_time_us": gps_time_us,
                "gps_time_s": gps_time_us / 1e6,
                "camera_offset_s": camera_offset_us / 1e6,
                "lidar_offset_s": lidar_offset_us / 1e6,
                "radar_offset_s": radar_offset_us / 1e6,
                "camera_lidar_position_error_m": pose_position_error_m(lidar, camera),
                "radar_lidar_position_error_m": pose_position_error_m(lidar, radar),
                "camera_easting": camera["easting"],
                "camera_northing": camera["northing"],
                "camera_altitude": camera["altitude"],
                "lidar_easting": lidar["easting"],
                "lidar_northing": lidar["northing"],
                "lidar_altitude": lidar["altitude"],
                "radar_easting": radar["easting"],
                "radar_northing": radar["northing"],
                "radar_altitude": radar["altitude"],
            }
        )
    summary = summarize_pose_sync_rows(root.name, rows, overlap, step, max_offset_s)
    return rows, summary


def summarize_pose_sync_rows(sequence_id, rows, overlap, step, max_offset_s):
    def max_value(key):
        values = [row[key] for row in rows if row.get(key) is not None]
        return max(values) if values else None

    def mean_value(key):
        values = [row[key] for row in rows if row.get(key) is not None]
        return sum(values) / len(values) if values else None

    return {
        "sequence_id": sequence_id,
        "available": bool(rows),
        "rows": len(rows),
        "source_overlap_duration_s": overlap.get("duration_s", 0.0),
        "step": int(step),
        "max_allowed_offset_s": float(max_offset_s),
        "start_gps_time_us": rows[0]["gps_time_us"] if rows else None,
        "end_gps_time_us": rows[-1]["gps_time_us"] if rows else None,
        "duration_s": (rows[-1]["gps_time_us"] - rows[0]["gps_time_us"]) / 1e6 if len(rows) > 1 else 0.0,
        "max_camera_offset_s": max_value("camera_offset_s"),
        "max_lidar_offset_s": max_value("lidar_offset_s"),
        "max_radar_offset_s": max_value("radar_offset_s"),
        "mean_camera_lidar_position_error_m": mean_value("camera_lidar_position_error_m"),
        "mean_radar_lidar_position_error_m": mean_value("radar_lidar_position_error_m"),
        "max_camera_lidar_position_error_m": max_value("camera_lidar_position_error_m"),
        "max_radar_lidar_position_error_m": max_value("radar_lidar_position_error_m"),
    }


def write_pose_sync_csv(rows, output_path):
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "gps_time_us",
        "gps_time_s",
        "camera_offset_s",
        "lidar_offset_s",
        "radar_offset_s",
        "camera_lidar_position_error_m",
        "radar_lidar_position_error_m",
        "camera_easting",
        "camera_northing",
        "camera_altitude",
        "lidar_easting",
        "lidar_northing",
        "lidar_altitude",
        "radar_easting",
        "radar_northing",
        "radar_altitude",
    )
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output


def summarize_time_map_csv(path):
    time_path = Path(path)
    if not time_path.exists() or time_path.stat().st_size == 0:
        return {"path": str(time_path), "exists": time_path.exists(), "rows": 0, "available": False}
    rows = 0
    skipped_rows = 0
    first_ros = None
    last_ros = None
    first_gps = None
    last_gps = None
    with time_path.open("r", encoding="utf-8", errors="replace") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if not row:
                continue
            if row.get("ROS Time") in (None, "") or row.get("GPS Time") in (None, ""):
                skipped_rows += 1
                continue
            rows += 1
            ros_time = float(row["ROS Time"])
            gps_time = float(row["GPS Time"])
            first_ros = ros_time if first_ros is None else first_ros
            first_gps = gps_time if first_gps is None else first_gps
            last_ros = ros_time
            last_gps = gps_time
    duration_s = (last_gps - first_gps) if first_gps is not None and last_gps is not None else 0.0
    return {
        "path": str(time_path),
        "exists": True,
        "rows": rows,
        "skipped_rows": skipped_rows,
        "available": rows > 0,
        "start_ros_time_ns": first_ros,
        "end_ros_time_ns": last_ros,
        "start_gps_time_s": first_gps,
        "end_gps_time_s": last_gps,
        "duration_s": duration_s,
    }


def pose_time_overlap(poses):
    available = {key: value for key, value in poses.items() if value.get("available")}
    if not available:
        return {"available": False, "streams": [], "duration_s": 0.0}
    start_us = max(value["start_gps_time_us"] for value in available.values())
    end_us = min(value["end_gps_time_us"] for value in available.values())
    return {
        "available": end_us >= start_us,
        "streams": sorted(available.keys()),
        "start_gps_time_us": start_us,
        "end_gps_time_us": end_us,
        "duration_s": max(0.0, (end_us - start_us) / 1e6),
    }


def summarize_boreas_smoke(sequence_root):
    root = Path(sequence_root).expanduser().resolve()
    calibration_files = (
        "calib/P_camera.txt",
        "calib/T_applanix_lidar.txt",
        "calib/T_camera_lidar.txt",
        "calib/T_radar_lidar.txt",
    )
    calibrations = {}
    for relative in calibration_files:
        path = root / relative
        matrix = read_matrix_txt(path) if path.exists() else []
        calibrations[relative] = {
            "path": str(path),
            "exists": path.exists(),
            "rows": len(matrix),
            "cols": len(matrix[0]) if matrix else 0,
            "matrix": matrix,
        }
    poses = {
        "camera": summarize_pose_csv(root / "applanix/camera_poses.csv"),
        "lidar": summarize_pose_csv(root / "applanix/lidar_poses.csv"),
        "radar": summarize_pose_csv(root / "applanix/radar_poses.csv"),
    }
    time_map = summarize_time_map_csv(root / "applanix/ros_and_gps_time.csv")
    return {
        "sequence_id": root.name,
        "root": str(root),
        "calibration_ok": all(item["exists"] and item["rows"] in (3, 4) and item["cols"] == 4 for item in calibrations.values()),
        "available_pose_streams": sorted(key for key, value in poses.items() if value["available"]),
        "calibrations": calibrations,
        "poses": poses,
        "pose_time_overlap": pose_time_overlap(poses),
        "time_map": time_map,
    }


def inspect_boreas_sequence(root, sequence_id, expected_sizes=None):
    root_path = Path(root).expanduser().resolve()
    sequence_root = root_path / sequence_id
    expected_sizes = expected_sizes or {}
    files = {}
    for relative in BOREAS_METADATA_FILES:
        path = sequence_root / relative
        local_size = path.stat().st_size if path.exists() else 0
        expected_size = expected_sizes.get(relative)
        complete = path.exists() and (expected_size is None or int(expected_size) == int(local_size))
        entry = {
            "relative_path": relative,
            "path": str(path),
            "exists": path.exists(),
            "local_size": local_size,
            "expected_size": expected_size,
            "complete": complete,
        }
        if path.suffix == ".csv" and complete:
            entry["rows"] = count_csv_rows(path)
        files[relative] = entry
    calib_ok = all(files[relative]["complete"] for relative in BOREAS_METADATA_FILES if relative.startswith("calib/"))
    pose_ok = all(files[relative]["complete"] for relative in BOREAS_METADATA_FILES if relative.startswith("applanix/"))
    return {
        "sequence_id": sequence_id,
        "root": str(sequence_root),
        "calib_ok": calib_ok,
        "pose_ok": pose_ok,
        "metadata_ok": calib_ok and pose_ok,
        "files": files,
    }


def inspect_boreas_root(root, expected_manifest=None):
    root_path = Path(root).expanduser().resolve()
    expected = expected_manifest or {}
    sequences = []
    for path in sorted(root_path.glob("boreas-*")):
        if path.is_dir():
            sequence_expected = (expected.get("sequences") or {}).get(path.name, {})
            sequences.append(inspect_boreas_sequence(root_path, path.name, expected_sizes=sequence_expected))
    return {
        "dataset": "boreas",
        "root": str(root_path),
        "sequence_count": len(sequences),
        "sequences": sequences,
    }


def write_json(data, output_path):
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return output
