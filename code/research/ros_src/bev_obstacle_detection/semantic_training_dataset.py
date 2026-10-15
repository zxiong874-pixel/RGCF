"""Utilities for onboarding semantic-mask training datasets."""

from pathlib import Path

import numpy as np


DDOS_CLASS_MAPPING = {
    "ultra_thin": 255,
    "thin_structures": 240,
    "small_mesh": 220,
    "large_mesh": 200,
    "trees": 180,
    "buildings": 160,
    "vehicles": 140,
    "animals": 100,
    "other": 80,
}


def parse_ddos_label_values(text):
    if text is None:
        return tuple(sorted(DDOS_CLASS_MAPPING.values()))
    normalized_text = str(text).strip().lower()
    if normalized_text in ("", "all", "nonzero"):
        return tuple(sorted(DDOS_CLASS_MAPPING.values()))
    values = []
    for item in normalized_text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        if stripped in DDOS_CLASS_MAPPING:
            values.append(DDOS_CLASS_MAPPING[stripped])
        else:
            values.append(int(stripped))
    if not values:
        raise ValueError("expected at least one DDOS label value")
    return tuple(values)


def binary_mask_from_label_image(label_image, obstacle_values=None):
    labels = np.asarray(label_image)
    if labels.ndim == 3:
        labels = labels[:, :, 0]
    if obstacle_values in (None, (), "nonzero", "all"):
        return np.where(labels > 0, 255, 0).astype(np.uint8)
    values = np.asarray(tuple(obstacle_values), dtype=labels.dtype)
    return np.where(np.isin(labels, values), 255, 0).astype(np.uint8)


def discover_ddos_samples(dataset_root, splits=("train", "validation"), environments=()):
    root = Path(dataset_root).expanduser().resolve()
    data_root = root / "data" if (root / "data").is_dir() else root
    selected_envs = set(environments or ())
    samples = []
    for split in splits:
        split_root = data_root / split
        if not split_root.is_dir():
            continue
        for environment_root in sorted(path for path in split_root.iterdir() if path.is_dir()):
            environment = environment_root.name
            if selected_envs and environment not in selected_envs:
                continue
            for flight_root in sorted(path for path in environment_root.iterdir() if path.is_dir()):
                image_root = flight_root / "image"
                segmentation_root = flight_root / "segmentation"
                depth_root = flight_root / "depth"
                if not image_root.is_dir() or not segmentation_root.is_dir():
                    continue
                for image_path in sorted(image_root.glob("*.png")):
                    mask_path = segmentation_root / image_path.name
                    if not mask_path.exists():
                        continue
                    depth_path = depth_root / image_path.name
                    samples.append(
                        {
                            "split": split,
                            "environment": environment,
                            "flight": flight_root.name,
                            "frame": image_path.stem,
                            "image_path": image_path,
                            "mask_path": mask_path,
                            "depth_path": depth_path if depth_path.exists() else None,
                        }
                    )
    return samples


def mask_to_yolo_segmentation_rows(binary_mask, min_area_px=20, simplify_epsilon=1.5):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to export YOLO segmentation labels") from exc

    mask = np.asarray(binary_mask, dtype=np.uint8)
    height, width = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rows = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < float(min_area_px):
            continue
        epsilon = float(simplify_epsilon)
        if epsilon > 0:
            contour = cv2.approxPolyDP(contour, epsilon, closed=True)
        points = contour.reshape((-1, 2))
        if points.shape[0] < 3:
            continue
        coords = []
        for x, y in points:
            coords.append(max(0.0, min(1.0, float(x) / float(max(1, width - 1)))))
            coords.append(max(0.0, min(1.0, float(y) / float(max(1, height - 1)))))
        rows.append("0 " + " ".join("{:.6f}".format(value) for value in coords))
    return rows


def relative_sample_id(sample):
    return "{split}/{environment}/{flight}/{frame}".format(**sample)
