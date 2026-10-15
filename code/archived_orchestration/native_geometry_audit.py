#!/usr/bin/env python3
"""Audit native SDK objects on CPU; export geometry without fitting any model."""
import argparse
import csv
import hashlib
import inspect
import json
import math
import pickle
from pathlib import Path
import subprocess
import sys

import numpy as np
import evaluate_boreas_detection_csv as evaluator


TOLERANCE = 0.001
MIN_IOU = 0.9999
CLASSES = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
           "motorcycle", "bicycle", "pedestrian", "traffic_cone"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields=None):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def footprint(box):
    return np.asarray(evaluator.rect_corners(dict(zip(
        ["center_x", "center_y", "size_x", "size_y", "yaw"],
        [box[0], box[1], box[3], box[4], box[6]]))), dtype=float)


def ordered(points):
    points = np.unique(points, axis=0)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(-angles)].tolist()


def compare(canonical, native):
    dist = np.linalg.norm(canonical[:, None] - native[None], axis=-1)
    error = float(max(dist.min(axis=0).max(), dist.min(axis=1).max()))
    a, b = ordered(canonical), ordered(native)
    aa, ab = evaluator.polygon_area(a), evaluator.polygon_area(b)
    inter = evaluator.polygon_area(evaluator.polygon_clip(a, b))
    union = aa + ab - inter
    iou = inter / union if union > 0 else None
    return error, iou


def canonicalize(box_object):
    if not hasattr(type(box_object), "corners"):
        raise TypeError("Native corners required; plain CSV/tensor is not evidence")
    boxes = array(box_object.tensor).copy()
    if len(boxes) == 0:
        return boxes, np.empty((0, 8, 3), dtype=float)
    corners = array(box_object.corners)
    if len(boxes):
        edge = corners[:, 4, :2] - corners[:, 0, :2]
        boxes[:, 6] = np.arctan2(edge[:, 1], edge[:, 0])
    return boxes, corners


def summarize(records, error_key="canonical_error_m", iou_key="canonical_iou"):
    values = np.asarray([r[error_key] for r in records], dtype=float)
    ious = [r[iou_key] for r in records if r[iou_key] is not None]
    finite = np.isfinite(values)
    return dict(boxes=len(records), max_error_m=float(values.max()) if len(values) else 0,
                median_error_m=float(np.median(values)) if len(values) else 0,
                p95_error_m=float(np.quantile(values, .95)) if len(values) else 0,
                nonfinite_errors=int((~finite).sum()),
                over_tolerance=int((values > TOLERANCE).sum()),
                minimum_polygon_iou=min(ious) if ious else None,
                iou_below_minimum=sum(v < MIN_IOU for v in ious),
                degenerate_footprints=len(records) - len(ious))


def synthetic(box_class, out):
    angles = [0, math.pi / 6, -math.pi / 6, math.pi / 4, -math.pi / 4,
              math.pi / 2, -math.pi / 2, math.pi - 1e-6, -math.pi + 1e-6]
    angles += np.random.RandomState(20260909).uniform(-math.pi, math.pi, 100).tolist()
    records = []
    for origin in [(0.5, 0.5, 0), (0.5, 0.5, 0.5), (0, 0, 0)]:
        for dims in [(4.7, 1.8, 1.6), (1.8, 4.7, 1.6)]:
            for angle in angles:
                for operation in ["identity", "translate", "rotate"]:
                    obj = box_class(np.array([[51.2, -51.2, -.8, *dims, angle]],
                                             dtype=np.float32), box_dim=7, origin=origin)
                    if operation == "translate":
                        obj.translate(np.array([-102.4, 51.2, 1.3], dtype=np.float32))
                    elif operation == "rotate":
                        obj.rotate(0.37)
                    boxes, native = canonicalize(obj)
                    error, iou = compare(footprint(boxes[0]), native[0, :, :2])
                    records.append(dict(origin=str(origin), dims=str(dims), angle=angle,
                                        operation=operation, canonical_error_m=error,
                                        canonical_iou=iou))
    empty = box_class(np.empty((0, 7), dtype=np.float32), box_dim=7)
    try:
        native_empty = list(empty.corners.shape)
    except AssertionError:
        native_empty = "SDK asserts on empty box corners; no footprint is evaluated"
    boxes, corners = canonicalize(empty)
    assert boxes.shape == (0, 7) and corners.shape == (0, 8, 3)
    write_csv(out / "synthetic_geometry_cases.csv", records)
    result = summarize(records)
    result.update(seed=20260909, empty_shape=list(corners.shape), native_empty_behavior=native_empty,
                  boundary_centers="(+51.2,-51.2), translation crosses x boundary",
                  operations=["identity", "translate", "native SDK rotate(0.37)"],
                  tolerance_m=TOLERANCE, minimum_iou=MIN_IOU)
    dump(out / "synthetic_geometry_summary.json", result)
    assert not result["over_tolerance"] and not result["iou_below_minimum"]
    return result


def environment(box_class, repo):
    versions = {}
    for name in ["torch", "mmcv", "mmdet", "mmdet3d", "numpy"]:
        module = __import__(name)
        versions[name] = dict(version=getattr(module, "__version__", "unknown"),
                              file=getattr(module, "__file__", "unknown"))
    source_path = Path(inspect.getfile(box_class))
    def git(*args):
        p = subprocess.run(["git", "-C", str(repo), *args], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, universal_newlines=True)
        return dict(exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)
    return dict(python=sys.version, executable=sys.executable, packages=versions,
                repo=str(repo), commit=git("rev-parse", "HEAD"),
                working_changes=git("diff", "--", "mmdet3d/core/bbox/structures"),
                box_class=box_class.__module__ + "." + box_class.__qualname__,
                class_file=str(source_path), class_file_sha256=sha(source_path),
                init_signature=str(inspect.signature(box_class.__init__)),
                init_source=inspect.getsource(box_class.__init__),
                corners_source=inspect.getsource(box_class.corners.fget),
                coordinate="SDK LiDAR frame; x/y unchanged; distances in input metre units",
                dimensions="tensor[3:6] = native local X,Y,Z extents",
                canonical_yaw="atan2(native_corner4_y-native_corner0_y, native_corner4_x-native_corner0_x)",
                evaluator_sha256=sha(evaluator.__file__), audit_script_sha256=sha(__file__))


def audit_job(job, output, box_class):
    out = output / job["name"]
    out.mkdir(parents=True, exist_ok=False)
    with open(job["raw"], "rb") as f:
        raw = pickle.load(f)
    with open(job["info"], "rb") as f:
        payload = pickle.load(f)
        infos = payload["infos"] if isinstance(payload, dict) else payload
    assert len(raw) == len(infos), (len(raw), len(infos))
    records, prediction, samples, frames = [], [], [], []
    sample_indices = {0, len(raw) // 2, len(raw) - 1}
    failure_sample_count = 0
    for frame_index, (info, result) in enumerate(zip(infos, raw)):
        result = result.get("pts_bbox", result)
        obj = result.get("boxes_3d", result.get("boxes"))
        assert isinstance(obj, box_class), type(obj)
        boxes, corners = canonicalize(obj)
        original = array(obj.tensor)
        scores = array(result.get("scores_3d", result.get("scores")))
        labels = array(result.get("labels_3d", result.get("labels")))
        token = str(info["token"])
        frames.append(dict(frame_index=frame_index, frame_token=token, raw_boxes=len(boxes)))
        retained = 0
        for ordinal, (box, old, native, score, label) in enumerate(zip(boxes, original, corners, scores, labels)):
            polygon = footprint(box)
            ce, ci = compare(polygon, native[:, :2])
            le, li = compare(footprint(old), native[:, :2])
            if float(score) >= .10:
                retained += 1
            row = dict(frame_index=frame_index, frame_token=token, raw_box_index=ordinal,
                       retained_obstacle_id=retained if float(score) >= .1 else "",
                       old_yaw=float(old[6]), canonical_yaw=float(box[6]),
                       center_x=float(box[0]), center_y=float(box[1]),
                       size_x=float(box[3]), size_y=float(box[4]), score=float(score),
                       class_id=int(label), canonical_error_m=ce, canonical_iou=ci,
                       legacy_error_m=le, legacy_iou=li)
            records.append(row)
            if float(score) >= .10:
                prediction.append(dict(timestamp="%.6f" % (int(token) / 1e6),
                    method=job["name"], frame_id="boreas_lidar", obstacle_id=retained,
                    center_x="%.6f" % box[0], center_y="%.6f" % box[1], center_z="%.6f" % box[2],
                    size_x="%.6f" % box[3], size_y="%.6f" % box[4], size_z="%.6f" % box[5],
                    yaw="%.9f" % box[6], score="%.9f" % score, class_id=int(label),
                    class_name=CLASSES[int(label)], sample_token=token, source=job["name"],
                    raw_box_index=ordinal, tensor_yaw="%.9f" % old[6]))
            failure_sample = le > TOLERANCE and failure_sample_count < 10
            if frame_index in sample_indices or failure_sample:
                samples.append(dict(**row, native_xy=native[:, :2].tolist(),
                                    canonical_xy=polygon.tolist(), legacy_xy=footprint(old).tolist()))
            if failure_sample:
                failure_sample_count += 1
        if frame_index % 100 == 0:
            print(job["name"], frame_index, len(records), flush=True)
    write_csv(out / "native_geometry_audit.csv", records)
    write_csv(out / "canonical_detections.csv", prediction)
    write_csv(out / "frame_manifest.csv", frames)
    dump(out / "fixed_geometry_samples.json", samples)
    summary = dict(name=job["name"], raw_path=job["raw"], raw_sha256=sha(job["raw"]),
                   info_path=job["info"], info_sha256=sha(job["info"]), frames=len(raw),
                   unique_tokens=len({r["frame_token"] for r in frames}),
                   canonical=summarize(records), legacy=summarize(records, "legacy_error_m", "legacy_iou"),
                   score_threshold=.10, class_filter="none; preserved raw labels",
                   roi_filter="none; coordinate conversion precedes any downstream filtering",
                   fixed_sample_frames=sorted(sample_indices),
                   failure_sample_rule="first 10 raw boxes in file order with legacy corner error > 0.001m",
                   polygon_iou_implementation="Frozen project Sutherland-Hodgman evaluator; clockwise convex polygons",
                   tolerance_m=TOLERANCE, minimum_iou=MIN_IOU,
                   all_boxes_including_below_score_threshold_checked=True)
    summary["status"] = "pass" if not any(summary["canonical"][k] for k in
        ["nonfinite_errors", "over_tolerance", "iou_below_minimum", "degenerate_footprints"]) else "fail"
    dump(out / "native_geometry_audit.json", summary)
    dump(out / "output_sha256.json", {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs-json", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    import torch
    torch.set_num_threads(2)
    # Pickles can contain CUDA storages; deserialize on CPU without changing bytes.
    import io
    torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu")
    from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "environment_lock.json", environment(LiDARInstance3DBoxes, args.repo))
    synthetic(LiDARInstance3DBoxes, out)
    summaries = []
    for job in json.loads(Path(args.jobs_json).read_text()):
        summaries.append(audit_job(job, out, LiDARInstance3DBoxes))
    dump(out / "all_jobs_summary.json", summaries)
    print(json.dumps([dict(name=s["name"], frames=s["frames"], status=s["status"],
                           canonical=s["canonical"], legacy=s["legacy"]) for s in summaries], indent=2), flush=True)
    assert all(s["status"] == "pass" for s in summaries)


if __name__ == "__main__":
    main()
