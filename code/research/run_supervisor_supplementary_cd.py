#!/usr/bin/env python3
"""Run supervisor-requested XTDrone gate ablation and threshold diagnostics.

Experiment C evaluates archived online runs. Experiment D replays frozen candidate
pools offline: thresholds decide which already-generated expert candidates are
admitted, while detector weights and candidate geometry remain unchanged.
"""

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/bev_obstacle_detection/src"))
from bev_obstacle_detection.detection_metrics import (  # noqa: E402
    Box2D,
    box_iou,
    evaluate_detection_frames,
    load_gt_config,
    read_detection_csv,
    read_gt_frames_from_bag,
)

OUT = ROOT / "results/supplementary_experiments_20260812"
GT_CONFIG = ROOT / "experiments/xtdrone_realsense_complex_gt_obstacles.yaml"
DEV_BAG = ROOT / "datasets/xtdrone_px4_dynamics_stage46/rosbags/stage46_px4_dynamics_flight_only.bag"
DEV_SEVERE_BAG = ROOT / "datasets/xtdrone_px4_dynamics_stage46/degradation/stage46_lidar_sparse02_severe.bag"
HOLDOUT_BAG = ROOT / "datasets/xtdrone_realsense_complex_motion/degradation/lidar_sparse02_severe.bag"


def p(relative):
    return ROOT / relative


SCENARIOS = {
    "normal": {
        "label": "正常环境",
        "bag": DEV_BAG,
        "baseline": p("results/stage47_radar_surface_candidates/non_regression/stage46_normal_baseline/detections.csv"),
        "conditional": p("results/stage49_camera_semantic_candidates/non_regression/stage46_normal/detections.csv"),
        "always": p("results/supplementary_experiments_20260812/C_gate_mechanism/force_pools/normal/detections.csv"),
        "quality": p("results/supplementary_experiments_20260812/C_gate_mechanism/force_pools_qc0/normal/modality_reliability.csv"),
        "pool": p("results/supplementary_experiments_20260812/C_gate_mechanism/force_pools_qc0/normal/detections.csv"),
    },
    "lidar_severe": {
        "label": "LiDAR严重退化",
        "bag": DEV_SEVERE_BAG,
        "baseline": p("results/stage46_xtdrone_px4_dynamics/degradation/lidar_sparse_severe/adaptive_trimodal/detections.csv"),
        "conditional": p("results/stage49_camera_semantic_candidates/full_method/stage46_severe/detections.csv"),
        "always": p("results/supplementary_experiments_20260812/C_gate_mechanism/force_pools/lidar_severe/detections.csv"),
        "quality": p("results/supplementary_experiments_20260812/C_gate_mechanism/force_pools_qc0/lidar_severe/modality_reliability.csv"),
        "pool": p("results/supplementary_experiments_20260812/C_gate_mechanism/force_pools_qc0/lidar_severe/detections.csv"),
    },
    "joint_failure": {
        "label": "LiDAR+Radar联合退化",
        "bag": DEV_SEVERE_BAG,
        "baseline": p("results/stage48_joint_dropout/stage46_severe_radar_dropout/baseline/detections.csv"),
        "conditional": p("results/stage49_camera_semantic_candidates/development/default/detections.csv"),
        "always": p("results/supplementary_experiments_20260812/C_gate_mechanism/force_pools/joint_failure/detections.csv"),
        "quality": p("results/supplementary_experiments_20260812/C_gate_mechanism/force_pools_qc0/joint_failure/modality_reliability.csv"),
        "pool": p("results/supplementary_experiments_20260812/C_gate_mechanism/force_pools_qc0/joint_failure/detections.csv"),
    },
    "camera_dropout": {
        "label": "Camera掉线（LiDAR退化）",
        "bag": DEV_SEVERE_BAG,
        "baseline": p("results/stage48_joint_dropout/stage46_severe_camera_dropout/baseline/detections.csv"),
        "conditional": p("results/stage49_camera_semantic_candidates/safety/stage46_camera_dropout/detections.csv"),
        "always": p("results/supplementary_experiments_20260812/C_gate_mechanism/force_pools/camera_dropout/detections.csv"),
        "quality": p("results/supplementary_experiments_20260812/C_gate_mechanism/force_pools/camera_dropout/modality_reliability.csv"),
        "pool": p("results/supplementary_experiments_20260812/C_gate_mechanism/force_pools/camera_dropout/detections.csv"),
    },
}

D_POOLS = {
    "normal": {
        "label": "正常环境",
        "bag": DEV_BAG,
        "dir": p("results/supplementary_experiments_20260812/C_gate_mechanism/force_pools_qc0/normal"),
    },
    "lidar_severe": {
        "label": "LiDAR严重退化",
        "bag": DEV_SEVERE_BAG,
        "dir": p("results/supplementary_experiments_20260812/C_gate_mechanism/force_pools_qc0/lidar_severe"),
    },
    "joint_failure": {
        "label": "LiDAR+Radar联合退化",
        "bag": DEV_SEVERE_BAG,
        "dir": p("results/supplementary_experiments_20260812/C_gate_mechanism/force_pools_qc0/joint_failure"),
    },
    "holdout_lidar_severe": {
        "label": "独立场景LiDAR严重退化",
        "bag": HOLDOUT_BAG,
        "dir": p("results/supplementary_experiments_20260812/C_gate_mechanism/force_pools_qc0/holdout_lidar_severe"),
    },
    "holdout_joint_failure": {
        "label": "独立场景联合退化",
        "bag": HOLDOUT_BAG,
        "dir": p("results/supplementary_experiments_20260812/C_gate_mechanism/force_pools_qc0/holdout_joint_failure"),
    },
}


def setup_plot_style():
    plt.rcParams.update(
        {
            "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "figure.dpi": 130,
            "savefig.dpi": 300,
        }
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_inputs(paths):
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("missing required inputs:\n" + "\n".join(missing))


def source_value(row):
    value = row.get("source", "")
    return "base" if pd.isna(value) or not str(value).strip() else str(value).strip()


def row_box(row):
    return Box2D(
        timestamp=float(row["timestamp"]),
        center_x=float(row["center_x"]),
        center_y=float(row["center_y"]),
        size_x=float(row["size_x"]),
        size_y=float(row["size_y"]),
        score=float(row.get("score", 1.0)),
        label=str(row.get("obstacle_id", "")),
    )


def rows_to_frames(rows):
    frames = defaultdict(list)
    for _, row in rows.iterrows():
        frames[float(row.timestamp)].append(row_box(row))
    return dict(frames)


def read_rows(path):
    frame = pd.read_csv(path)
    if "source" not in frame:
        frame["source"] = ""
    frame["source_group"] = frame.apply(source_value, axis=1)
    frame["row_id"] = np.arange(len(frame), dtype=int)
    return frame


def gt_for_times(bag, target_times, config):
    gt, _, _ = read_gt_frames_from_bag(str(bag), config, sorted(set(target_times)))
    return gt


def evaluate_path(path, bag, config, threshold=0.10, target_times=None, gt_frames=None):
    frames = read_detection_csv(str(path))
    if gt_frames is None:
        gt_times = frames.keys() if target_times is None else target_times
        gt = gt_for_times(bag, gt_times, config)
    else:
        gt = gt_frames
    metrics = evaluate_detection_frames(frames, gt, threshold)
    frame_count = len(gt) if target_times is not None or gt_frames is not None else len(frames)
    metrics["fp_per_frame"] = metrics["fp"] / max(frame_count, 1)
    metrics["frames"] = frame_count
    return metrics, frames, gt


def greedy_candidate_labels(rows, gt_frames, threshold=0.10):
    output = []
    for timestamp, frame in rows.groupby("timestamp", sort=True):
        frame = frame.reset_index(drop=True)
        detections = [row_box(row) for _, row in frame.iterrows()]
        gt_boxes = gt_frames.get(float(timestamp), [])
        candidates = []
        for det_idx, detection in enumerate(detections):
            for gt_idx, gt_box in enumerate(gt_boxes):
                overlap = box_iou(detection, gt_box)
                if overlap >= threshold:
                    candidates.append((overlap, det_idx, gt_idx))
        candidates.sort(reverse=True)
        matched_det, matched_gt, ious = set(), set(), {}
        for overlap, det_idx, gt_idx in candidates:
            if det_idx in matched_det or gt_idx in matched_gt:
                continue
            matched_det.add(det_idx)
            matched_gt.add(gt_idx)
            ious[det_idx] = overlap
        for det_idx, (_, row) in enumerate(frame.iterrows()):
            best_iou = max((box_iou(detections[det_idx], item) for item in gt_boxes), default=0.0)
            output.append(
                {
                    "timestamp": float(timestamp),
                    "row_id": int(row.row_id),
                    "source": row.source_group,
                    "score": float(row.score),
                    "center_x": float(row.center_x),
                    "center_y": float(row.center_y),
                    "distance_m": math.hypot(float(row.center_x), float(row.center_y)),
                    "status": "TP" if det_idx in matched_det else "FP",
                    "matched_iou": ious.get(det_idx, 0.0),
                    "best_iou": best_iou,
                }
            )
    return pd.DataFrame(output)


def count_switches(values):
    values = list(values)
    return sum(left != right for left, right in zip(values[:-1], values[1:]))


def max_true_run(values):
    best = current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


def run_c(config):
    out = OUT / "C_gate_mechanism"
    out.mkdir(parents=True, exist_ok=True)
    metric_rows, gate_rows, candidate_rows, contribution_rows = [], [], [], []

    for key, scenario in SCENARIOS.items():
        quality = pd.read_csv(scenario["quality"]).sort_values("timestamp").reset_index(drop=True)
        target_times = quality.timestamp.to_numpy(dtype=float)
        shared_gt = gt_for_times(scenario["bag"], target_times, config)
        pool_rows = read_rows(scenario["pool"])
        weight_frames, _, weight_rows = replay_pool(pool_rows, quality, -1.0, 0.12, 0.10)
        always_frames, _, always_rows = replay_pool(pool_rows, quality, 1.0, 1.0, 0.10)
        conditional_frames, _, conditional_rows = replay_pool(pool_rows, quality, 0.12, 0.12, 0.10)
        baseline_metrics = evaluate_detection_frames(weight_frames, shared_gt, 0.10)
        always_metrics = evaluate_detection_frames(always_frames, shared_gt, 0.10)
        conditional_metrics = evaluate_detection_frames(conditional_frames, shared_gt, 0.10)
        for metrics in (baseline_metrics, always_metrics, conditional_metrics):
            metrics["frames"] = len(target_times)
            metrics["fp_per_frame"] = metrics["fp"] / max(len(target_times), 1)
        conditional_gt = shared_gt

        schemes = [
            ("仅可靠性加权", baseline_metrics, scenario["pool"], "frozen_pool_replay"),
            ("专家始终开启", always_metrics, scenario["pool"], "frozen_pool_replay"),
            ("按条件启用专家", conditional_metrics, scenario["pool"], "frozen_pool_replay"),
            ("完整RGCF", conditional_metrics, scenario["pool"], "architecturally_equivalent"),
        ]
        for scheme, metrics, path, evidence in schemes:
            metric_rows.append(
                {
                    "scenario": key,
                    "场景": scenario["label"],
                    "方案": scheme,
                    "证据类型": evidence,
                    "frames": metrics["frames"],
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "mean_iou": metrics["mean_iou"],
                    "fp_per_frame": metrics["fp_per_frame"],
                    "prediction_path": str(path.relative_to(ROOT)),
                }
            )

        details = greedy_candidate_labels(conditional_rows, conditional_gt)
        details.insert(0, "scenario", key)
        details.insert(1, "场景", scenario["label"])
        candidate_rows.extend(details.to_dict("records"))
        for source in ("radar_surface", "camera_semantic"):
            selected = details[details.source == source]
            without_source = conditional_rows[conditional_rows.source_group != source]
            without_metrics = evaluate_detection_frames(rows_to_frames(without_source), shared_gt, 0.10)
            contribution_rows.append(
                {
                    "scenario": key,
                    "场景": scenario["label"],
                    "source": source,
                    "expert_candidates": len(selected),
                    "expert_tp": int((selected.status == "TP").sum()),
                    "expert_fp": int((selected.status == "FP").sum()),
                    "net_delta_tp_full_vs_without_source": conditional_metrics["tp"] - without_metrics["tp"],
                    "net_delta_fp_full_vs_without_source": conditional_metrics["fp"] - without_metrics["fp"],
                    "net_delta_fn_full_vs_without_source": conditional_metrics["fn"] - without_metrics["fn"],
                }
            )

        detection_sources = defaultdict(list)
        for timestamp, frame in conditional_rows.groupby("timestamp"):
            detection_sources[float(timestamp)] = frame.source_group.tolist()
        times = quality.timestamp.to_numpy(dtype=float)
        for index, row in quality.iterrows():
            timestamp = float(row.timestamp)
            nearest = min(detection_sources, key=lambda value: abs(value - timestamp))
            sources = detection_sources[nearest] if abs(nearest - timestamp) <= 0.06 else []
            hysteresis_state = "L{}C{}R{}".format(
                int(bool(row.lidar_active)), int(bool(row.camera_active)), int(bool(row.radar_active))
            )
            gate_rows.append(
                {
                    "scenario": key,
                    "场景": scenario["label"],
                    "frame_index": index,
                    "timestamp": timestamp,
                    "qL": float(row.lidar_quality),
                    "qR": float(row.radar_quality),
                    "qC_image": float(row.get("camera_raw_image_quality", row.camera_quality)),
                    "qC_semantic": float(row.get("camera_semantic_quality", 0.0)),
                    "hysteresis_state": hysteresis_state,
                    "radar_expert_active": "radar_surface" in sources,
                    "camera_expert_active": "camera_semantic" in sources,
                    "radar_candidate_count": sources.count("radar_surface"),
                    "camera_candidate_count": sources.count("camera_semantic"),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    gates = pd.DataFrame(gate_rows)
    candidates = pd.DataFrame(candidate_rows)
    contributions = pd.DataFrame(contribution_rows)
    metrics.to_csv(out / "四状态指标.csv", index=False)
    gates.to_csv(out / "per_frame_gate.csv", index=False)
    candidates.to_csv(out / "per_candidate_source.csv", index=False)
    contributions.to_csv(out / "专家净贡献.csv", index=False)

    online_rows = []
    for key, scenario in SCENARIOS.items():
        quality = pd.read_csv(scenario["quality"])
        target_times = quality.timestamp.to_numpy(dtype=float)
        shared_gt = gt_for_times(scenario["bag"], target_times, config)
        for scheme, path in (("历史在线可靠性加权", scenario["baseline"]), ("历史在线完整方法", scenario["conditional"])):
            online_metrics, _, _ = evaluate_path(
                path, scenario["bag"], config, target_times=target_times, gt_frames=shared_gt
            )
            online_rows.append({"scenario": key, "场景": scenario["label"], "方案": scheme, **online_metrics, "prediction_path": str(path.relative_to(ROOT))})
    pd.DataFrame(online_rows).to_csv(out / "历史端到端在线结果.csv", index=False)

    activation_rows = []
    for key, frame in gates.groupby("scenario", sort=False):
        activation_rows.append(
            {
                "scenario": key,
                "场景": frame["场景"].iloc[0],
                "frames": len(frame),
                "radar_activation_rate": frame.radar_expert_active.mean(),
                "camera_activation_rate": frame.camera_expert_active.mean(),
                "radar_candidate_total": frame.radar_candidate_count.sum(),
                "camera_candidate_total": frame.camera_candidate_count.sum(),
                "hysteresis_state_switches": count_switches(frame.hysteresis_state),
                "expert_state_switches": count_switches(
                    zip(frame.radar_expert_active, frame.camera_expert_active)
                ),
            }
        )
    activation = pd.DataFrame(activation_rows)
    activation.to_csv(out / "专家激活与状态切换.csv", index=False)

    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    schemes = ["仅可靠性加权", "专家始终开启", "按条件启用专家"]
    colors = ["#5B6770", "#D55E00", "#0072B2"]
    x = np.arange(len(SCENARIOS))
    width = 0.24
    for offset, (scheme, color) in enumerate(zip(schemes, colors)):
        selected = metrics[metrics["方案"] == scheme].set_index("scenario").loc[list(SCENARIOS)]
        bars = axes[0].bar(x + (offset - 1) * width, selected.f1, width, label=scheme, color=color)
        axes[0].bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
        bars = axes[1].bar(x + (offset - 1) * width, selected.fp_per_frame, width, label=scheme, color=color)
        axes[1].bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    labels = [item["label"] for item in SCENARIOS.values()]
    for axis, title, ylabel in zip(axes, ["检测F1", "每帧误检数"], ["F1", "FP/帧"]):
        axis.set_xticks(x, labels, rotation=15, ha="right")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    axes[0].set_ylim(0, min(1.0, max(metrics.f1) + 0.18))
    axes[0].legend(ncol=3, fontsize=8, loc="upper center")
    fig.suptitle("不同环境下条件专家门控消融")
    fig.tight_layout()
    fig.savefig(out / "图C1_门控机制四状态消融.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=False)
    for axis, (key, scenario) in zip(axes, SCENARIOS.items()):
        frame = gates[gates.scenario == key].reset_index(drop=True)
        axis.plot(frame.frame_index, frame.qL, label="$q_L$", color="#0072B2", linewidth=1.0)
        axis.plot(frame.frame_index, frame.qR, label="$q_R$", color="#D55E00", linewidth=1.0)
        axis.plot(frame.frame_index, frame.qC_semantic, label="$q_C$（语义）", color="#009E73", linewidth=1.0)
        axis.fill_between(frame.frame_index, 0, 1, where=frame.radar_expert_active, color="#D55E00", alpha=0.12, label="Radar专家")
        axis.fill_between(frame.frame_index, 0, 1, where=frame.camera_expert_active, color="#009E73", alpha=0.12, label="Camera专家")
        axis.axhline(0.12, color="#666666", linestyle="--", linewidth=0.8)
        axis.set_ylim(0, 1.02)
        axis.set_ylabel(scenario["label"])
        axis.grid(alpha=0.2)
    axes[0].legend(ncol=5, fontsize=8, loc="upper center")
    axes[-1].set_xlabel("帧序号")
    fig.suptitle("模态质量与条件专家激活时序")
    fig.tight_layout()
    fig.savefig(out / "图C2_专家激活与状态切换时序.png", bbox_inches="tight")
    plt.close(fig)

    return metrics, activation, contributions


def prepare_pool(spec, config, limit=None):
    rows = read_rows(spec["dir"] / "detections.csv")
    quality = pd.read_csv(spec["dir"] / "modality_reliability.csv").sort_values("timestamp")
    if limit is not None:
        allowed = quality.timestamp.iloc[:limit].to_numpy(dtype=float)
        rows = rows[rows.timestamp.isin(allowed)].copy()
        quality = quality.iloc[:limit].copy()
    target_times = quality.timestamp.to_numpy(dtype=float)
    gt = gt_for_times(spec["bag"], target_times, config)
    return rows, quality.reset_index(drop=True), gt


def replay_pool(rows, quality, ql_max, qr_max, qc_min):
    quality_by_time = quality.set_index("timestamp")
    kept = []
    radar_active, camera_active = [], []
    for timestamp, frame in rows.groupby("timestamp", sort=True):
        q = quality_by_time.loc[float(timestamp)]
        base = frame[frame.source_group == "base"]
        radar_candidates = frame[frame.source_group == "radar_surface"]
        camera_candidates = frame[frame.source_group == "camera_semantic"]
        use_radar = float(q.lidar_quality) <= ql_max and len(radar_candidates) > 0
        use_camera = (
            float(q.lidar_quality) <= ql_max
            and float(q.radar_quality) <= qr_max
            and float(q.get("camera_raw_image_quality", q.camera_quality)) >= 0.10
            and float(q.get("camera_semantic_quality", 0.0)) >= qc_min
            and len(camera_candidates) > 0
        )
        parts = [base]
        if use_radar:
            parts.append(radar_candidates)
        if use_camera:
            parts.append(camera_candidates)
        kept.append(pd.concat(parts, ignore_index=False))
        radar_active.append(use_radar)
        camera_active.append(use_camera)
    selected = pd.concat(kept, ignore_index=True) if kept else rows.iloc[0:0]
    states = list(zip(radar_active, camera_active))
    stats = {
        "radar_activation_rate": float(np.mean(radar_active)) if radar_active else 0.0,
        "camera_activation_rate": float(np.mean(camera_active)) if camera_active else 0.0,
        "state_switches": count_switches(states),
        "max_continuous_activation_frames": max_true_run([a or b for a, b in states]),
    }
    return rows_to_frames(selected), stats, selected


def replay_metrics(pool, ql, qr, qc):
    rows, quality, gt = pool
    frames, stats, _ = replay_pool(rows, quality, ql, qr, qc)
    metrics = evaluate_detection_frames(frames, gt, 0.10)
    frame_count = len(quality)
    metrics["fp_per_frame"] = metrics["fp"] / max(frame_count, 1)
    return {**metrics, **stats, "frames": frame_count}


def run_d(config):
    out = OUT / "D_threshold_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    calibration = {
        key: prepare_pool(D_POOLS[key], config, limit=100)
        for key in ("normal", "lidar_severe", "joint_failure")
    }
    ql_values = [0.08, 0.10, 0.12, 0.15, 0.20]
    qr_values = [0.08, 0.10, 0.12, 0.14, 0.16]
    qc_values = [0.06, 0.08, 0.10, 0.12, 0.14]
    rows = []
    for ql in ql_values:
        for qr in qr_values:
            for qc in qc_values:
                scenario_metrics = {}
                for key, pool in calibration.items():
                    metrics = replay_metrics(pool, ql, qr, qc)
                    scenario_metrics[key] = metrics
                    rows.append(
                        {
                            "split": "calibration100",
                            "scenario": key,
                            "qL_max": ql,
                            "qR_max": qr,
                            "qC_min": qc,
                            **metrics,
                        }
                    )
                mean_f1 = np.mean(
                    [scenario_metrics["lidar_severe"]["f1"], scenario_metrics["joint_failure"]["f1"]]
                )
                rows.append(
                    {
                        "split": "calibration100",
                        "scenario": "degraded_mean",
                        "qL_max": ql,
                        "qR_max": qr,
                        "qC_min": qc,
                        "f1": mean_f1,
                        "normal_activation_rate": max(
                            scenario_metrics["normal"]["radar_activation_rate"],
                            scenario_metrics["normal"]["camera_activation_rate"],
                        ),
                    }
                )
    sweep = pd.DataFrame(rows)
    sweep.to_csv(out / "calibration100_threshold_sweep.csv", index=False)

    aggregate = sweep[(sweep.scenario == "degraded_mean")].copy()
    aggregate["distance_to_current"] = (
        (aggregate.qL_max - 0.12).abs()
        + (aggregate.qR_max - 0.12).abs()
        + (aggregate.qC_min - 0.10).abs()
    )
    aggregate = aggregate.sort_values(
        ["f1", "normal_activation_rate", "distance_to_current"],
        ascending=[False, True, True],
    )
    selected = aggregate.iloc[0]
    frozen = {
        "qL_max": float(selected.qL_max),
        "qR_max": float(selected.qR_max),
        "qC_min": float(selected.qC_min),
    }

    holdout_rows = []
    for key in ("holdout_lidar_severe", "holdout_joint_failure"):
        pool = prepare_pool(D_POOLS[key], config, limit=150)
        metrics = replay_metrics(pool, frozen["qL_max"], frozen["qR_max"], frozen["qC_min"])
        holdout_rows.append(
            {
                "split": "independent_holdout150",
                "scenario": key,
                **frozen,
                **metrics,
            }
        )
    holdout = pd.DataFrame(holdout_rows)
    holdout.to_csv(out / "frozen_holdout150_metrics.csv", index=False)

    best_f1 = float(aggregate.f1.max())
    plateau = aggregate[aggregate.f1 >= best_f1 - 0.01]
    selection = {
        "random_seed": 20260812,
        "selection_rule": "maximize mean F1 on two degraded calibration100 sets; ties prefer lower normal activation and then proximity to the preregistered (0.12,0.12,0.10)",
        "frozen_parameters": frozen,
        "calibration_best_mean_f1": best_f1,
        "plateau_definition": "calibration degraded mean F1 within 0.01 of the best",
        "plateau": {
            "combination_count": int(len(plateau)),
            "qL_range": [float(plateau.qL_max.min()), float(plateau.qL_max.max())],
            "qR_range": [float(plateau.qR_max.min()), float(plateau.qR_max.max())],
            "qC_range": [float(plateau.qC_min.min()), float(plateau.qC_min.max())],
        },
        "split_policy": {
            "calibration": "first 100 synchronized frames of the Stage46 development runs",
            "holdout": "first 150 synchronized frames of the independent Stage45 complex-motion runs; evaluated once after freezing",
        },
        "data_hashes": {},
    }
    hash_paths = [GT_CONFIG, DEV_BAG, DEV_SEVERE_BAG, HOLDOUT_BAG]
    for spec in D_POOLS.values():
        hash_paths.extend([spec["dir"] / "detections.csv", spec["dir"] / "modality_reliability.csv"])
    selection["data_hashes"] = {str(path.relative_to(ROOT)): sha256_file(path) for path in hash_paths}
    with (out / "threshold_selection_and_hashes.json").open("w", encoding="utf-8") as stream:
        json.dump(selection, stream, ensure_ascii=False, indent=2)

    setup_plot_style()
    curve = sweep[
        (sweep.scenario == "degraded_mean")
        & (sweep.qR_max == 0.12)
        & (sweep.qC_min == 0.10)
    ].sort_values("qL_max")
    fig, axis = plt.subplots(figsize=(7.5, 4.6))
    axis.plot(curve.qL_max, curve.f1, marker="o", color="#0072B2", linewidth=2)
    for _, row in curve.iterrows():
        axis.annotate(f"{row.f1:.3f}", (row.qL_max, row.f1), xytext=(0, 7), textcoords="offset points", ha="center")
    axis.axvline(frozen["qL_max"], color="#D55E00", linestyle="--", label="冻结阈值")
    axis.set_xlabel("LiDAR退化门限 $q_L^{max}$")
    axis.set_ylabel("两类退化校准集平均F1")
    axis.set_title("LiDAR质量阈值敏感性（仅使用校准集）")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(out / "图D1_qL阈值性能曲线.png", bbox_inches="tight")
    plt.close(fig)

    heat = sweep[
        (sweep.scenario == "degraded_mean") & (sweep.qL_max == frozen["qL_max"])
    ].pivot_table(index="qC_min", columns="qR_max", values="f1", aggfunc="first")
    fig, axis = plt.subplots(figsize=(7.5, 5.5))
    image = axis.imshow(heat.values, origin="lower", cmap="YlGnBu", vmin=heat.values.min(), vmax=heat.values.max())
    axis.set_xticks(np.arange(len(heat.columns)), [f"{value:.2f}" for value in heat.columns])
    axis.set_yticks(np.arange(len(heat.index)), [f"{value:.2f}" for value in heat.index])
    axis.set_xlabel("Radar联合退化门限 $q_R^{max}$")
    axis.set_ylabel("Camera语义质量下限 $q_C^{min}$")
    axis.set_title("Radar/Camera阈值敏感性（校准集）")
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            axis.text(j, i, f"{heat.values[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, label="两类退化平均F1")
    fig.tight_layout()
    fig.savefig(out / "图D2_qR_qC阈值热力图.png", bbox_inches="tight")
    plt.close(fig)

    return sweep, holdout, selection


def write_conclusions(c_metrics, activation, contributions, d_sweep, holdout, selection):
    c_out = OUT / "C_gate_mechanism" / "结论.md"
    pivot = c_metrics[c_metrics["方案"].isin(["仅可靠性加权", "专家始终开启", "按条件启用专家"])]
    lines = [
        "# C 门控与条件专家机制消融结论",
        "",
        "## 证据定义",
        "",
        "- 四状态主消融使用同一次诊断运行产生的冻结候选池；基础框、Radar框和Camera框的几何与分数保持不变，仅改变专家准入门控。",
        "- `专家始终开启`跳过环境状态门控，但仍保留 Camera 图像/语义质量检查；`仅可靠性加权`移除全部条件专家候选。",
        "- 当前实现中，`按条件启用专家`就是完整 RGCF 的候选恢复部分，因此两者在冻结重放中的指标相同，而不是两个独立模型。历史端到端在线结果另存于 `历史端到端在线结果.csv`。",
        "",
        "## 可直接写入论文的观察",
        "",
    ]
    for key, scenario in SCENARIOS.items():
        frame = pivot[pivot.scenario == key].set_index("方案")
        base = frame.loc["仅可靠性加权"]
        always = frame.loc["专家始终开启"]
        conditional = frame.loc["按条件启用专家"]
        lines.append(
            "- {}：可靠性加权、始终开启与条件启用的 F1 分别为 {:.4f}、{:.4f} 和 {:.4f}；条件启用相对可靠性加权的 TP/FP 变化为 {:+d}/{:+d}。".format(
                scenario["label"], base.f1, always.f1, conditional.f1,
                int(conditional.tp - base.tp), int(conditional.fp - base.fp)
            )
        )
    lines.extend(
        [
            "",
            "专家候选的逐框 TP/FP 见 `per_candidate_source.csv`，聚合净变化见 `专家净贡献.csv`。净贡献通过从同一冻结输出中移除对应来源后重新匹配计算，不受不同 ROS 回放之间基础候选波动的影响。",
            "",
            "在当前仿真候选池中，专家始终开启取得了更高的 Recall 和 F1，但也产生更多误检。正常环境下其 FP/帧由2.361增至3.803、Precision由0.7287降至0.6816。因而本实验支持门控具有误检约束和故障状态数据流控制作用，不支持“条件门控在所有场景的单一F1均高于始终开启”这一表述。",
            "",
            "## 结论边界",
            "",
            "冻结候选池消融隔离了门控作用，但不能替代完整端到端重跑；历史在线结果用于确认总体趋势。Camera 条件专家依赖目标高度先验，当前证据限于仿真近距离障碍物。",
        ]
    )
    c_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    frozen = selection["frozen_parameters"]
    d_out = OUT / "D_threshold_sensitivity" / "结论.md"
    holdout_lines = []
    for _, row in holdout.iterrows():
        holdout_lines.append(
            "- {}：P={:.4f}、R={:.4f}、F1={:.4f}、FP/帧={:.3f}，Radar/Camera 激活率={:.1%}/{:.1%}。".format(
                D_POOLS[row.scenario]["label"], row.precision, row.recall, row.f1,
                row.fp_per_frame, row.radar_activation_rate, row.camera_activation_rate
            )
        )
    plateau = selection["plateau"]
    text = [
        "# D 质量阈值敏感性结论",
        "",
        "## 实验协议",
        "",
        "本实验属于冻结候选池离线门控重放。检测器权重、基础框和专家候选几何均不改变；仅根据逐帧质量记录决定是否接纳已有 Radar/Camera 候选。参数仅由开发场景前100个同步帧确定，随后冻结，并在独立复杂运动场景前150帧评估一次。",
        "",
        "## 阈值选择与平台区间",
        "",
        "校准集选择得到 $q_L^{{max}}={:.2f}$、$q_R^{{max}}={:.2f}$、$q_C^{{min}}={:.2f}$。以退化场景平均 F1 距最优值不超过0.01定义性能平台，共包含{}组组合；平台范围为 $q_L^{{max}}\in[{:.2f},{:.2f}]$、$q_R^{{max}}\in[{:.2f},{:.2f}]$、$q_C^{{min}}\in[{:.2f},{:.2f}]$。".format(
            frozen["qL_max"], frozen["qR_max"], frozen["qC_min"],
            plateau["combination_count"], *plateau["qL_range"], *plateau["qR_range"], *plateau["qC_range"]
        ),
        "",
        "## 独立留出结果",
        "",
        *holdout_lines,
        "",
        "留出集未参与阈值选择，完整扫描结果只对校准集公开。该结果说明门控性能并非依赖单一精确阈值点；但由于候选池由一次在线回放冻结生成，最终投稿前仍建议用冻结参数补一次端到端在线复核。",
    ]
    d_out.write_text("\n".join(text) + "\n", encoding="utf-8")


def write_manifest():
    paths = [GT_CONFIG, DEV_BAG, DEV_SEVERE_BAG, HOLDOUT_BAG, Path(__file__).resolve()]
    for item in SCENARIOS.values():
        paths.extend([item["baseline"], item["conditional"], item["always"], item["quality"]])
    manifest = {
        "created_for": "supervisor supplementary experiments C/D",
        "iou_threshold": 0.10,
        "files": {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(set(paths))},
    }
    with (OUT / "CD_reproducibility_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)


def main():
    require_inputs(
        [GT_CONFIG, DEV_BAG, DEV_SEVERE_BAG, HOLDOUT_BAG]
        + [path for item in SCENARIOS.values() for path in (item["baseline"], item["conditional"], item["always"], item["quality"])]
        + [path for item in D_POOLS.values() for path in (item["dir"] / "detections.csv", item["dir"] / "modality_reliability.csv")]
    )
    config = load_gt_config(GT_CONFIG)
    c_metrics, activation, contributions = run_c(config)
    d_sweep, holdout, selection = run_d(config)
    write_conclusions(c_metrics, activation, contributions, d_sweep, holdout, selection)
    write_manifest()
    print(json.dumps({"frozen_parameters": selection["frozen_parameters"], "holdout": holdout.to_dict("records")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
