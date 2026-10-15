#!/usr/bin/env python3
"""Evaluate PX4 and VINS trajectories against Gazebo ground truth."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rosbag
from matplotlib import font_manager


def load_interval(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("trajectory tracking CSV is empty")
    return float(rows[0]["stamp_sec"]), float(rows[-1]["stamp_sec"])


def extract_trajectories(bag_path, model_name, start, end):
    topics = (
        "/gazebo/model_states",
        "/iris_0/mavros/local_position/pose",
        "/iris_0/vins_estimator/odometry",
    )
    trajectories = {"GT": [], "PX4": [], "VINS": []}
    with rosbag.Bag(str(bag_path), "r") as bag:
        for topic, msg, stamp in bag.read_messages(topics=topics):
            timestamp = stamp.to_sec()
            if timestamp < start or timestamp > end:
                continue
            if topic == "/gazebo/model_states":
                try:
                    index = msg.name.index(model_name)
                except ValueError:
                    continue
                point = msg.pose[index].position
                trajectories["GT"].append((timestamp, point.x, point.y, point.z))
            elif topic == "/iris_0/mavros/local_position/pose":
                point = msg.pose.position
                trajectories["PX4"].append((timestamp, point.x, point.y, point.z))
            else:
                point = msg.pose.pose.position
                trajectories["VINS"].append((timestamp, point.x, point.y, point.z))
    return {name: np.asarray(values, dtype=float) for name, values in trajectories.items()}


def nearest_ground_truth(ground_truth, estimates, max_delta=0.05):
    indices = np.searchsorted(ground_truth[:, 0], estimates[:, 0])
    indices = np.clip(indices, 1, len(ground_truth) - 1)
    left = indices - 1
    choose_left = (
        np.abs(ground_truth[left, 0] - estimates[:, 0])
        <= np.abs(ground_truth[indices, 0] - estimates[:, 0])
    )
    selected = np.where(choose_left, left, indices)
    deltas = np.abs(ground_truth[selected, 0] - estimates[:, 0])
    valid = deltas <= max_delta
    return ground_truth[selected[valid]], estimates[valid]


def rigid_align(source, target):
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return (rotation @ source.T).T + translation, rotation, translation


def path_length(points):
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def evaluate_method(ground_truth, estimates, stable_altitude):
    gt, estimate = nearest_ground_truth(ground_truth, estimates)
    if len(estimate) < 3:
        raise RuntimeError("too few synchronized samples")
    aligned, rotation, translation = rigid_align(estimate[:, 1:4], gt[:, 1:4])
    errors = np.linalg.norm(aligned - gt[:, 1:4], axis=1)
    stable = gt[:, 3] >= stable_altitude

    def summarize(mask):
        selected = errors[mask]
        gt_points = gt[mask, 1:4]
        estimate_points = aligned[mask]
        return {
            "samples": int(len(selected)),
            "ate_rmse_m": float(np.sqrt(np.mean(selected ** 2))),
            "ate_mean_m": float(np.mean(selected)),
            "ate_p95_m": float(np.percentile(selected, 95)),
            "ate_max_m": float(np.max(selected)),
            "gt_path_length_m": path_length(gt_points),
            "estimated_path_length_m": path_length(estimate_points),
        }

    return {
        "all": summarize(np.ones(len(errors), dtype=bool)),
        "stable": summarize(stable),
        "timestamps": estimate[:, 0],
        "ground_truth": gt[:, 1:4],
        "aligned": aligned,
        "errors": errors,
        "stable_mask": stable,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
    }


def configure_chinese_font():
    for font_path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ):
        if Path(font_path).is_file():
            font_manager.fontManager.addfont(font_path)
            name = font_manager.FontProperties(fname=font_path).get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return


def plot_results(path, evaluations):
    configure_chinese_font()
    colors = {"GT": "#222222", "PX4": "#2A9D8F", "VINS": "#D95F59"}
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    first = next(iter(evaluations.values()))
    axes[0].plot(
        first["ground_truth"][:, 0],
        first["ground_truth"][:, 1],
        label="Gazebo GT",
        color=colors["GT"],
        linewidth=2.2,
    )
    for name, result in evaluations.items():
        axes[0].plot(
            result["aligned"][:, 0],
            result["aligned"][:, 1],
            label=name,
            color=colors[name],
            linewidth=1.4,
        )
        relative_time = result["timestamps"] - result["timestamps"][0]
        axes[1].plot(
            relative_time,
            result["errors"],
            label=name,
            color=colors[name],
            linewidth=1.2,
        )
    axes[0].set_title("水平轨迹对比（刚体对齐）")
    axes[0].set_xlabel("X / m")
    axes[0].set_ylabel("Y / m")
    axes[0].axis("equal")
    axes[1].set_title("绝对轨迹误差 ATE")
    axes[1].set_xlabel("飞行时间 / s")
    axes[1].set_ylabel("位置误差 / m")
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)
    fig.suptitle("PX4 动力学轨迹下的 VINS 一致性验证", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary_csv(path, evaluations):
    rows = []
    for method, result in evaluations.items():
        for interval in ("all", "stable"):
            rows.append({"method": method, "interval": interval, **result[interval]})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--tracking-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="iris_0")
    parser.add_argument("--stable-altitude", type=float, default=0.8)
    args = parser.parse_args()

    start, end = load_interval(args.tracking_csv)
    trajectories = extract_trajectories(args.bag, args.model_name, start, end)
    evaluations = {
        method: evaluate_method(trajectories["GT"], trajectories[method], args.stable_altitude)
        for method in ("PX4", "VINS")
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "stage46_trajectory_consistency.csv"
    json_path = args.out_dir / "stage46_trajectory_consistency.json"
    plot_path = args.out_dir / "stage46_trajectory_consistency.png"
    write_summary_csv(csv_path, evaluations)
    payload = {
        "bag": str(args.bag),
        "tracking_csv": str(args.tracking_csv),
        "interval_sec": [start, end],
        "stable_altitude_m": args.stable_altitude,
        "methods": {
            method: {
                "all": result["all"],
                "stable": result["stable"],
                "rotation": result["rotation"],
                "translation": result["translation"],
            }
            for method, result in evaluations.items()
        },
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_results(plot_path, evaluations)
    print(csv_path)
    print(json_path)
    print(plot_path)


if __name__ == "__main__":
    main()
