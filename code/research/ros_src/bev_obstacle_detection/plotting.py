"""Metrics plotting helpers without a pandas dependency."""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STANDARD_PLOTS = (
    ("mAP", "map_by_environment.png", "mAP by environment", "mAP"),
    ("precision", "precision_by_environment.png", "Precision by environment", "Precision"),
    ("recall", "recall_by_environment.png", "Recall by environment", "Recall"),
    ("f1", "f1_by_environment.png", "F1 by environment", "F1"),
    ("mean_iou", "mean_iou_by_environment.png", "Mean IoU by environment", "Mean IoU"),
    ("ate_rmse", "ate_rmse_by_environment.png", "ATE RMSE by environment", "ATE RMSE"),
    (
        "tracking_failure_rate",
        "failure_rate_by_environment.png",
        "Tracking failure rate by environment",
        "Failure rate",
    ),
)


def read_metrics_csv(path):
    with open(path, "r", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def numeric_value(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def ordered_unique(rows, key):
    values = []
    seen = set()
    for row in rows:
        value = row.get(key, "")
        if value and value not in seen:
            values.append(value)
            seen.add(value)
    return values


def mean_by_environment_method(rows, metric):
    values = {}
    for row in rows:
        environment = row.get("environment", "")
        method = row.get("method", "")
        number = numeric_value(row.get(metric))
        if not environment or not method or number is None:
            continue
        values.setdefault((environment, method), []).append(number)
    return {key: sum(items) / len(items) for key, items in values.items()}


def plot_grouped_metric(rows, metric, output_path, title=None, ylabel=None):
    environments = ordered_unique(rows, "environment")
    methods = ordered_unique(rows, "method")
    means = mean_by_environment_method(rows, metric)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(8.0, len(environments) * 0.8), 4.8))
    if not environments or not methods or not means:
        ax.text(0.5, 0.5, "No numeric data yet", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        x = np.arange(len(environments))
        width = 0.8 / max(1, len(methods))
        for index, method in enumerate(methods):
            y = [means.get((environment, method), np.nan) for environment in environments]
            offset = (index - (len(methods) - 1) / 2.0) * width
            ax.bar(x + offset, y, width=width, label=method)
        ax.set_xticks(x)
        ax.set_xticklabels(environments, rotation=30, ha="right")
        ax.set_ylabel(ylabel or metric)
        ax.set_title(title or metric)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=160)
    plt.close(fig)
    return output_path


def plot_latency_fps(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    plotters = (("latency_ms", "Latency by environment", "Latency (ms)"), ("fps", "FPS by environment", "FPS"))

    for ax, (metric, title, ylabel) in zip(axes, plotters):
        environments = ordered_unique(rows, "environment")
        methods = ordered_unique(rows, "method")
        means = mean_by_environment_method(rows, metric)
        if not environments or not methods or not means:
            ax.text(0.5, 0.5, "No numeric data yet", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        x = np.arange(len(environments))
        width = 0.8 / max(1, len(methods))
        for index, method in enumerate(methods):
            y = [means.get((environment, method), np.nan) for environment in environments]
            offset = (index - (len(methods) - 1) / 2.0) * width
            ax.bar(x + offset, y, width=width, label=method)
        ax.set_xticks(x)
        ax.set_xticklabels(environments, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=160)
    plt.close(fig)
    return output_path


def generate_standard_plots(metrics_path, figures_dir):
    rows = read_metrics_csv(metrics_path)
    figures_dir = Path(figures_dir)
    outputs = []
    for metric, filename, title, ylabel in STANDARD_PLOTS:
        outputs.append(plot_grouped_metric(rows, metric, figures_dir / filename, title=title, ylabel=ylabel))
    outputs.append(plot_latency_fps(rows, figures_dir / "latency_fps.png"))
    return outputs


def mean_by_distance_method(rows, metric):
    values = {}
    for row in rows:
        distance_bin = row.get("distance_bin", "")
        method = row.get("method", "")
        number = numeric_value(row.get(metric))
        if not distance_bin or not method or number is None:
            continue
        values.setdefault((distance_bin, method), []).append(number)
    return {key: sum(items) / len(items) for key, items in values.items()}


def plot_distance_metric(rows, metric, output_path, title=None, ylabel=None):
    distance_bins = ordered_unique(rows, "distance_bin")
    methods = [method for method in ordered_unique(rows, "method") if method != "vins_only"]
    means = mean_by_distance_method(rows, metric)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(8.0, len(distance_bins) * 1.2), 4.8))
    if not distance_bins or not methods or not means:
        ax.text(0.5, 0.5, "No numeric distance data yet", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        x = np.arange(len(distance_bins))
        width = 0.8 / max(1, len(methods))
        for index, method in enumerate(methods):
            y = [means.get((distance_bin, method), np.nan) for distance_bin in distance_bins]
            offset = (index - (len(methods) - 1) / 2.0) * width
            ax.bar(x + offset, y, width=width, label=method)
        ax.set_xticks(x)
        ax.set_xticklabels(distance_bins, rotation=20, ha="right")
        ax.set_ylabel(ylabel or metric)
        ax.set_title(title or "{} by distance".format(metric))
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=160)
    plt.close(fig)
    return output_path


def generate_distance_plots(distance_summary_csv, figures_dir):
    rows = read_metrics_csv(distance_summary_csv)
    figures_dir = Path(figures_dir)
    outputs = []
    plots = (
        ("recall", "recall_by_distance.png", "Recall by distance", "Recall"),
        ("precision", "precision_by_distance.png", "Precision by distance", "Precision"),
        ("f1", "f1_by_distance.png", "F1 by distance", "F1"),
        ("mean_iou", "mean_iou_by_distance.png", "Mean IoU by distance", "Mean IoU"),
    )
    for metric, filename, title, ylabel in plots:
        outputs.append(plot_distance_metric(rows, metric, figures_dir / filename, title=title, ylabel=ylabel))
    return outputs
