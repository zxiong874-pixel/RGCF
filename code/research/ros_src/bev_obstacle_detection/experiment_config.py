"""Experiment configuration helpers for reproducible BEV + VINS runs."""

from dataclasses import dataclass
from pathlib import Path

import yaml


DEFAULT_METRICS = [
    "mAP",
    "precision",
    "recall",
    "f1",
    "mean_iou",
    "ate_rmse",
    "rpe_rmse",
    "tracking_failure_rate",
    "miss_rate",
    "false_positive_rate",
    "collision_rate",
    "near_miss_rate",
    "latency_ms",
    "fps",
    "cpu_percent",
    "gpu_percent",
    "memory_mb",
]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    type: str
    root: str
    rosbag: str = ""
    notes: str = ""


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    description: str
    workspace_root: str
    datasets: tuple
    methods: tuple
    environments: tuple
    topics: dict
    outputs: dict
    metrics: tuple

    @property
    def output_root(self):
        return self.outputs.get("root", str(Path(self.workspace_root) / "results"))

    @property
    def metrics_filename(self):
        return self.outputs.get("metrics_filename", "metrics.csv")

    @property
    def config_snapshot(self):
        return self.outputs.get("config_snapshot", "config.yaml")

    def result_dir(self, output_root_override=None):
        root = Path(output_root_override or self.output_root).expanduser()
        return root / self.experiment_id

    def validate(self):
        if not self.experiment_id:
            raise ValueError("experiment_id must be non-empty")
        if "/" in self.experiment_id or "\\" in self.experiment_id:
            raise ValueError("experiment_id must not contain path separators")
        if not self.datasets:
            raise ValueError("at least one dataset is required")
        if not self.methods:
            raise ValueError("at least one method is required")
        if not self.environments:
            raise ValueError("at least one environment is required")
        if not self.metrics:
            raise ValueError("at least one metric is required")
        for dataset in self.datasets:
            if not dataset.name:
                raise ValueError("dataset name must be non-empty")
            if not dataset.type:
                raise ValueError("dataset type must be non-empty")


def load_experiment_config(path):
    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}

    datasets = tuple(
        DatasetConfig(
            name=str(item.get("name", "")),
            type=str(item.get("type", "")),
            root=str(item.get("root", "")),
            rosbag=str(item.get("rosbag", "")),
            notes=str(item.get("notes", "")),
        )
        for item in raw.get("datasets", [])
    )

    config = ExperimentConfig(
        experiment_id=str(raw.get("experiment_id", "")),
        description=str(raw.get("description", "")),
        workspace_root=str(raw.get("workspace_root", "")),
        datasets=datasets,
        methods=tuple(str(item) for item in raw.get("methods", [])),
        environments=tuple(str(item) for item in raw.get("environments", [])),
        topics=dict(raw.get("topics", {})),
        outputs=dict(raw.get("outputs", {})),
        metrics=tuple(str(item) for item in raw.get("metrics", DEFAULT_METRICS)),
    )
    config.validate()
    return config, raw
