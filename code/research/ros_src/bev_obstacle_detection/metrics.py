"""Metrics CSV template generation for staged experiments."""

import csv


BASE_COLUMNS = ["experiment_id", "dataset", "environment", "method"]


def metric_columns(metrics):
    columns = []
    for metric in metrics:
        metric_name = str(metric)
        if metric_name not in columns:
            columns.append(metric_name)
    return columns


def metrics_template_rows(config):
    for dataset in config.datasets:
        for environment in config.environments:
            for method in config.methods:
                row = {
                    "experiment_id": config.experiment_id,
                    "dataset": dataset.name,
                    "environment": environment,
                    "method": method,
                }
                for metric in config.metrics:
                    row[metric] = ""
                yield row


def write_metrics_template(path, config):
    columns = BASE_COLUMNS + metric_columns(config.metrics)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in metrics_template_rows(config):
            writer.writerow(row)
    return columns
