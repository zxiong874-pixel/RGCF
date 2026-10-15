"""BEV grid indexing utilities."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BevGridSpec:
    x_min: float = -10.0
    x_max: float = 30.0
    y_min: float = -20.0
    y_max: float = 20.0
    z_min: float = -2.0
    z_max: float = 5.0
    resolution: float = 0.2

    @property
    def width(self):
        return int(round((self.x_max - self.x_min) / self.resolution))

    @property
    def height(self):
        return int(round((self.y_max - self.y_min) / self.resolution))

    def validate(self):
        if self.resolution <= 0.0:
            raise ValueError("resolution must be positive")
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")
        if self.y_max <= self.y_min:
            raise ValueError("y_max must be greater than y_min")
        if self.z_max <= self.z_min:
            raise ValueError("z_max must be greater than z_min")


def metric_xy_to_grid_rc(points_xy, spec):
    spec.validate()
    points = np.asarray(points_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape Nx2")

    cols = np.floor((points[:, 0] - spec.x_min) / spec.resolution).astype(int)
    rows = np.floor((spec.y_max - points[:, 1]) / spec.resolution).astype(int)
    return rows, cols


def grid_rc_to_metric_xy_centers(rows, cols, spec):
    spec.validate()
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    if rows.shape != cols.shape:
        raise ValueError("rows and cols must have the same shape")

    x = spec.x_min + (cols.astype(float) + 0.5) * spec.resolution
    y = spec.y_max - (rows.astype(float) + 0.5) * spec.resolution
    return x, y


def in_bounds(points_xyz, spec):
    spec.validate()
    points = np.asarray(points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape Nx3")

    return (
        (points[:, 0] >= spec.x_min)
        & (points[:, 0] < spec.x_max)
        & (points[:, 1] >= spec.y_min)
        & (points[:, 1] < spec.y_max)
        & (points[:, 2] >= spec.z_min)
        & (points[:, 2] < spec.z_max)
    )


def points_to_grid_indices(points_xyz, spec):
    points = np.asarray(points_xyz, dtype=float)
    mask = in_bounds(points, spec)
    rows, cols = metric_xy_to_grid_rc(points[mask, :2], spec)
    return rows, cols, mask
