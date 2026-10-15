"""Temporal point cloud fusion using VINS poses."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from bev_obstacle_detection.fusion_core import align_points_to_target_frame


@dataclass(frozen=True)
class StampedPointCloud:
    stamp_sec: float
    points_xyz: np.ndarray
    world_from_body: np.ndarray


class TemporalPointCloudBuffer:
    def __init__(self, max_frames=3, max_age_sec=0.5, decay_tau_sec=0.0):
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if max_age_sec <= 0.0:
            raise ValueError("max_age_sec must be positive")
        if decay_tau_sec < 0.0:
            raise ValueError("decay_tau_sec must be non-negative")
        self.max_frames = int(max_frames)
        self.max_age_sec = float(max_age_sec)
        self.decay_tau_sec = float(decay_tau_sec)
        self._items = deque()

    def add(self, stamp_sec, points_xyz, world_from_body):
        points = np.asarray(points_xyz, dtype=float).reshape((-1, 3))
        pose = np.asarray(world_from_body, dtype=float)
        if pose.shape != (4, 4):
            raise ValueError("world_from_body must be 4x4")
        self._items.append(StampedPointCloud(float(stamp_sec), points, pose))
        while len(self._items) > self.max_frames:
            self._items.popleft()
        self.prune(float(stamp_sec))

    def prune(self, reference_stamp_sec):
        reference = float(reference_stamp_sec)
        while self._items and reference - self._items[0].stamp_sec > self.max_age_sec:
            self._items.popleft()

    def clear(self):
        self._items.clear()

    def aligned_points(self, target_world_from_body, target_stamp_sec):
        points, _ = self.aligned_points_with_weights(target_world_from_body, target_stamp_sec)
        return points

    def aligned_points_with_weights(self, target_world_from_body, target_stamp_sec):
        self.prune(target_stamp_sec)
        if not self._items:
            return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)
        aligned = []
        weights = []
        for item in self._items:
            if item.points_xyz.size == 0:
                continue
            aligned.append(align_points_to_target_frame(item.points_xyz, item.world_from_body, target_world_from_body))
            age = max(0.0, float(target_stamp_sec) - item.stamp_sec)
            if self.decay_tau_sec > 0.0:
                weight = float(np.exp(-age / self.decay_tau_sec))
            else:
                weight = 1.0
            weights.append(np.full((item.points_xyz.shape[0],), weight, dtype=float))
        if not aligned:
            return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)
        return np.vstack(aligned), np.concatenate(weights)

    def __len__(self):
        return len(self._items)
