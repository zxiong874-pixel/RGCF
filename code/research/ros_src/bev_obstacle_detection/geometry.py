"""Coordinate transform helpers for VINS-aligned BEV fusion."""

import numpy as np


def normalize_quaternion_xyzw(quaternion_xyzw):
    q = np.asarray(quaternion_xyzw, dtype=float)
    if q.shape != (4,):
        raise ValueError("quaternion must be [x, y, z, w]")

    norm = np.linalg.norm(q)
    if norm <= 0.0:
        raise ValueError("quaternion norm must be positive")

    return q / norm


def quaternion_xyzw_to_rotation_matrix(quaternion_xyzw):
    x, y, z, w = normalize_quaternion_xyzw(quaternion_xyzw)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def transform_from_translation_quaternion(translation_xyz, quaternion_xyzw):
    translation = np.asarray(translation_xyz, dtype=float)
    if translation.shape != (3,):
        raise ValueError("translation must be [x, y, z]")

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = quaternion_xyzw_to_rotation_matrix(quaternion_xyzw)
    transform[:3, 3] = translation
    return transform


def inverse_transform(transform):
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError("transform must be 4x4")

    inverse = np.eye(4, dtype=float)
    rotation = transform[:3, :3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ transform[:3, 3]
    return inverse


def transform_points(points_xyz, transform):
    points = np.asarray(points_xyz, dtype=float)
    transform = np.asarray(transform, dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape Nx3")
    if transform.shape != (4, 4):
        raise ValueError("transform must be 4x4")

    return points @ transform[:3, :3].T + transform[:3, 3]
