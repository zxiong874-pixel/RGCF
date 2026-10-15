"""Deterministic odometry degradation helpers for VINS robustness tests."""

import math

import numpy as np


def validate_odometry_degradation_params(position_noise_std=0.0, yaw_noise_std=0.0, drop_rate=0.0):
    if position_noise_std < 0.0:
        raise ValueError("position_noise_std must be non-negative")
    if yaw_noise_std < 0.0:
        raise ValueError("yaw_noise_std must be non-negative")
    if drop_rate < 0.0 or drop_rate > 1.0:
        raise ValueError("drop_rate must be in [0, 1]")


def yaw_from_quaternion_xyzw(quaternion_xyzw):
    x, y, z, w = [float(value) for value in quaternion_xyzw]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_xyzw_from_yaw(yaw):
    half = 0.5 * float(yaw)
    return (0.0, 0.0, math.sin(half), math.cos(half))


def degrade_odometry_message(
    odometry,
    rng,
    position_noise_std=0.0,
    yaw_noise_std=0.0,
):
    validate_odometry_degradation_params(
        position_noise_std=position_noise_std,
        yaw_noise_std=yaw_noise_std,
        drop_rate=0.0,
    )
    degraded = odometry
    pose = degraded.pose.pose
    if position_noise_std > 0.0:
        noise = rng.normal(0.0, float(position_noise_std), size=3)
        pose.position.x += float(noise[0])
        pose.position.y += float(noise[1])
        pose.position.z += float(noise[2])
    if yaw_noise_std > 0.0:
        q = pose.orientation
        yaw = yaw_from_quaternion_xyzw((q.x, q.y, q.z, q.w))
        yaw += float(rng.normal(0.0, float(yaw_noise_std)))
        q.x, q.y, q.z, q.w = quaternion_xyzw_from_yaw(yaw)
    return degraded


def should_drop_odometry(rng, drop_rate=0.0):
    validate_odometry_degradation_params(drop_rate=drop_rate)
    return drop_rate > 0.0 and rng.random_sample() < float(drop_rate)


def odometry_degradation_label(position_noise_std=0.0, yaw_noise_std=0.0, drop_rate=0.0):
    parts = []
    if position_noise_std > 0.0:
        parts.append("pos_noise_{:.3f}".format(position_noise_std).replace(".", "p"))
    if yaw_noise_std > 0.0:
        parts.append("yaw_noise_{:.3f}".format(yaw_noise_std).replace(".", "p"))
    if drop_rate > 0.0:
        parts.append("odom_drop_{:.2f}".format(drop_rate).replace(".", "p"))
    return "_".join(parts) if parts else "clean"
