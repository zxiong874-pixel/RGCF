"""Reusable XTDrone trajectory definitions for reproducible recordings."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PoseSegment:
    name: str
    duration_sec: float
    x: float
    y: float
    z: float
    yaw: float


def complex_motion_segments(altitude=1.2, duration_scale=1.0):
    """Return a safe near-origin trajectory for the complex obstacle map."""
    if altitude <= 0.0:
        raise ValueError("altitude must be positive")
    if duration_scale <= 0.0:
        raise ValueError("duration_scale must be positive")

    raw_segments = (
        ("takeoff_hold", 6.0, 0.00, 0.00, altitude, 0.00),
        ("left_peek", 5.0, 0.40, 0.55, altitude, 0.10),
        ("right_peek", 6.0, 1.05, -0.45, altitude, -0.08),
        ("left_forward_peek", 6.0, 1.35, 0.45, altitude, 0.12),
        ("center_return", 5.0, 0.40, 0.00, altitude, 0.00),
        ("final_hover", 4.0, 0.00, 0.00, altitude, 0.00),
        ("final_stabilize", 3.0, 0.00, 0.00, altitude, 0.00),
    )
    return tuple(
        PoseSegment(
            name=name,
            duration_sec=duration * duration_scale,
            x=x,
            y=y,
            z=z,
            yaw=yaw,
        )
        for name, duration, x, y, z, yaw in raw_segments
    )


def total_duration(segments):
    return sum(segment.duration_sec for segment in segments)


def shortest_angle_delta(start, end):
    """Return the signed shortest angular displacement from start to end."""
    return math.atan2(math.sin(end - start), math.cos(end - start))


def smoothstep(value):
    """Clamp value to [0, 1] and apply a zero-slope cubic transition."""
    clamped = min(max(float(value), 0.0), 1.0)
    return clamped * clamped * (3.0 - 2.0 * clamped)


def interpolate_pose(start, end, progress):
    """Interpolate two pose segments without discontinuous yaw wrapping."""
    weight = smoothstep(progress)
    yaw = start.yaw + shortest_angle_delta(start.yaw, end.yaw) * weight
    return PoseSegment(
        name=end.name,
        duration_sec=end.duration_sec,
        x=start.x + (end.x - start.x) * weight,
        y=start.y + (end.y - start.y) * weight,
        z=start.z + (end.z - start.z) * weight,
        yaw=yaw,
    )
