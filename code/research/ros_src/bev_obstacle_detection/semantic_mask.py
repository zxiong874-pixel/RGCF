"""Simple semantic mask helpers for camera-BEV fusion experiments."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraProjectionConfig:
    horizontal_fov_deg: float = 69.0
    vertical_fov_deg: float = 42.0
    min_depth_m: float = 0.2
    obstacle_height_m: float = 1.5
    min_box_pixels: int = 3
    dilation_pixels: int = 2


def project_body_point_to_image(x, y, z, image_width, image_height, config=None):
    config = config or CameraProjectionConfig()
    depth = float(x)
    if depth <= float(config.min_depth_m):
        return None
    width = int(image_width)
    height = int(image_height)
    fx = width / (2.0 * math.tan(math.radians(float(config.horizontal_fov_deg)) / 2.0))
    fy = height / (2.0 * math.tan(math.radians(float(config.vertical_fov_deg)) / 2.0))
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    u = fx * (-float(y) / depth) + cx
    v = fy * (-float(z) / depth) + cy
    if u < 0.0 or u >= width or v < 0.0 or v >= height:
        return None
    return u, v


def _dilate_mask(mask, radius_pixels):
    radius = int(radius_pixels)
    if radius <= 0 or not np.any(mask):
        return mask
    source = mask.astype(bool)
    output = source.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            y_src_start = max(0, -dy)
            y_src_end = source.shape[0] - max(0, dy)
            x_src_start = max(0, -dx)
            x_src_end = source.shape[1] - max(0, dx)
            y_dst_start = max(0, dy)
            y_dst_end = source.shape[0] - max(0, -dy)
            x_dst_start = max(0, dx)
            x_dst_end = source.shape[1] - max(0, -dx)
            output[y_dst_start:y_dst_end, x_dst_start:x_dst_end] |= source[
                y_src_start:y_src_end,
                x_src_start:x_src_end,
            ]
    return output


def semantic_mask_from_obstacle_centers(obstacle_centers, image_width, image_height, config=None):
    config = config or CameraProjectionConfig()
    width = int(image_width)
    height = int(image_height)
    mask = np.zeros((height, width), dtype=bool)
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")

    for obstacle in obstacle_centers:
        center_x = float(obstacle["center_x"])
        center_y = float(obstacle["center_y"])
        if center_x <= float(config.min_depth_m):
            continue

        half_width = max(float(obstacle.get("size_y", 0.0)) / 2.0, 0.1)
        bottom_z = 0.0
        top_z = float(config.obstacle_height_m)
        corners = (
            (center_x, center_y - half_width, bottom_z),
            (center_x, center_y + half_width, bottom_z),
            (center_x, center_y - half_width, top_z),
            (center_x, center_y + half_width, top_z),
        )
        pixels = [
            project_body_point_to_image(x, y, z, width, height, config=config)
            for x, y, z in corners
        ]
        pixels = [pixel for pixel in pixels if pixel is not None]
        if not pixels:
            continue
        us = [pixel[0] for pixel in pixels]
        vs = [pixel[1] for pixel in pixels]
        u_min = int(max(0, math.floor(min(us))))
        u_max = int(min(width - 1, math.ceil(max(us))))
        v_min = int(max(0, math.floor(min(vs))))
        v_max = int(min(height - 1, math.ceil(max(vs))))
        min_pixels = max(1, int(config.min_box_pixels))
        if u_max - u_min + 1 < min_pixels:
            pad = int(math.ceil((min_pixels - (u_max - u_min + 1)) / 2.0))
            u_min = max(0, u_min - pad)
            u_max = min(width - 1, u_max + pad)
        if v_max - v_min + 1 < min_pixels:
            pad = int(math.ceil((min_pixels - (v_max - v_min + 1)) / 2.0))
            v_min = max(0, v_min - pad)
            v_max = min(height - 1, v_max + pad)
        mask[v_min : v_max + 1, u_min : u_max + 1] = True

    return (_dilate_mask(mask, config.dilation_pixels).astype(np.uint8) * 255)
