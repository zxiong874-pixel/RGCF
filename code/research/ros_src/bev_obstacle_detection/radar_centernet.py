"""Frozen Radar CenterNet runtime shared by ROS and offline validation."""

import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


RADAR_RESOLUTION_UPGRADE_TIME = 1632182400


def image_message_to_array(message):
    if message.encoding not in ("mono8", "8UC1"):
        raise ValueError("Radar CenterNet requires mono8 input, got {}".format(message.encoding))
    row_bytes = int(message.step)
    data = np.frombuffer(message.data, dtype=np.uint8)
    expected = row_bytes * int(message.height)
    if data.size < expected or row_bytes < int(message.width):
        raise ValueError("invalid Radar Image buffer")
    return data[:expected].reshape(message.height, row_bytes)[:, : message.width].copy()


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.block(inputs)


class TinyRadarCenterNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(1, 24, stride=2),
            ConvBlock(24, 48, stride=2),
            ConvBlock(48, 64),
            ConvBlock(64, 64),
        )
        self.heatmap = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
        self.regression = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 6, 1),
        )

    def forward(self, inputs):
        features = self.backbone(inputs)
        return self.heatmap(features), self.regression(features)


def radar_resolution(timestamp_seconds):
    return 0.04381 if timestamp_seconds > RADAR_RESOLUTION_UPGRADE_TIME else 0.0596


def decode_navtech_array(raw, timestamp_seconds):
    raw = np.asarray(raw)
    if raw.dtype != np.uint8 or raw.ndim != 2 or raw.shape[1] <= 11:
        raise ValueError("Navtech Radar input must be a uint8 HxW array with metadata")
    resolution = radar_resolution(float(timestamp_seconds))
    azimuths = (
        raw[:, 8:10].copy().view(np.uint16).reshape(-1)
        / 5600.0
        * 2.0
        * np.pi
    ).astype(np.float32)
    fft_data = raw[:, 11:].astype(np.float32) / 255.0
    fft_data[:, : int(round(2.5 / resolution))] = 0.0
    # Stage39 training retained every azimuth row. Keep inference byte-for-byte
    # compatible with that frozen preprocessing contract.
    return azimuths, fft_data, resolution


def polar_to_cartesian(azimuths, fft_data, radar_resolution_m, cart_resolution=0.25, width=256):
    if len(azimuths) < 2:
        raise ValueError("at least two valid Radar azimuth rows are required")
    cart_min_range = (width / 2.0 - 0.5) * cart_resolution if width % 2 == 0 else (width // 2) * cart_resolution
    coords = np.linspace(-cart_min_range, cart_min_range, width, dtype=np.float32)
    y_coord, x_coord = np.meshgrid(coords, -coords)
    sample_range = np.sqrt(y_coord * y_coord + x_coord * x_coord)
    sample_angle = np.arctan2(y_coord, x_coord)
    sample_angle += (sample_angle < 0).astype(np.float32) * 2.0 * np.pi
    sample_u = (sample_range - radar_resolution_m / 2.0) / radar_resolution_m
    upper = np.searchsorted(azimuths, sample_angle.squeeze())
    upper[upper == azimuths.shape[0]] -= 1
    lower = upper - 1
    lower[lower < 0] += 1
    upper_angle = azimuths[upper]
    difference = sample_angle.squeeze() - upper_angle
    lower_angle = azimuths[lower]
    interpolation = difference * (difference < 0) * (upper > 0) / (upper_angle - lower_angle + 1.0e-14)
    sample_v = (upper + interpolation).astype(np.float32)
    sample_u[sample_u < 0] = 0
    wrapped_fft = np.concatenate((fft_data[-1:], fft_data, fft_data[:1]), axis=0)
    warp = np.stack((sample_u, sample_v + 1), axis=-1)
    return cv2.remap(wrapped_fft, warp, None, cv2.INTER_LINEAR)


def rotated_iou(first, second):
    first_rect = ((first["center_x"], first["center_y"]), (first["size_x"], first["size_y"]), math.degrees(first["yaw"]))
    second_rect = ((second["center_x"], second["center_y"]), (second["size_x"], second["size_y"]), math.degrees(second["yaw"]))
    _, intersection = cv2.rotatedRectangleIntersection(first_rect, second_rect)
    area = 0.0 if intersection is None else abs(cv2.contourArea(intersection))
    union = first["size_x"] * first["size_y"] + second["size_x"] * second["size_y"] - area
    return area / union if union > 0.0 else 0.0


def decode_predictions(outputs, threshold, top_k=80, cart_resolution=0.25, stride=4, nms_iou=0.20):
    heatmap_logits, regression = outputs
    heatmap = torch.sigmoid(heatmap_logits)
    heatmap *= heatmap.eq(functional.max_pool2d(heatmap, 3, stride=1, padding=1)).float()
    flattened = heatmap[0, 0].flatten()
    scores, indices = torch.topk(flattened, min(top_k, flattened.numel()))
    width = heatmap.shape[-1]
    center_pixel = (width * stride - 1) / 2.0
    detections = []
    for score, flat_index in zip(scores.tolist(), indices.tolist()):
        if score < threshold:
            continue
        row, col = divmod(flat_index, width)
        values = regression[0, :, row, col].detach().cpu().numpy()
        pixel_col = (col + float(values[0])) * stride
        pixel_row = (row + float(values[1])) * stride
        detection = {
            "center_x": (center_pixel - pixel_row) * cart_resolution,
            "center_y": (pixel_col - center_pixel) * cart_resolution,
            "size_x": float(np.clip(np.exp(values[2]), 0.5, 10.0)),
            "size_y": float(np.clip(np.exp(values[3]), 0.5, 10.0)),
            "yaw": math.atan2(float(values[4]), float(values[5])),
            "score": float(score),
        }
        if all(rotated_iou(detection, previous) < nms_iou for previous in detections):
            detections.append(detection)
    return detections


def transform_detection(detection, radar_to_output):
    center = radar_to_output @ np.asarray([detection["center_x"], detection["center_y"], 0.0, 1.0])
    heading = radar_to_output[:3, :3] @ np.asarray([math.cos(detection["yaw"]), math.sin(detection["yaw"]), 0.0])
    output = dict(detection)
    output.update(center_x=float(center[0]), center_y=float(center[1]), center_z=float(center[2]), yaw=math.atan2(float(heading[1]), float(heading[0])))
    return output


class RadarCenterNetRuntime:
    def __init__(self, checkpoint_path, device="cpu", score_threshold=None, cart_resolution=0.25, cart_width=256):
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.device = torch.device(device)
        self.cart_resolution = float(cart_resolution)
        self.cart_width = int(cart_width)
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model = TinyRadarCenterNet().to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.score_threshold = float(checkpoint["validation_threshold"] if score_threshold is None else score_threshold)
        self.checkpoint_sha256 = hashlib.sha256(self.checkpoint_path.read_bytes()).hexdigest()

    def infer_cartesian(self, cartesian):
        radar = np.asarray(cartesian, dtype=np.float32)
        if radar.shape != (self.cart_width, self.cart_width):
            raise ValueError("unexpected Cartesian Radar shape: {}".format(radar.shape))
        radar = np.log1p(20.0 * np.clip(radar, 0.0, 1.0)) / math.log(21.0)
        inputs = torch.from_numpy(radar[None, None]).to(self.device)
        with torch.no_grad():
            return decode_predictions(self.model(inputs), self.score_threshold, cart_resolution=self.cart_resolution)

    def infer_navtech(self, raw, timestamp_seconds):
        azimuths, fft_data, resolution = decode_navtech_array(raw, timestamp_seconds)
        cartesian = polar_to_cartesian(azimuths, fft_data, resolution, self.cart_resolution, self.cart_width)
        # Training samples were cached as float16 before being loaded as
        # float32. Reproduce that quantization for checkpoint parity.
        return self.infer_cartesian(cartesian.astype(np.float16).astype(np.float32))
