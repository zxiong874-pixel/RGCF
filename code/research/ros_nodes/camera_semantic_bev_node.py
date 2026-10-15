#!/usr/bin/env python3
"""Project semantic masks into Camera-only BEV obstacle proposals."""

import json
import time

import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from bev_obstacle_detection.bev_grid import BevGridSpec
from bev_obstacle_detection.camera_bev_projection import (
    load_camera_calibration,
    semantic_mask_ground_contact_points,
)
from bev_obstacle_detection.msg import Detection3D, Detection3DArray
from bev_obstacle_detection.multimodal_support_fusion import support_from_points
from bev_obstacle_detection.obstacle_detection import detect_occupied_obstacles


def mono8_array(message):
    if message.encoding not in ("mono8", "8UC1"):
        raise ValueError("semantic BEV frontend requires mono8 mask")
    data = np.frombuffer(message.data, dtype=np.uint8)
    expected = int(message.step) * int(message.height)
    if data.size < expected or int(message.step) < int(message.width):
        raise ValueError("invalid semantic mask buffer")
    return data[:expected].reshape(message.height, message.step)[:, : message.width].copy()


class CameraSemanticBevNode:
    def __init__(self):
        self.projection, self.extrinsic = load_camera_calibration(
            rospy.get_param("~projection_path"),
            rospy.get_param("~lidar_to_camera_transform_path"),
        )
        self.output_frame = rospy.get_param("~output_frame", "boreas_lidar")
        self.source = rospy.get_param("~source", "Camera_Semantic_Only")
        self.spec = BevGridSpec(
            x_min=float(rospy.get_param("~x_min", -10.0)),
            x_max=float(rospy.get_param("~x_max", 30.0)),
            y_min=float(rospy.get_param("~y_min", -20.0)),
            y_max=float(rospy.get_param("~y_max", 20.0)),
            z_min=float(rospy.get_param("~z_min", -2.0)),
            z_max=float(rospy.get_param("~z_max", 5.0)),
            resolution=float(rospy.get_param("~resolution", 0.2)),
        )
        self.min_component_area = int(rospy.get_param("~min_component_area_px", 128))
        self.max_component_fraction = float(rospy.get_param("~max_component_fraction", 0.15))
        self.column_stride = int(rospy.get_param("~column_stride_px", 8))
        self.dilation_cells = int(rospy.get_param("~dilation_cells", 4))
        self.min_cells = int(rospy.get_param("~min_cells", 2))
        self.min_size_x = float(rospy.get_param("~min_size_x", 4.0))
        self.min_size_y = float(rospy.get_param("~min_size_y", 2.0))
        self.max_size_x = float(rospy.get_param("~max_size_x", 8.0))
        self.max_size_y = float(rospy.get_param("~max_size_y", 8.0))
        self.base_score = float(rospy.get_param("~base_score", 0.35))
        output_topic = rospy.get_param("~output_topic", "/boreas/camera/semantic_bev_detections")
        self.publisher = rospy.Publisher(output_topic, Detection3DArray, queue_size=4)
        self.diagnostics = rospy.Publisher("~diagnostics", String, queue_size=4)
        input_topic = rospy.get_param("~input_topic", "/boreas/camera/semantic_mask")
        self.subscriber = rospy.Subscriber(input_topic, Image, self.callback, queue_size=1, buff_size=16 * 1024 * 1024)

    def callback(self, message):
        started = time.perf_counter()
        try:
            mask = mono8_array(message)
            points, projection_stats = semantic_mask_ground_contact_points(
                mask,
                self.projection,
                self.extrinsic,
                ground_z=-1.5,
                min_range_m=1.0,
                max_range_m=30.0,
                min_component_area_px=self.min_component_area,
                max_component_fraction=self.max_component_fraction,
                reject_border_components=True,
                column_stride_px=self.column_stride,
                bottom_band_px=3,
            )
            support = support_from_points(points, self.spec, dilation_cells=self.dilation_cells)
            obstacles = detect_occupied_obstacles(
                support > 0.0,
                self.spec,
                min_cells=self.min_cells,
                max_size_x=self.max_size_x,
                max_size_y=self.max_size_y,
            )
            score = float(np.clip(self.base_score + 0.08 * np.log1p(len(points)), 0.0, 0.99))
            output = Detection3DArray()
            output.header = message.header
            output.header.frame_id = self.output_frame
            sample_token = str(message.header.stamp.to_nsec())
            for obstacle in obstacles:
                detection = Detection3D()
                detection.obstacle_id = obstacle.obstacle_id
                detection.center_x = obstacle.center_x
                detection.center_y = obstacle.center_y
                detection.center_z = 0.0
                detection.size_x = max(obstacle.size_x, self.min_size_x)
                detection.size_y = max(obstacle.size_y, self.min_size_y)
                detection.size_z = 1.5
                detection.yaw = 0.0
                detection.score = score
                detection.class_id = -1
                detection.class_name = "obstacle"
                detection.source = self.source
                detection.sample_token = sample_token
                output.detections.append(detection)
            self.publisher.publish(output)
            self.diagnostics.publish(String(data=json.dumps({
                "timestamp": message.header.stamp.to_sec(),
                "mask_coverage": float(np.count_nonzero(mask)) / float(mask.size),
                "ground_points": len(points),
                "detections": len(output.detections),
                "processing_ms": (time.perf_counter() - started) * 1000.0,
                **projection_stats,
            }, sort_keys=True)))
        except Exception as error:
            rospy.logerr_throttle(2.0, "Camera semantic BEV projection failed: %s", error)


def main():
    rospy.init_node("camera_semantic_bev_frontend")
    CameraSemanticBevNode()
    rospy.spin()


if __name__ == "__main__":
    main()
