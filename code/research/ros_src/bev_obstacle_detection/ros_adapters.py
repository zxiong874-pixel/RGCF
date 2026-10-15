"""Small ROS message adapters kept separate from core BEV math."""

import numpy as np

from bev_obstacle_detection.geometry import transform_from_translation_quaternion


_POINT_FIELD_DTYPES = {
    1: "i1",
    2: "u1",
    3: "i2",
    4: "u2",
    5: "i4",
    6: "u4",
    7: "f4",
    8: "f8",
}


def pose_to_transform(pose):
    position = pose.position
    orientation = pose.orientation
    return transform_from_translation_quaternion(
        (position.x, position.y, position.z),
        (orientation.x, orientation.y, orientation.z, orientation.w),
    )


def odometry_to_transform(odometry):
    return pose_to_transform(odometry.pose.pose)


def message_stamp_sec(message):
    try:
        return message.header.stamp.to_sec()
    except AttributeError as exc:
        raise ValueError("message must have header.stamp") from exc


def pointcloud2_to_xyz_array(message):
    array = _pointcloud2_fields_array(message, ("x", "y", "z"))
    if array is not None:
        valid = np.all(np.isfinite(array[:, :3]), axis=1)
        return array[valid, :3]
    try:
        from sensor_msgs import point_cloud2
    except ImportError as exc:
        raise RuntimeError("sensor_msgs.point_cloud2 is required for PointCloud2 conversion") from exc

    points = point_cloud2.read_points(message, field_names=("x", "y", "z"), skip_nans=True)
    array = np.asarray([tuple(point) for point in points], dtype=float)
    return array.reshape((-1, 3))


def pointcloud2_to_xyz_intensity_arrays(message, intensity_field="intensity"):
    array = _pointcloud2_fields_array(
        message,
        ("x", "y", "z", intensity_field),
    )
    if array is not None:
        valid = np.all(np.isfinite(array[:, :3]), axis=1)
        array = array[valid]
        return array[:, :3], array[:, 3]
    try:
        from sensor_msgs import point_cloud2
    except ImportError as exc:
        raise RuntimeError("sensor_msgs.point_cloud2 is required for PointCloud2 conversion") from exc

    available_fields = {field.name for field in getattr(message, "fields", ())}
    if intensity_field not in available_fields:
        raise ValueError("PointCloud2 is missing {} field".format(intensity_field))
    points = point_cloud2.read_points(
        message,
        field_names=("x", "y", "z", intensity_field),
        skip_nans=True,
    )
    array = np.asarray([tuple(point) for point in points], dtype=float).reshape((-1, 4))
    return array[:, :3], array[:, 3]


def _pointcloud2_fields_array(message, field_names):
    """Vectorized PointCloud2 parsing for scalar standard ROS fields."""
    fields = {field.name: field for field in getattr(message, "fields", ())}
    selected = []
    endian = ">" if bool(getattr(message, "is_bigendian", False)) else "<"
    for name in field_names:
        field = fields.get(name)
        if field is None or int(getattr(field, "count", 1)) != 1:
            return None
        data_type = _POINT_FIELD_DTYPES.get(int(field.datatype))
        if data_type is None:
            return None
        selected.append(
            (str(name), endian + data_type, int(field.offset))
        )

    point_step = int(getattr(message, "point_step", 0))
    width = int(getattr(message, "width", 0))
    height = int(getattr(message, "height", 0))
    row_step = int(getattr(message, "row_step", 0))
    if point_step <= 0 or width < 0 or height <= 0:
        return None
    names = [item[0] for item in selected]
    formats = [item[1] for item in selected]
    offsets = [item[2] for item in selected]
    if any(
        offset < 0
        or offset + np.dtype(data_type).itemsize > point_step
        for offset, data_type in zip(offsets, formats)
    ):
        return None
    dtype = np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": offsets,
            "itemsize": point_step,
        }
    )
    data = memoryview(message.data)
    expected_row_step = point_step * width
    if row_step in (0, expected_row_step):
        required = expected_row_step * height
        if len(data) < required:
            return None
        structured = np.frombuffer(data, dtype=dtype, count=width * height)
    else:
        required = row_step * height
        if row_step < expected_row_step or len(data) < required:
            return None
        rows = [
            np.frombuffer(
                data,
                dtype=dtype,
                count=width,
                offset=row_index * row_step,
            )
            for row_index in range(height)
        ]
        structured = np.concatenate(rows) if rows else np.empty((0,), dtype=dtype)
    return np.column_stack(
        [structured[name] for name in names]
    ).astype(float, copy=False)


def image_to_gray_array(message, prefer_uint8=False):
    encoding = getattr(message, "encoding", "").lower()
    height = int(message.height)
    width = int(message.width)
    data = np.frombuffer(message.data, dtype=np.uint8)
    if height <= 0 or width <= 0:
        raise ValueError("image height and width must be positive")

    if encoding in ("mono8", "8uc1"):
        expected = height * width
        if data.size < expected:
            raise ValueError("image data is shorter than height*width")
        image = data[:expected].reshape((height, width))
        return image if prefer_uint8 else image.astype(np.float32)

    if encoding in ("rgb8", "bgr8"):
        expected = height * width * 3
        if data.size < expected:
            raise ValueError("image data is shorter than height*width*3")
        image_uint8 = data[:expected].reshape((height, width, 3))
        try:
            import cv2

            code = (
                cv2.COLOR_BGR2GRAY
                if encoding == "bgr8"
                else cv2.COLOR_RGB2GRAY
            )
            if prefer_uint8:
                return cv2.cvtColor(image_uint8, code)
            image = image_uint8.astype(np.float32)
            return cv2.cvtColor(image, code)
        except ImportError:
            image = image_uint8.astype(np.float32)
            if encoding == "bgr8":
                return (0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2]).astype(np.float32)
            return (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(np.float32)

    if encoding in ("rgba8", "bgra8"):
        expected = height * width * 4
        if data.size < expected:
            raise ValueError("image data is shorter than height*width*4")
        image_uint8 = data[:expected].reshape((height, width, 4))
        try:
            import cv2

            code = (
                cv2.COLOR_BGRA2GRAY
                if encoding == "bgra8"
                else cv2.COLOR_RGBA2GRAY
            )
            if prefer_uint8:
                return cv2.cvtColor(image_uint8, code)
            image = image_uint8.astype(np.float32)
            return cv2.cvtColor(image, code)
        except ImportError:
            image = image_uint8.astype(np.float32)
            if encoding == "bgra8":
                return (0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2]).astype(np.float32)
            return (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(np.float32)

    raise ValueError("unsupported image encoding: {}".format(encoding))


def image_to_bgr_array(message):
    encoding = getattr(message, "encoding", "").lower()
    height = int(message.height)
    width = int(message.width)
    data = np.frombuffer(message.data, dtype=np.uint8)
    if height <= 0 or width <= 0:
        raise ValueError("image height and width must be positive")

    if encoding in ("mono8", "8uc1"):
        expected = height * width
        if data.size < expected:
            raise ValueError("image data is shorter than height*width")
        gray = data[:expected].reshape((height, width))
        return np.stack((gray, gray, gray), axis=2)

    if encoding in ("rgb8", "bgr8"):
        expected = height * width * 3
        if data.size < expected:
            raise ValueError("image data is shorter than height*width*3")
        image = data[:expected].reshape((height, width, 3))
        if encoding == "rgb8":
            return image[:, :, ::-1].copy()
        return image.copy()

    if encoding in ("rgba8", "bgra8"):
        expected = height * width * 4
        if data.size < expected:
            raise ValueError("image data is shorter than height*width*4")
        image = data[:expected].reshape((height, width, 4))
        if encoding == "rgba8":
            return image[:, :, 2::-1].copy()
        return image[:, :, :3].copy()

    raise ValueError("unsupported image encoding: {}".format(encoding))


def transform_point_cloud_xyz(points_xyz, transform_name="none"):
    points = np.asarray(points_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape Nx3")

    if transform_name in ("", "none", "identity", None):
        return points

    if transform_name == "realsense_depth_to_body":
        transformed = np.empty_like(points)
        transformed[:, 0] = points[:, 2]
        transformed[:, 1] = -points[:, 0]
        transformed[:, 2] = -points[:, 1]
        return transformed

    raise ValueError("unsupported point cloud transform: {}".format(transform_name))
