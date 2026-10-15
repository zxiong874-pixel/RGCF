"""Adapters for turning real segmentation model outputs into binary obstacle masks."""

from pathlib import Path

import numpy as np


DEFAULT_OBSTACLE_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
)


def parse_name_list(text):
    if text in ("", None):
        return []
    return [item.strip() for item in str(text).split(",") if item.strip()]


def parse_int_list(text):
    if text in ("", None):
        return []
    values = []
    for item in str(text).split(","):
        stripped = item.strip()
        if stripped:
            values.append(int(stripped))
    return values


def parse_float_tuple(text, expected_len=None):
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if expected_len is not None and len(values) != expected_len:
        raise ValueError("expected {} comma-separated values".format(expected_len))
    return tuple(values)


def postprocess_binary_mask(mask, min_area_px=0, dilation_pixels=0, close_pixels=0):
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2:
        raise ValueError("mask must be a 2D array")
    binary = np.where(binary > 0, 255, 0).astype(np.uint8)

    try:
        import cv2
    except ImportError:
        cv2 = None

    if min_area_px > 0 and cv2 is not None:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        filtered = np.zeros_like(binary)
        for index in range(1, count):
            if int(stats[index, cv2.CC_STAT_AREA]) >= int(min_area_px):
                filtered[labels == index] = 255
        binary = filtered

    if close_pixels > 0 and cv2 is not None:
        kernel_size = int(close_pixels) * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    if dilation_pixels > 0 and cv2 is not None:
        kernel_size = int(dilation_pixels) * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)

    return binary


def resize_mask(mask, width, height):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to resize segmentation masks") from exc

    return cv2.resize(np.asarray(mask, dtype=np.uint8), (int(width), int(height)), interpolation=cv2.INTER_NEAREST)


class UltralyticsSegmentationModel:
    def __init__(self, model_path, obstacle_classes=None, confidence=0.25, device=""):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. Install it or use --backend opencv_dnn with an ONNX model."
            ) from exc

        self.model_path = str(Path(model_path).expanduser())
        self.model = YOLO(self.model_path)
        self.accept_all_classes = any(str(name).strip().lower() == "all" for name in (obstacle_classes or ()))
        self.obstacle_classes = set(obstacle_classes or DEFAULT_OBSTACLE_CLASSES)
        self.confidence = float(confidence)
        self.device = device

    def predict_mask(self, image_bgr):
        height, width = image_bgr.shape[:2]
        kwargs = {"conf": self.confidence, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        results = self.model.predict(image_bgr, **kwargs)
        combined = np.zeros((height, width), dtype=np.uint8)
        if not results:
            return combined
        result = results[0]
        if getattr(result, "masks", None) is None or getattr(result, "boxes", None) is None:
            return combined
        names = getattr(result, "names", {}) or {}
        class_ids = result.boxes.cls.cpu().numpy().astype(int).tolist()
        masks = result.masks.data.cpu().numpy()
        for class_id, mask in zip(class_ids, masks):
            class_name = names.get(class_id, str(class_id))
            if not self.accept_all_classes and class_name not in self.obstacle_classes:
                continue
            combined = np.maximum(combined, resize_mask(mask > 0.5, width, height))
        return np.where(combined > 0, 255, 0).astype(np.uint8)


class OpenCVSemanticSegmentationModel:
    def __init__(
        self,
        model_path,
        obstacle_class_indices,
        input_width=512,
        input_height=512,
        scale=1.0 / 255.0,
        mean=(0.0, 0.0, 0.0),
        swap_rb=True,
    ):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for --backend opencv_dnn") from exc

        if not obstacle_class_indices:
            raise ValueError("--obstacle-class-indices is required for --backend opencv_dnn")
        self.cv2 = cv2
        self.net = cv2.dnn.readNet(str(Path(model_path).expanduser()))
        self.obstacle_class_indices = set(int(value) for value in obstacle_class_indices)
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.scale = float(scale)
        self.mean = tuple(float(value) for value in mean)
        self.swap_rb = bool(swap_rb)

    def predict_mask(self, image_bgr):
        height, width = image_bgr.shape[:2]
        blob = self.cv2.dnn.blobFromImage(
            image_bgr,
            scalefactor=self.scale,
            size=(self.input_width, self.input_height),
            mean=self.mean,
            swapRB=self.swap_rb,
            crop=False,
        )
        self.net.setInput(blob)
        output = self.net.forward()
        labels = labels_from_semantic_output(output)
        selected = np.isin(labels, list(self.obstacle_class_indices)).astype(np.uint8) * 255
        return resize_mask(selected, width, height)


def labels_from_semantic_output(output):
    array = np.asarray(output)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3:
        return np.argmax(array, axis=0).astype(np.int32)
    if array.ndim == 2:
        return array.astype(np.int32)
    raise ValueError("unsupported semantic segmentation output shape: {}".format(array.shape))


def create_segmenter(
    backend,
    model_path,
    obstacle_classes=None,
    obstacle_class_indices=None,
    confidence=0.25,
    device="",
    input_width=512,
    input_height=512,
    scale=1.0 / 255.0,
    mean=(0.0, 0.0, 0.0),
    swap_rb=True,
):
    normalized = str(backend).strip().lower()
    if normalized == "ultralytics":
        return UltralyticsSegmentationModel(
            model_path,
            obstacle_classes=obstacle_classes or DEFAULT_OBSTACLE_CLASSES,
            confidence=confidence,
            device=device,
        )
    if normalized == "opencv_dnn":
        return OpenCVSemanticSegmentationModel(
            model_path,
            obstacle_class_indices=obstacle_class_indices or (),
            input_width=input_width,
            input_height=input_height,
            scale=scale,
            mean=mean,
            swap_rb=swap_rb,
        )
    raise ValueError("unsupported segmentation backend: {}".format(backend))
