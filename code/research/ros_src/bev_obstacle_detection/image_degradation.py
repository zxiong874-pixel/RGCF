"""Deterministic image degradation helpers for robustness tests."""

import numpy as np


CHANNELS_BY_ENCODING = {
    "mono8": 1,
    "8UC1": 1,
    "rgb8": 3,
    "bgr8": 3,
    "rgba8": 4,
    "bgra8": 4,
}


def validate_image_degradation_params(
    brightness_scale=1.0,
    gaussian_noise_std=0.0,
    blur_kernel=0,
    frame_drop_rate=0.0,
):
    if brightness_scale < 0.0:
        raise ValueError("brightness_scale must be non-negative")
    if gaussian_noise_std < 0.0:
        raise ValueError("gaussian_noise_std must be non-negative")
    if blur_kernel < 0:
        raise ValueError("blur_kernel must be non-negative")
    if blur_kernel not in (0, 1) and blur_kernel % 2 == 0:
        raise ValueError("blur_kernel must be 0, 1, or an odd positive integer")
    if frame_drop_rate < 0.0 or frame_drop_rate > 1.0:
        raise ValueError("frame_drop_rate must be in [0, 1]")


def _box_blur_uint8(image, kernel_size):
    if kernel_size in (0, 1):
        return image
    radius = kernel_size // 2
    source = image.astype(np.float32)
    if source.ndim == 2:
        padded = np.pad(source, ((radius, radius), (radius, radius)), mode="edge")
        blurred = np.zeros_like(source)
        for dy in range(kernel_size):
            for dx in range(kernel_size):
                blurred += padded[dy : dy + source.shape[0], dx : dx + source.shape[1]]
    else:
        padded = np.pad(source, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
        blurred = np.zeros_like(source)
        for dy in range(kernel_size):
            for dx in range(kernel_size):
                blurred += padded[dy : dy + source.shape[0], dx : dx + source.shape[1], :]
    blurred /= float(kernel_size * kernel_size)
    return np.clip(np.rint(blurred), 0, 255).astype(np.uint8)


def degrade_image_array(
    image,
    rng,
    brightness_scale=1.0,
    gaussian_noise_std=0.0,
    blur_kernel=0,
):
    validate_image_degradation_params(
        brightness_scale=brightness_scale,
        gaussian_noise_std=gaussian_noise_std,
        blur_kernel=blur_kernel,
        frame_drop_rate=0.0,
    )
    array = np.asarray(image, dtype=np.uint8)
    degraded = array.astype(np.float32) * float(brightness_scale)
    if gaussian_noise_std > 0.0:
        degraded += rng.normal(0.0, float(gaussian_noise_std), size=degraded.shape).astype(np.float32)
    degraded = np.clip(np.rint(degraded), 0, 255).astype(np.uint8)
    return _box_blur_uint8(degraded, int(blur_kernel))


def should_drop_image(rng, frame_drop_rate=0.0):
    validate_image_degradation_params(frame_drop_rate=frame_drop_rate)
    return frame_drop_rate > 0.0 and rng.random_sample() < float(frame_drop_rate)


def image_degradation_label(
    brightness_scale=1.0,
    gaussian_noise_std=0.0,
    blur_kernel=0,
    frame_drop_rate=0.0,
):
    parts = []
    if brightness_scale != 1.0:
        parts.append("brightness_{:.2f}".format(brightness_scale).replace(".", "p"))
    if gaussian_noise_std > 0.0:
        parts.append("image_noise_{:.1f}".format(gaussian_noise_std).replace(".", "p"))
    if blur_kernel not in (0, 1):
        parts.append("blur_{}".format(int(blur_kernel)))
    if frame_drop_rate > 0.0:
        parts.append("image_drop_{:.2f}".format(frame_drop_rate).replace(".", "p"))
    return "_".join(parts) if parts else "clean"


def image_message_to_array(message):
    encoding = message.encoding
    if encoding not in CHANNELS_BY_ENCODING:
        raise ValueError("unsupported image encoding: {}".format(encoding))
    channels = CHANNELS_BY_ENCODING[encoding]
    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    raw = np.frombuffer(message.data, dtype=np.uint8)
    if raw.size < height * step:
        raise ValueError("image data is shorter than height * step")
    rows = raw[: height * step].reshape((height, step))
    useful_width = width * channels
    if step < useful_width:
        raise ValueError("image step is smaller than width * channels")
    pixels = rows[:, :useful_width]
    if channels == 1:
        return pixels.reshape((height, width)).copy()
    return pixels.reshape((height, width, channels)).copy()


def array_to_image_message(message, image):
    degraded = np.asarray(image, dtype=np.uint8)
    output = message
    output.height = int(degraded.shape[0])
    output.width = int(degraded.shape[1])
    channels = 1 if degraded.ndim == 2 else int(degraded.shape[2])
    output.step = int(output.width * channels)
    output.data = degraded.reshape((-1,)).tobytes()
    output.is_bigendian = 0
    return output
