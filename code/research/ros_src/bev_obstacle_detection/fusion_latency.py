"""Latency accounting for LiDAR-triggered multimodal fusion."""


def compute_fusion_latency(
    trigger_stamp_sec,
    ros_output_stamp_sec,
    callback_arrival_monotonic,
    processing_start_monotonic,
    processing_end_monotonic,
    configured_sync_wait_sec,
    selected_offsets_sec,
    budget_ms,
):
    arrival = float(callback_arrival_monotonic)
    start = float(processing_start_monotonic)
    end = float(processing_end_monotonic)
    if start < arrival:
        raise ValueError("processing start must not precede callback arrival")
    if end < start:
        raise ValueError("processing end must not precede processing start")
    configured_wait_ms = max(0.0, float(configured_sync_wait_sec) * 1000.0)
    budget_ms = float(budget_ms)
    if budget_ms <= 0.0:
        raise ValueError("latency budget must be positive")

    queue_wait_ms = (start - arrival) * 1000.0
    processing_ms = (end - start) * 1000.0
    total_compute_ms = (end - arrival) * 1000.0
    sensor_to_output_ms = (
        (float(ros_output_stamp_sec) - float(trigger_stamp_sec)) * 1000.0
    )
    ros_time_valid = sensor_to_output_ms >= 0.0
    if not ros_time_valid:
        sensor_to_output_ms = -1.0
    offsets = [
        abs(float(value))
        for value in selected_offsets_sec
        if value is not None
    ]
    return {
        "configured_sync_wait_ms": configured_wait_ms,
        "queue_wait_ms": queue_wait_ms,
        "contention_wait_ms": max(0.0, queue_wait_ms - configured_wait_ms),
        "processing_ms": processing_ms,
        "total_compute_ms": total_compute_ms,
        "sensor_to_output_ms": sensor_to_output_ms,
        "ros_time_valid": ros_time_valid,
        "max_selected_sync_offset_ms": (
            max(offsets) * 1000.0 if offsets else -1.0
        ),
        "available_modalities": len(offsets),
        "deadline_budget_ms": budget_ms,
        "deadline_met": total_compute_ms <= budget_ms,
    }
