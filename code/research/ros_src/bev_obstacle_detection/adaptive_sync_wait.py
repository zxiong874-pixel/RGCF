"""Bounded feedback controller for multimodal synchronization wait time."""


def snapshot_ready_for_early_processing(
    snapshot,
    modalities,
    reference_stamp_sec,
    max_past_offset_sec,
):
    reference = float(reference_stamp_sec)
    max_past = float(max_past_offset_sec)
    if max_past < 0.0:
        raise ValueError("maximum past offset must be non-negative")
    for modality in modalities:
        item = snapshot.get(modality)
        if item is None:
            return False
        signed_offset = float(item.stamp) - reference
        if signed_offset < 0.0 and abs(signed_offset) > max_past:
            return False
    return True


class AdaptiveSyncWaitController:
    def __init__(
        self,
        initial_wait_sec,
        min_wait_sec,
        max_wait_sec,
        increase_step_sec=0.005,
        decrease_step_sec=0.002,
        complete_frames_before_decrease=3,
    ):
        self.min_wait_sec = float(min_wait_sec)
        self.max_wait_sec = float(max_wait_sec)
        self.increase_step_sec = float(increase_step_sec)
        self.decrease_step_sec = float(decrease_step_sec)
        self.complete_frames_before_decrease = int(
            complete_frames_before_decrease
        )
        initial = float(initial_wait_sec)
        if self.min_wait_sec < 0.0:
            raise ValueError("minimum wait must be non-negative")
        if self.max_wait_sec < self.min_wait_sec:
            raise ValueError("maximum wait must be at least minimum wait")
        if not self.min_wait_sec <= initial <= self.max_wait_sec:
            raise ValueError("initial wait must be within wait bounds")
        if self.increase_step_sec <= 0.0:
            raise ValueError("increase step must be positive")
        if self.decrease_step_sec <= 0.0:
            raise ValueError("decrease step must be positive")
        if self.complete_frames_before_decrease <= 0:
            raise ValueError(
                "complete frame threshold must be positive"
            )
        self.current_wait_sec = initial
        self.consecutive_complete_frames = 0

    def observe(self, inputs_complete):
        if not bool(inputs_complete):
            self.consecutive_complete_frames = 0
            self.current_wait_sec = min(
                self.max_wait_sec,
                self.current_wait_sec + self.increase_step_sec,
            )
            return self.current_wait_sec

        self.consecutive_complete_frames += 1
        if (
            self.consecutive_complete_frames
            >= self.complete_frames_before_decrease
        ):
            self.current_wait_sec = max(
                self.min_wait_sec,
                self.current_wait_sec - self.decrease_step_sec,
            )
            self.consecutive_complete_frames = 0
        return self.current_wait_sec
