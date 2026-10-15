"""Small message cache and deterministic frame selection for sensor streams."""

from collections import deque, namedtuple


SynchronizedMessage = namedtuple(
    "SynchronizedMessage",
    ("message", "stamp", "offset_sec"),
)


class MonotonicStampGuard:
    """Reject duplicate or out-of-order fusion triggers."""

    def __init__(self, epsilon_sec=1e-9):
        epsilon_sec = float(epsilon_sec)
        if epsilon_sec < 0.0:
            raise ValueError("epsilon_sec must be non-negative")
        self.epsilon_sec = epsilon_sec
        self.last_accepted_stamp = None

    def accept(self, stamp_sec):
        stamp = float(stamp_sec)
        previous = self.last_accepted_stamp
        if previous is None:
            self.last_accepted_stamp = stamp
            return True, "accepted"
        if abs(stamp - previous) <= self.epsilon_sec:
            return False, "duplicate"
        if stamp < previous:
            return False, "out_of_order"
        self.last_accepted_stamp = stamp
        return True, "accepted"


class MessageCache:
    def __init__(self, max_age_sec=0.25, max_messages_per_modality=50):
        if max_age_sec <= 0.0:
            raise ValueError("max_age_sec must be positive")
        self.max_age_sec = float(max_age_sec)
        self.max_messages_per_modality = max(1, int(max_messages_per_modality))
        self._messages = {}
        self._history = {}

    def update(self, name, message, stamp_sec=None):
        if not name:
            raise ValueError("name must be non-empty")

        stamp = self._stamp_from_message(message) if stamp_sec is None else float(stamp_sec)
        self._messages[name] = (message, stamp)
        self._history.setdefault(
            name,
            deque(maxlen=self.max_messages_per_modality),
        ).append((message, stamp))

    def get(self, name):
        return self._messages.get(name, (None, None))[0]

    def stamp(self, name):
        return self._messages.get(name, (None, None))[1]

    def age(self, name, reference_stamp_sec):
        stamp = self.stamp(name)
        if stamp is None:
            return None
        return float(reference_stamp_sec) - stamp

    def is_ready(self, names, reference_stamp_sec):
        for name in names:
            age = self.age(name, reference_stamp_sec)
            if age is None or age < 0.0 or age > self.max_age_sec:
                return False
        return True

    def nearest(self, name, reference_stamp_sec, max_abs_age_sec=None):
        item = self._nearest_item(name, reference_stamp_sec, max_abs_age_sec)
        return item[0] if item is not None else None

    def nearest_stamp(self, name, reference_stamp_sec, max_abs_age_sec=None):
        item = self._nearest_item(name, reference_stamp_sec, max_abs_age_sec)
        return item[1] if item is not None else None

    def nearest_age(self, name, reference_stamp_sec, max_abs_age_sec=None):
        stamp = self.nearest_stamp(name, reference_stamp_sec, max_abs_age_sec)
        if stamp is None:
            return None
        return abs(float(reference_stamp_sec) - stamp)

    def is_ready_nearest(self, names, reference_stamp_sec, max_abs_age_sec=None):
        return all(
            self.nearest(name, reference_stamp_sec, max_abs_age_sec) is not None
            for name in names
        )

    def synchronized_snapshot(
        self,
        names,
        reference_stamp_sec,
        max_abs_age_sec=None,
    ):
        """Select each requested modality once against one reference timestamp."""

        snapshot = {}
        reference = float(reference_stamp_sec)
        for name in names:
            item = self._nearest_item(name, reference, max_abs_age_sec)
            if item is None:
                continue
            message, stamp = item
            snapshot[name] = SynchronizedMessage(
                message=message,
                stamp=float(stamp),
                offset_sec=abs(reference - float(stamp)),
            )
        return snapshot

    @staticmethod
    def snapshot_is_ready(snapshot, required_names):
        return all(name in snapshot for name in required_names)

    def snapshot(self):
        return dict(self._messages)

    def _nearest_item(self, name, reference_stamp_sec, max_abs_age_sec):
        history = self._history.get(name)
        if not history:
            return None
        reference = float(reference_stamp_sec)
        # A coalescing producer may refine the same timestamp more than once.
        # Reverse iteration makes the newest refinement win an exact tie.
        item = min(
            reversed(history),
            key=lambda value: (
                abs(value[1] - reference),
                value[1] > reference,
                -value[1],
            ),
        )
        limit = self.max_age_sec if max_abs_age_sec is None else float(max_abs_age_sec)
        if abs(item[1] - reference) > limit:
            return None
        return item

    @staticmethod
    def _stamp_from_message(message):
        try:
            return message.header.stamp.to_sec()
        except AttributeError as exc:
            raise ValueError("message must have header.stamp or stamp_sec must be provided") from exc
