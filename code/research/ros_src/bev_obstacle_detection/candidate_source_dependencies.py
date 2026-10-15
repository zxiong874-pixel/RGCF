"""Runtime dependency rules for external multimodal detector candidates."""


def candidate_source_dependencies(source):
    value = str(source or "").strip().lower().replace("-", "_")
    has_camera = "camera" in value or "semantic" in value
    has_lidar = "lidar" in value
    has_radar = "radar" in value or "lira" in value
    dependencies = set()
    if has_camera:
        dependencies.add("camera")
    if has_lidar:
        dependencies.add("lidar")
    if has_radar:
        dependencies.add("radar")
    return frozenset(dependencies)


def source_is_allowed(source, active_modalities, reject_unknown=True):
    dependencies = candidate_source_dependencies(source)
    if not dependencies:
        return not reject_unknown
    active = {str(name): bool(value) for name, value in active_modalities.items()}
    return all(active.get(name, False) for name in dependencies)


def filter_candidates_by_health(detections, active_modalities, reject_unknown=True):
    kept = []
    dropped = []
    for detection in detections:
        if source_is_allowed(
            getattr(detection, "source", ""),
            active_modalities,
            reject_unknown=reject_unknown,
        ):
            kept.append(detection)
        else:
            dropped.append(detection)
    return kept, dropped
