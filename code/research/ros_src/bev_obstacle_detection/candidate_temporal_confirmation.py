"""Temporal confirmation and hysteresis for adaptive BEV candidates."""

from dataclasses import dataclass, replace

import numpy as np


@dataclass
class _CandidateTrack:
    track_id: int
    x: float
    y: float
    timestamp: float
    coordinate_mode: str
    ever_confirmed: bool
    unconfirmed_streak: int


def _candidate_position(obstacle, world_from_body):
    if world_from_body is None:
        return float(obstacle.center_x), float(obstacle.center_y), "body"
    transform = np.asarray(world_from_body, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("world_from_body must be a finite 4x4 transform")
    body = np.array(
        [float(obstacle.center_x), float(obstacle.center_y), 0.0, 1.0],
        dtype=float,
    )
    world = transform.dot(body)
    return float(world[0]), float(world[1]), "world"


class CandidateTemporalConfirmationTracker:
    """Associate candidates and suppress persistently unconfirmed occupancy."""

    def __init__(
        self,
        match_distance=1.0,
        grace_frames=1,
        max_unconfirmed_frames=3,
        unconfirmed_decay=0.75,
        max_track_age_sec=1.0,
        min_active_modalities=1,
        require_initial_confirmation=True,
    ):
        self.match_distance = float(match_distance)
        self.grace_frames = int(grace_frames)
        self.max_unconfirmed_frames = int(max_unconfirmed_frames)
        self.unconfirmed_decay = float(unconfirmed_decay)
        self.max_track_age_sec = float(max_track_age_sec)
        self.min_active_modalities = int(min_active_modalities)
        self.require_initial_confirmation = bool(require_initial_confirmation)
        if self.match_distance <= 0.0:
            raise ValueError("match_distance must be positive")
        if self.grace_frames < 0:
            raise ValueError("grace_frames must be non-negative")
        if self.max_unconfirmed_frames < self.grace_frames:
            raise ValueError(
                "max_unconfirmed_frames must be at least grace_frames"
            )
        if self.unconfirmed_decay < 0.0 or self.unconfirmed_decay > 1.0:
            raise ValueError("unconfirmed_decay must be in [0, 1]")
        if self.max_track_age_sec <= 0.0:
            raise ValueError("max_track_age_sec must be positive")
        if self.min_active_modalities < 1:
            raise ValueError("min_active_modalities must be positive")
        self._tracks = {}
        self._next_track_id = 1
        self._last_timestamp = None

    def clear(self):
        self._tracks.clear()
        self._last_timestamp = None

    def _prune(self, timestamp):
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if timestamp - track.timestamp <= self.max_track_age_sec
            and timestamp >= track.timestamp
        }

    def _matches(self, positions):
        candidates = []
        try:
            from scipy.spatial import cKDTree

            tracks_by_mode = {}
            for track_id, track in self._tracks.items():
                tracks_by_mode.setdefault(track.coordinate_mode, []).append(
                    (track_id, track)
                )
            trees = {}
            for mode, tracks in tracks_by_mode.items():
                tracks.sort(key=lambda item: item[0])
                coordinates = np.asarray(
                    [(track.x, track.y) for _, track in tracks],
                    dtype=float,
                )
                trees[mode] = (tracks, cKDTree(coordinates))
            for candidate_index, (x, y, mode) in enumerate(positions):
                item = trees.get(mode)
                if item is None:
                    continue
                tracks, tree = item
                neighbor_indices = tree.query_ball_point(
                    (x, y),
                    self.match_distance,
                )
                for neighbor_index in neighbor_indices:
                    track_id, track = tracks[int(neighbor_index)]
                    distance = float(
                        np.hypot(x - track.x, y - track.y)
                    )
                    if distance <= self.match_distance:
                        candidates.append(
                            (distance, candidate_index, track_id)
                        )
        except ImportError:
            for candidate_index, (x, y, mode) in enumerate(positions):
                for track_id, track in self._tracks.items():
                    if track.coordinate_mode != mode:
                        continue
                    distance = float(np.hypot(x - track.x, y - track.y))
                    if distance <= self.match_distance:
                        candidates.append(
                            (distance, candidate_index, track_id)
                        )
        candidates.sort()
        matched_candidates = set()
        matched_tracks = set()
        matches = {}
        for _, candidate_index, track_id in candidates:
            if (
                candidate_index in matched_candidates
                or track_id in matched_tracks
            ):
                continue
            matched_candidates.add(candidate_index)
            matched_tracks.add(track_id)
            matches[candidate_index] = track_id
        return matches

    def update(
        self,
        obstacles,
        diagnostics,
        timestamp,
        world_from_body=None,
        allow_initial_unconfirmed=False,
    ):
        """Return temporally accepted obstacles and augmented diagnostics."""
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            self.clear()
        self._last_timestamp = timestamp
        self._prune(timestamp)

        diagnostic_by_id = {
            int(row["obstacle_id"]): dict(row) for row in diagnostics
        }
        positions = [
            _candidate_position(obstacle, world_from_body)
            for obstacle in obstacles
        ]
        matches = self._matches(positions)
        accepted = []
        updated_by_id = {}

        for index, (obstacle, position) in enumerate(zip(obstacles, positions)):
            obstacle_id = int(obstacle.obstacle_id)
            diagnostic = diagnostic_by_id.get(
                obstacle_id,
                {
                    "obstacle_id": obstacle_id,
                    "active_modality_count": 0,
                    "accepted": True,
                },
            )
            active_count = int(diagnostic.get("active_modality_count", 0))
            currently_confirmed = active_count >= self.min_active_modalities
            matched_track = self._tracks.get(matches.get(index))
            ever_confirmed = currently_confirmed or (
                matched_track.ever_confirmed
                if matched_track is not None
                else False
            )
            unconfirmed_streak = (
                0
                if currently_confirmed
                else (
                    matched_track.unconfirmed_streak + 1
                    if matched_track is not None
                    else 1
                )
            )
            initial_ok = (
                ever_confirmed
                or not self.require_initial_confirmation
                or bool(allow_initial_unconfirmed)
            )
            temporal_keep = (
                initial_ok
                and unconfirmed_streak <= self.max_unconfirmed_frames
            )
            decay_frames = max(
                0,
                unconfirmed_streak - self.grace_frames,
            )
            score_scale = (
                self.unconfirmed_decay**decay_frames
                if not currently_confirmed
                else 1.0
            )

            track_id = (
                matched_track.track_id
                if matched_track is not None
                else self._next_track_id
            )
            if matched_track is None:
                self._next_track_id += 1
            x, y, coordinate_mode = position
            if temporal_keep or currently_confirmed:
                self._tracks[track_id] = _CandidateTrack(
                    track_id=track_id,
                    x=x,
                    y=y,
                    timestamp=timestamp,
                    coordinate_mode=coordinate_mode,
                    ever_confirmed=ever_confirmed,
                    unconfirmed_streak=unconfirmed_streak,
                )
            elif matched_track is not None:
                self._tracks.pop(track_id, None)

            temporal_score = float(
                np.clip(float(obstacle.score) * score_scale, 0.0, 1.0)
            )
            diagnostic.update(
                {
                    "accepted": bool(
                        diagnostic.get("accepted", True) and temporal_keep
                    ),
                    "temporal_track_id": track_id,
                    "temporal_confirmed": currently_confirmed,
                    "temporal_ever_confirmed": ever_confirmed,
                    "temporal_unconfirmed_streak": unconfirmed_streak,
                    "temporal_score_scale": score_scale,
                    "temporal_coordinate_mode": coordinate_mode,
                    "temporal_accepted": temporal_keep,
                    "temporal_failsafe": bool(
                        allow_initial_unconfirmed and not ever_confirmed
                    ),
                }
            )
            updated_by_id[obstacle_id] = diagnostic
            if temporal_keep:
                accepted.append(replace(obstacle, score=temporal_score))

        updated_diagnostics = []
        for row in diagnostics:
            obstacle_id = int(row["obstacle_id"])
            if obstacle_id in updated_by_id:
                updated_diagnostics.append(updated_by_id[obstacle_id])
                continue
            diagnostic = dict(row)
            diagnostic.update(
                {
                    "temporal_track_id": -1,
                    "temporal_confirmed": False,
                    "temporal_ever_confirmed": False,
                    "temporal_unconfirmed_streak": 0,
                    "temporal_score_scale": 0.0,
                    "temporal_coordinate_mode": "none",
                    "temporal_accepted": False,
                    "temporal_failsafe": False,
                }
            )
            updated_diagnostics.append(diagnostic)
        return tuple(accepted), tuple(updated_diagnostics)
