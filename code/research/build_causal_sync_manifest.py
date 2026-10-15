#!/usr/bin/env python3
"""Build a timestamp-causal Boreas sensor association manifest."""

import argparse
import csv
import json
from bisect import bisect_right
from pathlib import Path


def timestamps_in(directory):
    values = []
    for path in Path(directory).glob("*.png"):
        if path.stem.isdigit():
            values.append(int(path.stem))
    return sorted(values)


def latest_not_after(anchor, values, max_lag):
    index = bisect_right(values, anchor) - 1
    if index < 0:
        return None
    value = values[index]
    if anchor - value > max_lag:
        return None
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--legacy-sync-manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--out-split-manifest", required=True)
    parser.add_argument("--out-audit-csv", required=True)
    parser.add_argument("--camera-max-lag-us", type=int, default=60000)
    parser.add_argument("--radar-max-lag-us", type=int, default=140000)
    parser.add_argument("--segment-gap-us", type=int, default=1000000)
    args = parser.parse_args()

    with Path(args.split_manifest).open(encoding="utf-8", newline="") as stream:
        split_rows = list(csv.DictReader(stream))
    split_rows.sort(key=lambda row: int(row["frame_order"]))

    legacy = json.loads(Path(args.legacy_sync_manifest).read_text(encoding="utf-8"))
    legacy_by_anchor = {
        int(item["lidar_timestamp_us"]): item.get("matches", {})
        for item in legacy.get("frames", [])
    }
    data_root = Path(args.data_root)
    available = {
        "camera": timestamps_in(data_root / "camera"),
        "radar": timestamps_in(data_root / "radar"),
    }
    max_lag = {"camera": args.camera_max_lag_us, "radar": args.radar_max_lag_us}

    frames = []
    audit_rows = []
    previous_anchor = None
    segment_id = -1
    for row in split_rows:
        anchor = int(row["frame_key"])
        if previous_anchor is None or anchor - previous_anchor > args.segment_gap_us:
            segment_id += 1
        previous_anchor = anchor
        matches = {}
        legacy_matches = legacy_by_anchor.get(anchor, {})
        audit = {
            "frame_key": str(anchor),
            "frame_order": int(row["frame_order"]),
            "split": row["split"],
            "segment_id": segment_id,
        }
        for sensor in ("camera", "radar"):
            selected = latest_not_after(anchor, available[sensor], max_lag[sensor])
            legacy_match = legacy_matches.get(sensor) or {}
            legacy_timestamp = legacy_match.get("timestamp_us")
            legacy_delta = (
                int(legacy_timestamp) - anchor if legacy_timestamp is not None else None
            )
            audit[f"legacy_{sensor}_timestamp_us"] = legacy_timestamp or ""
            audit[f"legacy_{sensor}_delta_us"] = legacy_delta if legacy_delta is not None else ""
            audit[f"legacy_{sensor}_causal"] = int(
                legacy_delta is not None and legacy_delta <= 0
            )
            if selected is None:
                audit[f"causal_{sensor}_timestamp_us"] = ""
                audit[f"causal_{sensor}_delta_us"] = ""
                audit[f"causal_{sensor}_available"] = 0
                continue
            delta = selected - anchor
            matches[sensor] = {
                "key": f"boreas-objects-v1/{sensor}/{selected}.png",
                "timestamp_us": selected,
                "delta_us": delta,
                "causal_association": True,
            }
            audit[f"causal_{sensor}_timestamp_us"] = selected
            audit[f"causal_{sensor}_delta_us"] = delta
            audit[f"causal_{sensor}_available"] = 1
        frames.append(
            {
                "lidar_timestamp_us": anchor,
                "frame_order": int(row["frame_order"]),
                "split": row["split"],
                "segment_id": segment_id,
                "matches": matches,
            }
        )
        audit_rows.append(audit)

    payload = {
        "format_version": 2,
        "association_policy": "latest_local_sensor_timestamp_not_after_lidar_anchor",
        "timestamp_causality_only": True,
        "arrival_timestamps_available": False,
        "camera_max_lag_us": args.camera_max_lag_us,
        "radar_max_lag_us": args.radar_max_lag_us,
        "segment_gap_us": args.segment_gap_us,
        "available_files": {key: len(value) for key, value in available.items()},
        "frames": frames,
    }
    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    split_fields = list(split_rows[0]) + ["segment_id", "segment_frame_order"]
    segment_orders = {}
    enriched = []
    for row, audit in zip(split_rows, audit_rows):
        segment = audit["segment_id"]
        order = segment_orders.get(segment, 0)
        segment_orders[segment] = order + 1
        enriched.append({**row, "segment_id": segment, "segment_frame_order": order})
    with Path(args.out_split_manifest).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=split_fields)
        writer.writeheader()
        writer.writerows(enriched)
    with Path(args.out_audit_csv).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    for sensor in ("camera", "radar"):
        available_count = sum(row[f"causal_{sensor}_available"] for row in audit_rows)
        legacy_future = sum(
            bool(row[f"legacy_{sensor}_delta_us"] != "" and row[f"legacy_{sensor}_delta_us"] > 0)
            for row in audit_rows
        )
        print(f"{sensor}: causal_available={available_count}/500 legacy_future={legacy_future}/500")
    if any(
        row[key] != "" and int(row[key]) > 0
        for row in audit_rows
        for key in ("causal_camera_delta_us", "causal_radar_delta_us")
    ):
        raise AssertionError("future sensor association survived causal manifest construction")


if __name__ == "__main__":
    main()
