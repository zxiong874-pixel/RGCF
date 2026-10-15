#!/usr/bin/env python3
"""Replay the two declared LiDAR thresholds on the same four archived pools."""
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import run_supervisor_supplementary_cd as legacy
from native_geometry_audit import dump, sha, write_csv
from bev_obstacle_detection.detection_metrics import greedy_match_frame


def main():
    out = Path(__file__).resolve().parents[1] / ("D_gating/four_state_replay_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    config = legacy.load_gt_config(str(legacy.GT_CONFIG))
    hashes = {str(legacy.GT_CONFIG): sha(legacy.GT_CONFIG),
              str(Path(legacy.__file__)): sha(legacy.__file__)}
    metrics, deltas, frame_rows, gate_rows = [], [], [], []
    for scenario, spec in legacy.SCENARIOS.items():
        print("Reading archived state:", scenario, flush=True)
        destination = out / scenario
        destination.mkdir()
        quality = pd.read_csv(spec["quality"]).sort_values("timestamp").reset_index(drop=True)
        rows = legacy.read_rows(spec["pool"])
        times = quality.timestamp.to_numpy(dtype=float)
        gt = legacy.gt_for_times(spec["bag"], times, config)
        assert len(gt) == len(quality) and sum(map(len, gt.values())) > 0
        assert set(rows.timestamp).issubset(set(times)), "Candidate timestamp outside frame manifest"
        write_csv(destination / "gt.csv", [asdict(b) for boxes in gt.values() for b in boxes])
        quality.to_csv(destination / "frame_quality.csv", index=False)
        for key in ["bag", "quality", "pool"]:
            path = spec[key]
            if str(path) not in hashes:
                hashes[str(path)] = sha(path)
        summaries = {}
        for threshold in [.12,.20]:
            predictions, stats, selected = legacy.replay_pool(rows, quality, threshold, .12, .10)
            selected.to_csv(destination / ("predictions_qL_" + str(threshold) + ".csv"), index=False)
            values = legacy.evaluate_detection_frames(predictions, gt, .10)
            summaries[threshold] = values
            metrics.append(dict(scenario=scenario, qLmax=threshold, qRmax=.12,
                qC_image_min=.10,qC_semantic_min=.10,frames=len(quality), **values))
            pool_by_time = {float(t):f for t,f in rows.groupby("timestamp")}
            for timestamp in times:
                timestamp=float(timestamp)
                tp,fp,fn,ious=greedy_match_frame(predictions.get(timestamp,[]),gt[timestamp],.10)
                frame_rows.append(dict(scenario=scenario, timestamp=timestamp,qLmax=threshold,
                    tp=tp,fp=fp,fn=fn,gt=len(gt[timestamp]),predictions=len(predictions.get(timestamp,[]))))
                q=quality.loc[quality.timestamp == timestamp].iloc[0]
                frame=pool_by_time.get(timestamp, rows.iloc[:0])
                nr=int((frame.source_group == "radar_surface").sum())
                nc=int((frame.source_group == "camera_semantic").sum())
                r=bool(q.lidar_quality <= threshold and nr)
                qi=float(q.get("camera_raw_image_quality",q.camera_quality))
                qs=float(q.get("camera_semantic_quality",0.0))
                c=bool(q.lidar_quality <= threshold and q.radar_quality <= .12 and
                       qi >= .1 and qs >= .1 and nc)
                gate_rows.append(dict(scenario=scenario,timestamp=timestamp,qLmax=threshold,
                    qL=q.lidar_quality,qR=q.radar_quality,qC_image=qi,
                    qC_semantic=qs,camera_quality_uses_legacy_default="camera_semantic_quality" not in q.index,
                    radar_pool_count=nr,camera_pool_count=nc,
                    radar_candidates_admitted=r,camera_candidates_admitted=c,both_admitted=r and c))
        deltas.append(dict(scenario=scenario, **{"delta_"+k:summaries[.20][k]-summaries[.12][k]
            for k in ["tp","fp","fn","precision","recall","f1","mean_iou"]}))
    write_csv(out / "four_state_metrics.csv",metrics)
    write_csv(out / "four_state_delta_0p20_minus_0p12.csv",deltas)
    write_csv(out / "per_frame_metrics.csv",frame_rows)
    write_csv(out / "per_frame_candidate_admission.csv",gate_rows)
    dump(out / "input_sha256.json",hashes)
    report=dict(status="pass",scenario_count=4,settings=[.12,.20],qRmax=.12,qCmin=.10,
        candidate_geometry_and_scores_unchanged=True, gt_from="archived Gazebo model_states at original frame timestamps",
        metrics="Historical XTDrone axis-aligned Box2D IoU greedy maximum-overlap matching; not Boreas rotated BEV protocol",
        scope="Fixed post-generation candidate pool replay, not rerun online extraction/merge/adaptation or VINS",
        activation_caveat="Candidate admission rates require an existing candidate and are NOT gate-dispatch rates",
        legacy_missing_camera_columns="Camera dropout archive lacks raw/semantic quality; use original replay fallback camera_quality and semantic=0, not fabricated measurements",
        empty_frame_preservation=True, no_new_threshold_search=True,production_code_modified=False,
        qL_0p20_identity="Previously selected replay configuration, not silently substituted for nominal 0.12",
        deltas=deltas)
    dump(out / "replay_audit.json",report)
    dump(out / "output_sha256.json",{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob("*")) if p.is_file()})
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()
