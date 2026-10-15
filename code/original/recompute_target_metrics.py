#!/usr/bin/env python3
"""Portable rotated-BEV evaluation from frozen CSVs; no ROS, GPU, or training."""
import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

METHODS={"BEVFusion L":"bevfusion_lidar_raw_roadusers", "BEVFusion C+L":"bevfusion_camera_lidar_roadusers",
         "LiRaFusion L+R":"lirafusion_calibrated_roadusers", "Base LiDAR":"rgcf_base_lidar_roadusers",
         "Simple fusion":"SimpleGeometry_FrozenTransfer", "RGCF transfer*":"RGCF_FrozenTransfer_NoPose"}


def load_core(task):
    spec=importlib.util.spec_from_file_location("frozen_metric_core",task / "runtime_scripts/jei_s1_evaluator_core.py")
    core=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=core
    spec.loader.exec_module(core)
    return core


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--task",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args()
    task=a.task.resolve();out=a.out_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    core=load_core(task)
    with (task / "prepared/frame_manifest.csv").open() as f:
        domain=[str(int((Decimal(r["timestamp_ns"])/1000).to_integral_value(rounding=ROUND_HALF_UP))) for r in csv.DictReader(f)]
    assert len(domain)==len(set(domain))==597
    gt_path=task / "prepared/gt_v2_1_native_lidar.csv"
    gt,tokens=core.load_gt(gt_path)
    assert set(tokens).issubset(domain) and sum(map(len,gt.values()))==1890
    masks=json.loads((task / "semantic_masks_manifest.json").read_text())
    assert masks["empty_masks"]==16 and masks["masks_ok"]==597
    predictions={};hashes={};mapping={}
    for method,name in METHODS.items():
        path=task / "predictions" / (name+".csv")
        frames,unmapped,counts=core.load_detections(path,domain)
        assert not unmapped and set(counts).issubset({"sample_token_exact"})
        for boxes in frames.values():
            for b in boxes:
                assert np.isfinite([b.center_x,b.center_y,b.size_x,b.size_y,b.yaw,b.score]).all()
                assert min(b.size_x,b.size_y)>0
        predictions[method]=frames;hashes[name]=core.sha256_file(path);mapping[method]=counts
    aggregates=[];per_frame=[];per_pred=[];per_gt=[];distance=[]
    in_roi=lambda b:abs(b.center_x)<=51.2 and abs(b.center_y)<=51.2
    for scope,count in [("full_annotated_domain",1890),("common_xy_roi",1157)]:
        scope_gt={t:[b for b in boxes if scope=="full_annotated_domain" or in_roi(b)] for t,boxes in gt.items()}
        assert sum(map(len,scope_gt.values()))==count
        for method,frames in predictions.items():
            scoped={t:[b for b in boxes if scope=="full_annotated_domain" or in_roi(b)] for t,boxes in frames.items()}
            for threshold in [.1,.25,.5]:
                pdets,pgts,pframes=core.match_method(method,scope_gt,scoped,domain,threshold)
                tp,fp,fn=[int(pframes[k].sum()) for k in ["tp","fp","fn"]]
                assert tp+fn==count and tp+fp==sum(map(len,scoped.values()))
                ious=pdets.loc[pdets.status=="TP","matched_iou"].dropna().tolist()
                aggregates.append(dict(scope=scope,method=method,iou_threshold=threshold,**core.metrics(tp,fp,fn,ious,597)))
                distance.extend(dict(scope=scope,**r) for r in core.distance_summary(method,threshold,pdets,pgts,597))
                for d,collection in [(pdets,per_pred),(pgts,per_gt),(pframes,per_frame)]:
                    d["scope"]=scope;collection.append(d)
            print("Evaluated",scope,method,flush=True)
    aggregate=pd.DataFrame(aggregates)
    aggregate.to_csv(out / "aggregate_metrics.csv",index=False)
    full_frame=pd.concat(per_frame,ignore_index=True)
    assert len(aggregate)==36 and len(full_frame)==21492
    full_frame.to_csv(out / "per_frame_details.csv",index=False)
    pred=pd.concat(per_pred,ignore_index=True)
    pred.to_csv(out / "per_prediction_details.csv",index=False)
    pd.concat(per_gt,ignore_index=True).to_csv(out / "per_gt_details.csv",index=False)
    pd.DataFrame(distance).to_csv(out / "distance_metrics.csv",index=False)
    paired=[]
    for scope in aggregate.scope.unique():
        for threshold in [.1,.25,.5]:
            d=full_frame[(full_frame.scope==scope)&(full_frame.iou_threshold==threshold)]
            rgcf=d[d.method=="RGCF transfer*"].set_index("frame_key")
            for base in ["Base LiDAR","Simple fusion"]:
                b=d[d.method==base].set_index("frame_key")
                assert set(rgcf.index)==set(b.index)
                delta=rgcf[["tp","fp","fn"]]-b.loc[rgcf.index,["tp","fp","fn"]]
                for token,r in delta.iterrows():
                    paired.append(dict(scope=scope,iou_threshold=threshold,frame_key=token,reference=base,
                        delta_tp=int(r.tp),delta_fp=int(r.fp),delta_fn=int(r.fn)))
    pd.DataFrame(paired).to_csv(out / "paired_frame_deltas.csv",index=False)
    pd.DataFrame(paired).groupby(["scope","iou_threshold","reference"],as_index=False)[["delta_tp","delta_fp","delta_fn"]].sum().to_csv(out / "paired_aggregate_deltas.csv",index=False)
    counts=pred.groupby(["scope","method","iou_threshold","source","status"],dropna=False).size().reset_index(name="count")
    counts.to_csv(out / "candidate_source_matches.csv",index=False)
    audit=dict(status="pass",frames=597,gt_full=1890,gt_common_roi=1157,summary_rows=36,
        per_frame_rows=len(full_frame),empty_masks=16,exact_mapping=mapping,
        gt_sha256=core.sha256_file(gt_path),predictions_sha256=hashes,
        matching_core_sha256=core.sha256_file(task / "runtime_scripts/jei_s1_evaluator_core.py"),
        metric="Rotated BEV score/row-id/IoU ordered one-to-one matching; not official mAP or 3D AP",
        target_tuning=False,intervals_added=False,all_scopes_and_thresholds_preserved=True)
    (out / "evaluation_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(aggregate.to_string(index=False),flush=True)


if __name__=="__main__":
    main()
