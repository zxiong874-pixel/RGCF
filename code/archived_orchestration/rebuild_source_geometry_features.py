#!/usr/bin/env python3
"""Rebuild yaw-dependent source labels/features after native provenance matching."""
from collections import defaultdict
from datetime import datetime
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

TASK = Path(__file__).resolve().parents[1]
ROOT = TASK.parents[1]
S3 = ROOT / "07_JEI实验补强/S3_因果与统计整改_20260903_190643"
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_boreas_detection_csv as evaluator
import build_boreas_candidate_feature_table as labels
from native_geometry_audit import dump, sha, write_csv


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_path(name):
    choices=sorted((TASK / "A_native_geometry").glob("*/" + name + "/native_geometry_audit.json"))
    if not choices:
        raise RuntimeError("Native job missing: " + name)
    path=choices[-1]
    report=json.loads(path.read_text())
    assert report["status"] == "pass"
    return path.parent / "canonical_detections.csv"


class NativeMatcher:
    def __init__(self,path):
        data=pd.read_csv(path,keep_default_na=False)
        self.path=path
        self.frames={str(int(k)):f for k,f in data.groupby("sample_token")}

    def match(self,row):
        token=str(int(round(float(row["timestamp"])*1e6)))
        frame=self.frames.get(token)
        if frame is None:
            raise ValueError("No raw native frame for " + token)
        fields=["center_x","center_y","size_x","size_y"]
        delta=np.max(np.abs(frame[fields].to_numpy(dtype=float)-np.array([float(row[f]) for f in fields])),axis=1)
        score=np.abs(frame.score.to_numpy(dtype=float)-float(row["score"]))
        angular=(frame.tensor_yaw.to_numpy(dtype=float)-float(row["yaw"])+math.pi)%(2*math.pi)-math.pi
        valid=np.flatnonzero((delta<=0.000101)&(score<=1.1e-6)&(np.abs(angular)<=1.1e-5))
        if len(valid)!=1:
            raise ValueError("Ambiguous/missing native box: " + str((token,row.get("candidate_id",row.get("obstacle_id")),len(valid),float(delta.min()))))
        native=frame.iloc[int(valid[0])]
        return native, float(delta[valid[0]]),float(score[valid[0]]),float(abs(angular[valid[0]]))


def main():
    paths={"camera_lidar":native_path("original_camera_lidar"),
           "radar_lirafusion":native_path("original_lidar_radar"),
           "lidar":native_path("original_lidar")}
    matchers={key:NativeMatcher(value) for key,value in paths.items()}
    out=TASK / ("B_source_rebuild/geometry_and_features_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True,exist_ok=False)
    old_path=S3 / "features/candidates_causal_full_support.csv"
    frame=pd.read_csv(old_path,keep_default_na=False)
    rows=frame.to_dict("records")
    original=[dict(r) for r in rows]
    provenance=[]
    for row in rows:
        native, error, score_error, yaw_error=matchers[row["source"]].match(row)
        provenance.append(dict(candidate_id=row["candidate_id"],timestamp=row["timestamp"],
            source=row["source"],split=row["split"],raw_box_index=int(native.raw_box_index),
            matched_native_csv=str(paths[row["source"]]),max_xy_dim_rounding_error=error,
            score_rounding_error=score_error,tensor_yaw_rounding_error=yaw_error))
        row["yaw"]=float(native.yaw)
    write_csv(out / "candidate_native_provenance.csv",provenance)
    print("All source feature candidates linked to native predictions",len(rows),flush=True)
    gt_path=TASK / "A_native_geometry/gt_pyboreas/boreas_train500_gt_canonical.csv"
    gt=evaluator.read_boxes(gt_path,is_gt=True)
    # The archived matcher accepts strings, but label generation expects floats.
    for r in rows:
        r["timestamp"]="%.6f"%float(r["timestamp"])
    labels.add_gt_labels(rows,gt)
    support=load_module("s3_causal_support",S3 / "scripts/build_causal_candidate_pool.py")
    groups=defaultdict(list)
    for row in rows:
        groups[row["timestamp"]].append(row)
    for index,items in enumerate(groups.values()):
        references={s:[r for r in items if r["source"]==s] for s in paths if s!="lidar"}
        for row in items:
            for source,prefix,total in [("camera_lidar","camera","camera_semantic_support"),("radar_lirafusion","radar","radar_support")]:
                value,iou,score,center=support.best_support(row,references[source])
                row[total]="%.6f"%value
                row[prefix+"_support_iou"]="%.6f"%iou
                row[prefix+"_support_score"]="%.6f"%score
                row[prefix+"_support_center_dist"]="%.4f"%center
        if index%100==0:
            print("Recomputed labels/cross-source IoU",index,flush=True)
    pd.DataFrame(rows).to_csv(out / "candidates_causal_full_support_canonical.csv",index=False)
    changes=[]
    summary=[]
    split_manifest=pd.read_csv(S3 / "manifests/split_manifest.csv",keep_default_na=False)
    split_lookup={str(r.frame_key):r.split for r in split_manifest.itertuples()}
    assert split_manifest["split"].value_counts().to_dict()==dict(model_development=250,calibration=100,holdout_test=150)
    for old,new in zip(original,rows):
        assert old["candidate_id"]==new["candidate_id"]
        assert split_lookup[str(int(round(float(new["timestamp"])*1e6)))]==new["split"]
        changed=[]
        for key in frame.columns:
            try:
                equal=abs(float(old[key])-float(new[key]))<=1e-12
            except (ValueError,TypeError):
                equal=str(old[key])==str(new[key])
            if not equal: changed.append(key)
        changes.append(dict(frame_id=str(int(round(float(new["timestamp"])*1e6))),
            candidate_id=new["candidate_id"],source=new["source"],split=new["split"],
            old_yaw=old["yaw"],new_yaw=new["yaw"],size_x=new["size_x"],size_y=new["size_y"],
            old_best_iou=old["best_gt_iou"],new_best_iou=new["best_gt_iou"],
            old_assigned_gt=old["best_gt_index"],new_assigned_gt=new["best_gt_index"],
            old_label=old["label_iou10"],new_label=new["label_iou10"],
            old_label25=old["label_iou25"],new_label25=new["label_iou25"],
            old_label50=old["label_iou50"],new_label50=new["label_iou50"],
            changed_fields=";".join(changed),reason="native yaw convention and GT geometry correction"))
    write_csv(out / "candidate_before_after_geometry_labels.csv",changes)
    change=pd.DataFrame(changes)
    for (source,split),g in change.groupby(["source","split"]):
        summary.append(dict(source=source,split=split,candidates=len(g),
            old_positive=int(g.old_label.sum()),new_positive=int(g.new_label.sum()),
            label_flips=int((g.old_label!=g.new_label).sum()),
            label25_flips=int((g.old_label25!=g.new_label25).sum()),
            label50_flips=int((g.old_label50!=g.new_label50).sum())))
    write_csv(out / "label_change_summary.csv",summary)
    # Trace every retained historical LiDAR candidate to its original SDK object.
    base_path=ROOT / "results/boreas_compare/bevfusion_lidar_train500_iou0p10/bevfusion_boreas_train500_detections.csv"
    base=pd.read_csv(base_path,keep_default_na=False)
    base_provenance=[]
    for index,row in base.iterrows():
        native,error,se,ye=matchers["lidar"].match(row.to_dict())
        base_provenance.append(dict(row_index=index,timestamp=row.timestamp,
            obstacle_id=row.obstacle_id,raw_box_index=int(native.raw_box_index),
            max_xy_dim_rounding_error=error,score_rounding_error=se,tensor_yaw_rounding_error=ye))
        base.at[index,"yaw"]=float(native.yaw)
    base.to_csv(out / "bevfusion_lidar_canonical_historical_precision.csv",index=False)
    write_csv(out / "base_native_provenance.csv",base_provenance)
    unchanged_fields=[k for k in frame.columns if all(k not in r["changed_fields"].split(";") for r in changes)]
    report=dict(status="pass",source_candidates=len(rows),base_candidates=len(base),
        identity_policy="same integer-us frame, unique match with max x/y/dx/dy error<=0.000101m, score<=1.1e-6, tensor_yaw<=1.1e-5rad",
        source_pool_identities_and_order_unchanged=True,source_centers_dimensions_scores_unchanged=True,
        changed_features="yaw; GT labels/matches; cross-source IoU support",
        unchanged_fields=unchanged_fields,match_provenance=provenance[:1],label_summary=summary,
        unchanged_feature_basis="Raw Camera/mask support uses projected center/range ROI; Navtech support uses center/range/max extent; proxy pose support uses centers. Source causal membership and temporal reset/offsets remain frozen.",
        limitations=["This audit rebuilds inputs only; refitting and calibration isolation require subsequent validation."])
    dump(out / "rebuild_audit.json",report)
    inputs=[old_path,base_path,gt_path,S3 / "manifests/split_manifest.csv",*paths.values(),Path(labels.__file__),Path(support.__file__)]
    dump(out / "input_sha256.json",{str(p):sha(p) for p in inputs})
    dump(out / "output_sha256.json",{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
