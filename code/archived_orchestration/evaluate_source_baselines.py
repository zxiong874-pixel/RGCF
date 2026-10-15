#!/usr/bin/env python3
"""Compare canonical source outputs using the same frozen final metric core."""
import json
from pathlib import Path

import pandas as pd

from rebuild_source_geometry_features import ROOT,TASK,S3,native_path,NativeMatcher
from recompute_target_metrics import load_core
from native_geometry_audit import sha,dump


def main():
    out=TASK / "B_source_rebuild/common_baseline_evaluation"
    out.mkdir(exist_ok=False)
    pred_dir=out / "predictions";pred_dir.mkdir()
    refit=Path(json.loads((TASK / "B_source_rebuild/impact_audit/impact_audit.json").read_text())["source_refit"])
    # Preserve exact historical C+L CSV fields and rounding except native yaw.
    cl_path=ROOT / "results/supplementary_experiments_20260812/inputs/bevfusion_boreas_train500_camera_lidar_detections.csv"
    data=pd.read_csv(cl_path,keep_default_na=False)
    matcher=NativeMatcher(native_path("original_camera_lidar"))
    for index,row in data.iterrows():
        native,_,_,_=matcher.match(row.to_dict());data.at[index,"yaw"]=float(native.yaw)
    cl_new=pred_dir / "bevfusion_camera_lidar_canonical.csv";data.to_csv(cl_new,index=False)
    lira=pd.read_csv(native_path("stage28_clean_lirafusion_lidar_radar"))
    threshold=float(pd.read_csv(refit / "frozen_source_assets/lirafusion_calibrated_threshold.csv").threshold.iloc[0])
    lira_new=pred_dir / "lirafusion_calibrated_canonical.csv"
    lira[lira.score>=threshold].to_csv(lira_new,index=False)
    methods={"BEVFusion L":native_path("stage28_clean_bevfusion_lidar"),"BEVFusion C+L":cl_new,
        "LiRaFusion L+R":lira_new,"Base LiDAR":refit/"base_calibration/boreas_train500_class_adaptive_detections.csv",
        "Simple fusion":refit/"frozen_source_assets/SimpleGeometryCausal_detections.csv",
        "RGCF":refit/"frozen_source_assets/RGCF_B2_Causal_detections.csv"}
    core=load_core(TASK / "C_target_recheck/kradar_sequence1")
    gt_path=TASK / "A_native_geometry/gt_pyboreas/boreas_train500_gt_canonical.csv"
    gt,domain=core.load_gt(gt_path)
    manifest=pd.read_csv(S3/"manifests/split_manifest.csv",dtype=str)
    splits={k:list(g.frame_key) for k,g in manifest.groupby("split")}
    splits["all500"]=domain
    all_metrics=[];frame_details=[];pred_details=[];gt_details=[];distance=[]
    for name,path in methods.items():
        predictions,unmapped,mapping=core.load_detections(path,domain)
        assert not unmapped
        for iou in [.1,.25,.5]:
            p,g,f=core.match_method(name,gt,predictions,domain,iou)
            for split,tokens in splits.items():
                sf=f[f.frame_key.isin(tokens)]
                sp=p[p.frame_key.isin(tokens)]
                sg=g[g.frame_key.isin(tokens)]
                tp,fp,fn=[int(sf[k].sum()) for k in ["tp","fp","fn"]]
                m=core.metrics(tp,fp,fn,sp.loc[sp.status=="TP","matched_iou"].tolist(),len(tokens))
                all_metrics.append(dict(method=name,split=split,iou_threshold=iou,**m))
                if split=="holdout_test":
                    distance.extend(core.distance_summary(name,iou,sp,sg,len(tokens)))
            frame_details.append(f);pred_details.append(p);gt_details.append(g)
        print("Source canonical comparison",name,flush=True)
    df=pd.DataFrame(all_metrics)
    df.to_csv(out/"source_multi_iou_metrics.csv",index=False)
    df[df.split=="holdout_test"].to_csv(out/"holdout150_metrics.csv",index=False)
    pd.concat(frame_details,ignore_index=True).to_csv(out/"per_frame_metrics.csv",index=False)
    pd.concat(pred_details,ignore_index=True).to_csv(out/"per_prediction_details.csv",index=False)
    pd.concat(gt_details,ignore_index=True).to_csv(out/"per_gt_details.csv",index=False)
    pd.DataFrame(distance).to_csv(out/"holdout150_distance_metrics.csv",index=False)
    dump(out/"evaluation_audit.json",dict(status="pass",methods={k:str(p) for k,p in methods.items()},
        prediction_sha256={k:sha(p) for k,p in methods.items()},gt_sha256=sha(gt_path),
        all_frames=500,holdout_frames=150,summary_rows=len(df),
        all_gt_classes_preserved=True,matching="Frozen S1/S5 rotated BEV final metric core",
        core_sha256=sha(TASK/"C_target_recheck/kradar_sequence1/runtime_scripts/jei_s1_evaluator_core.py"),
        note="C+L original exported CSV identity and rounding retained; L and L+R S1/S3 stage28 public baseline output identity retained. No oracle box adjustment."))
    dump(out/"output_sha256.json",{str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.is_file()})
    print(df[df.split=="holdout_test"].to_string(index=False),flush=True)


if __name__=="__main__":
    main()
