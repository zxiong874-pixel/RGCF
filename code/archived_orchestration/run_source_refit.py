#!/usr/bin/env python3
"""Refit only on the frozen source train/calibration identities."""
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from rebuild_source_geometry_features import ROOT, TASK, S3, load_module, native_path
from native_geometry_audit import dump, sha, write_csv
sys.path.insert(0,str(ROOT / "scripts"))
import evaluate_boreas_detection_csv as evaluator
import calibrate_boreas_detection_thresholds as base_calibration
import run_supervisor_supplementary_ab as lira_calibration


def strict_calibration_threshold(original, train_module, scored, gt, base, tokens, args):
    cal_rows=[r for r in scored if train_module.frame_token(r) in tokens]
    cal_gt=train_module.subset(gt,tokens)
    cal_base=train_module.subset(base,tokens)
    assert cal_rows and len(cal_gt)==100
    return original(cal_rows,cal_gt,cal_base,tokens,args)


def enable_geometry_cache():
    original=evaluator.bev_iou
    keys=("center_x","center_y","size_x","size_y","yaw")
    @lru_cache(maxsize=500000)
    def cached(a,b):
        return original(dict(zip(keys,a)),dict(zip(keys,b)))
    def iou(a,b):
        return cached(tuple(float(a[k]) for k in keys),tuple(float(b[k]) for k in keys))
    evaluator.bev_iou=iou
    return cached


def main():
    evidence=sorted((TASK / "B_source_rebuild").glob("geometry_and_features_*/rebuild_audit.json"))
    geometry=evidence[-1].parent
    assert json.loads(evidence[-1].read_text())["status"]=="pass"
    out=TASK / ("B_source_rebuild/source_refit_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True,exist_ok=False)
    gt_path=TASK / "A_native_geometry/gt_pyboreas/boreas_train500_gt_canonical.csv"
    feature_path=geometry / "candidates_causal_full_support_canonical.csv"
    split_path=S3 / "manifests/split_manifest.csv"
    split=pd.read_csv(split_path,dtype=str)
    tokens={name:set(g.frame_key) for name,g in split.groupby("split")}
    assert {k:len(v) for k,v in tokens.items()}==dict(model_development=250,calibration=100,holdout_test=150)
    assert len(set.union(*tokens.values()))==500
    cache=enable_geometry_cache()
    base_dir=out / "base_calibration"
    base_input=geometry / "bevfusion_lidar_canonical_historical_precision.csv"
    sys.argv=["calibrate_boreas_detection_thresholds.py","--gt-csv",str(gt_path),
        "--det-csv",str(base_input),"--out-dir",str(base_dir),"--method-name","BaseLiDAR_Canonical_SourceCal"]
    print("Calibrating source LiDAR classes on original 100 frames",flush=True)
    base_calibration.main()
    base_path=base_dir / "boreas_train500_class_adaptive_detections.csv"
    train=load_module("s3_train_causal_frozen",S3 / "scripts/train_causal_models.py")
    original_choose=train.choose_threshold
    train.choose_threshold=lambda scored,gt,base,cal_tokens,args: strict_calibration_threshold(
        original_choose,train,scored,gt,base,cal_tokens,args)
    args=SimpleNamespace(camera_min_score=.25,radar_min_score=.30,max_camera_per_frame=1,
        max_radar_per_frame=4,max_base_iou=.03,nms_iou=.10,precision_floor=.86,
        f1_tolerance=.00025,epochs=800,learning_rate=.08,l2=.01,label_key="label_iou10")
    fields,base_rows=train.read_rows(base_path)
    _,features=train.read_rows(feature_path)
    gt=train.evaluator.read_boxes(gt_path,is_gt=True)
    # Numeric order must exactly match frozen split identities, not a new split.
    expected=base_calibration.split_timestamps(gt,250,350)
    assert {train.frame_token({"timestamp":t}) for t in expected["calibration"]}==tokens["calibration"]
    models=out / "frozen_source_assets"
    models.mkdir()
    runs=[]
    for name,features_list in [("SimpleGeometryCausal",train.SIMPLE_FEATURES),("RGCF_B2_Causal",train.FULL_FEATURES)]:
        print("Fitting train250 and calibrating cal100:",name,flush=True)
        runs.append(train.run_model(name,features_list,features,fields,base_rows,gt,tokens,args,models))
        payload=json.loads((models/(name+"_model.json")).read_text())
        eligible=train.eligible(features,.25,.30)
        train_rows=[r for r in eligible if train.frame_token(r) in tokens["model_development"]]
        _,_,mean,std=train.matrix(train_rows,features_list)
        assert np.allclose(mean,payload["mean"],rtol=0,atol=1e-12)
        assert np.allclose(std,payload["std"],rtol=0,atol=1e-12)
    print("Calibrating LiRaFusion on original cal100, frozen score-order rule",flush=True)
    lira_path=native_path("stage28_clean_lirafusion_lidar_radar")
    lg=pd.read_csv(gt_path)
    ld=pd.read_csv(lira_path)
    for data in [lg,ld]:
        data["frame_key"]=data.timestamp.map(lira_calibration.frame_key)
        data["row_id"]=np.arange(len(data))
    cal_times={"%.6f"%(int(t)/1e6) for t in tokens["calibration"]}
    details,gdetail=lira_calibration.match_details(lg[lg.frame_key.isin(cal_times)],ld[ld.frame_key.isin(cal_times)],cal_times,.1)
    scores=ld.set_index("row_id").loc[details.row_id,"score"].to_numpy()
    label=(details.status=="TP").astype(int).to_numpy()
    threshold,f1=lira_calibration.best_threshold(label,scores,len(gdetail))
    write_csv(models / "lirafusion_calibrated_threshold.csv",[dict(class_name="DEFAULT",threshold=threshold)])
    details.to_csv(out / "lirafusion_cal100_match_details.csv",index=False)
    gdetail.to_csv(out / "lirafusion_cal100_gt_details.csv",index=False)
    # Copy table contents as a distinct frozen asset; original table is untouched.
    (models / "selected_class_thresholds.csv").write_bytes((base_dir / "selected_class_thresholds.csv").read_bytes())
    report=dict(status="pass",feature_rebuild=str(geometry),split_manifest=str(split_path),
        split_sha256=sha(split_path),train_frames=250,calibration_frames=100,holdout_frames=150,
        normalization_and_fit_only_train=True,threshold_grid_and_selection_only_calibration=True,
        historical_grid_issue="Original threshold_grid(scored) included unlabeled holdout/development score quantiles; corrected wrapper restricts candidates and GT/base maps to calibration100",
        no_target_data_access=True,model_family_unchanged=True,source_floor_budget_nms_unchanged=True,
        fit_hyperparameters=vars(args),source_inputs={str(p):sha(p) for p in [gt_path,feature_path,base_input,base_path,split_path,lira_path]},
        source_implementation={str(p):sha(p) for p in [Path(train.__file__),Path(base_calibration.__file__),Path(lira_calibration.__file__),Path(__file__)]},
        runs=[{k:r[k] for k in ["name","threshold","selected"]} for r in runs],
        lirafusion_threshold=threshold,lirafusion_calibration_f1=f1,
        geometry_cache="Memoizes unchanged evaluator for identical x,y,dx,dy,yaw tuples; score and identities do not enter IoU",
        cache_info=str(cache.cache_info()))
    dump(out / "source_refit_audit.json",report)
    dump(out / "output_sha256.json",{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob("*")) if p.is_file()})
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
