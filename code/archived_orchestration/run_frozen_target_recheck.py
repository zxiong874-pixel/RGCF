#!/usr/bin/env python3
"""Apply source-corrected assets to unchanged K-Radar features, then evaluate."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from rebuild_source_geometry_features import ROOT,TASK,S3,load_module
from native_geometry_audit import sha,dump,write_csv


def main():
    for p in [TASK / "A_native_geometry/summary/native_geometry_audit.json",TASK / "B_source_rebuild/impact_audit/impact_audit.json"]:
        assert json.loads(p.read_text())["status"]=="pass"
    refit=Path(json.loads((TASK / "B_source_rebuild/impact_audit/impact_audit.json").read_text())["source_refit"])
    s5=ROOT / "07_JEI实验补强/S5_跨采集验证/kradar_sequence1"
    target=TASK / "C_target_recheck/kradar_sequence1"
    target.mkdir(parents=True,exist_ok=False)
    for name in ["prepared","predictions","features","frozen_assets","runtime_scripts","scripts","logs"]:
        (target/name).mkdir()
    copies=[]
    def copy(src,dst):
        shutil.copy2(src,dst)
        copies.append(dict(source=str(src),target=str(dst),sha256=sha(dst)))
    for name in ["frame_manifest.csv","gt_v2_1_native_lidar.csv","network_sync_manifest.json","preparation_audit.json"]:
        copy(s5/"prepared"/name,target/"prepared"/name)
    for name in ["semantic_masks_manifest.json","semantic_masks_summary.csv"]:
        copy(s5/name,target/name)
    for path in (s5/"predictions").iterdir():
        if path.name.startswith(("SimpleGeometry_","RGCF_")):continue
        copy(path,target/"predictions"/path.name)
    for name in ["candidates_frozen_transfer.csv","feature_audit.json"]:
        copy(s5/"features"/name,target/"features"/name)
    for name in ["SimpleGeometryCausal_model.json","RGCF_B2_Causal_model.json","selected_class_thresholds.csv","lirafusion_calibrated_threshold.csv"]:
        copy(refit/"frozen_source_assets"/name,target/"frozen_assets"/name)
    # Tables are semantically unchanged; verify this before reusing filtered base outputs.
    for name in ["selected_class_thresholds.csv","lirafusion_calibrated_threshold.csv"]:
        old=pd.read_csv(s5/"frozen_boreas_assets"/name).fillna("")
        new=pd.read_csv(target/"frozen_assets"/name).fillna("")
        assert old.to_dict("records")==new.to_dict("records"), "Need regenerate base/class filters if source calibration changed"
    core=ROOT / "07_JEI实验补强/S4_独立场景验证_20260907_113617/runtime_scripts/jei_s1_evaluator_core.py"
    copy(core,target/"runtime_scripts/jei_s1_evaluator_core.py")
    copy(TASK/"scripts/recompute_target_metrics.py",target/"scripts/recompute_target_metrics.py")
    train=load_module("target_frozen_train_functions",S3 / "scripts/train_causal_models.py")
    fields,base=train.read_rows(target/"predictions/rgcf_base_lidar_roadusers.csv")
    _,features=train.read_rows(target/"features/candidates_frozen_transfer.csv")
    audits=[]
    for name,asset in [("SimpleGeometry_FrozenTransfer","SimpleGeometryCausal_model.json"),("RGCF_FrozenTransfer_NoPose","RGCF_B2_Causal_model.json")]:
        p=json.loads((target/"frozen_assets"/asset).read_text())
        model={k:p[k] for k in ["features","sources"]}
        model.update({k:np.asarray(p[k]) for k in ["mean","std","weights"]})
        policy=p["common_candidate_policy"]
        pool=train.eligible(features,policy["camera_min_score"],policy["radar_min_score"])
        scored=train.score(pool,model)
        selected=train.select(scored,train.gate.rows_to_frames(base),p["candidate_threshold"],policy["max_camera_per_frame"],policy["max_radar_per_frame"],policy["max_base_iou"],policy["nms_iou"])
        output=[]
        all_fields=fields+[k for k in ["learned_reliability_score","candidate_id"] if k not in fields]
        for index,r in enumerate(base+selected):
            row={k:r.get(k,"") for k in all_fields}
            row["method"]=name;row["obstacle_id"]=index
            if row["learned_reliability_score"]:row["score"]=row["learned_reliability_score"]
            assert row["class_name"] and row["sample_token"]
            output.append(row)
        write_csv(target/"predictions"/(name+".csv"),output,all_fields)
        write_csv(target/"features"/(name+"_scores.csv"),scored)
        audits.append(dict(method=name,model_sha256=sha(target/"frozen_assets"/asset),
            candidate_threshold=p["candidate_threshold"],base=len(base),eligible=len(pool),selected=len(selected),
            selected_by_source={s:sum(r["source"]==s for r in selected) for s in p["sources"]},
            target_gt_access_during_prediction=False,fit_performed=False))
    dump(target/"frozen_inference_audit.json",dict(status="pass",methods=audits,
        source_refit=str(refit),all_target_features_unchanged=True,all_base_thresholds_unchanged=True,
        no_target_tuning=True,no_gpu_inference_repeated=True,
        limitation="Frozen offline transfer, no actual VINS, no Navtech raw-quality evidence; one future camera match and unverified Radar timing unchanged"))
    dump(target/"copied_input_sha256.json",copies)
    command=[sys.executable,str(target/"scripts/recompute_target_metrics.py"),"--task",str(target),"--out-dir",str(target/"results")]
    with (target/"logs/evaluation.log").open("w",encoding="utf-8") as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0,"Inspect target evaluation log"
    old=pd.read_csv(s5/"results/aggregate_metrics.csv")
    new=pd.read_csv(target/"results/aggregate_metrics.csv")
    keys=["scope","method","iou_threshold"]
    changes=old.merge(new,on=keys,suffixes=("_old","_new"),validate="one_to_one")
    for key in ["tp","fp","fn","precision","recall","f1","matched_mean_iou","fp_per_frame"]:
        changes["delta_"+key]=changes[key+"_new"]-changes[key+"_old"]
    changes.to_csv(target/"results/before_after_metrics.csv",index=False)
    dump(target/"output_sha256.json",{str(p.relative_to(target)):sha(p) for p in sorted(target.rglob("*")) if p.is_file()})
    print(new[new.scope=="common_xy_roi"].to_string(index=False),flush=True)
    print(target,flush=True)


if __name__=="__main__":
    main()
