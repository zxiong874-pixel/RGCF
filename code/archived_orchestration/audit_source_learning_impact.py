#!/usr/bin/env python3
"""Separate geometry-only, feature-rebuild and refit effects without target tuning."""
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from rebuild_source_geometry_features import ROOT,TASK,S3,load_module
from native_geometry_audit import sha,dump,write_csv
from run_source_refit import enable_geometry_cache


def model(path):
    p=json.loads(path.read_text())
    m={k:p[k] for k in ["features","sources"]}
    m.update({k:np.asarray(p[k]) for k in ["mean","std","weights"]})
    return p,m


def geometry_only_prediction(rows):
    output=[]
    for r in rows:
        n=dict(r)
        # Every candidate source is linked to native objects in the preceding audit.
        # Geometry-only counterfactual preserves the legacy CSV precision and scores.
        if r.get("source","") != "radar_lirafusion":
            n["yaw"]=str(-float(r.get("yaw",0) or 0))
        output.append(n)
    return output


def main():
    refit=sorted((TASK / "B_source_rebuild").glob("source_refit_*/source_refit_audit.json"))[-1].parent
    report=json.loads((refit / "source_refit_audit.json").read_text())
    geometry=Path(report["feature_rebuild"])
    out=TASK / "B_source_rebuild/impact_audit"
    out.mkdir(exist_ok=False)
    train=load_module("impact_train",S3 / "scripts/train_causal_models.py")
    enable_geometry_cache()
    _,old_rows=train.read_rows(S3 / "features/candidates_causal_full_support.csv")
    _,new_rows=train.read_rows(geometry / "candidates_causal_full_support_canonical.csv")
    old_base_path=ROOT / "results/my_method_boreas_train500_iou0p10/class_adaptive_confidence_gate_v1/boreas_train500_class_adaptive_detections.csv"
    fields,old_base=train.read_rows(old_base_path)
    new_fields,new_base=train.read_rows(refit / "base_calibration/boreas_train500_class_adaptive_detections.csv")
    old_base_canonical=geometry_only_prediction(old_base)
    gt_old=train.evaluator.read_boxes(ROOT / "data/boreas_objects_obj_train_public_gt_500.csv",is_gt=True)
    gt_new=train.evaluator.read_boxes(TASK / "A_native_geometry/gt_pyboreas/boreas_train500_gt_canonical.csv",is_gt=True)
    manifest=pd.read_csv(S3 / "manifests/split_manifest.csv",dtype=str)
    splits={k:set(g.frame_key) for k,g in manifest.groupby("split")}
    splits["all500"]=set(manifest.frame_key)
    table=[]
    candidate_rows=[]
    original_validation=[]
    old_asset=S3 / "results/B2_past_only_retrained"
    asset_changes=[]
    for name in ["SimpleGeometryCausal","RGCF_B2_Causal"]:
        old_payload,old_model=model(old_asset / (name+"_model.json"))
        new_payload,new_model=model(refit / "frozen_source_assets" / (name+"_model.json"))
        old_pool=train.eligible(old_rows,.25,.30)
        new_pool=train.eligible(new_rows,.25,.30)
        assert [r["candidate_id"] for r in old_pool]==[r["candidate_id"] for r in new_pool]
        old_scored=train.score(old_pool,old_model)
        old_selected=train.select(old_scored,train.gate.rows_to_frames(old_base),old_payload["candidate_threshold"],1,4,.03,.1)
        _,archived=train.read_rows(old_asset / (name+"_selected_candidates.csv"))
        assert [r["candidate_id"] for r in old_selected]==[r["candidate_id"] for r in archived], "Original selection reproduction differs"
        original_validation.append(dict(method=name,selected_ids_identical=True,selected_count=len(old_selected)))
        _,old_prediction=train.read_rows(old_asset / (name+"_detections.csv"))
        geometry_prediction=geometry_only_prediction(old_prediction)
        feature_scores=train.score(new_pool,old_model)
        feature_selection=train.select(feature_scores,train.gate.rows_to_frames(old_base_canonical),old_payload["candidate_threshold"],1,4,.03,.1)
        scores=train.score(new_pool,new_model)
        selected=train.select(scores,train.gate.rows_to_frames(new_base),new_payload["candidate_threshold"],1,4,.03,.1)
        for stage,gt,frames in [
            ("historical",gt_old,train.gate.rows_to_frames(old_prediction)),
            ("geometry_only_frozen_predictions",gt_new,train.gate.rows_to_frames(geometry_prediction)),
            ("rebuilt_features_old_weights",gt_new,train.selected_frames(train.gate.rows_to_frames(old_base_canonical),feature_selection)),
            ("source_refit_and_calibration",gt_new,train.selected_frames(train.gate.rows_to_frames(new_base),selected))]:
            for split,tokens in splits.items():
                for iou in [.1,.25,.5]:
                    metrics=train.evaluator.evaluate(train.subset(gt,tokens),train.subset(frames,tokens),iou)
                    table.append(dict(method=name,stage=stage,split=split,iou_threshold=iou,**metrics))
        old_scores={r["candidate_id"]:r for r in train.score(old_rows,old_model)}
        new_scores={r["candidate_id"]:r for r in train.score(new_rows,new_model)}
        old_ids={r["candidate_id"] for r in old_selected}
        new_ids={r["candidate_id"] for r in selected}
        eligible_ids={r["candidate_id"] for r in old_pool}
        for old,new in zip(old_rows,new_rows):
            cid=old["candidate_id"]
            os=float(old_scores[cid]["learned_reliability_score"])
            ns=float(new_scores[cid]["learned_reliability_score"])
            def reason(score,threshold,chosen):
                if cid not in eligible_ids:return "source_floor_or_sync"
                if score<threshold:return "below_admission_threshold"
                return "accepted" if chosen else "base_overlap_or_greedy_budget_nms"
            candidate_rows.append(dict(method=name,frame_id=train.frame_token(new),candidate_id=cid,
                source=new["source"],split=new["split"],old_yaw=old["yaw"],new_yaw=new["yaw"],
                old_size_x=old["size_x"],new_size_x=new["size_x"],old_size_y=old["size_y"],new_size_y=new["size_y"],
                old_best_iou=old["best_gt_iou"],new_best_iou=new["best_gt_iou"],
                old_assigned_gt=old["best_gt_index"],new_assigned_gt=new["best_gt_index"],
                old_label=old["label_iou10"],new_label=new["label_iou10"],
                old_admission_score=os,new_admission_score=ns,eligible=cid in eligible_ids,
                old_decision=cid in old_ids,new_decision=cid in new_ids,
                old_reason=reason(os,old_payload["candidate_threshold"],cid in old_ids),
                new_reason=reason(ns,new_payload["candidate_threshold"],cid in new_ids)))
        asset_changes.append(dict(name=name,old_sha256=sha(old_asset/(name+"_model.json")),
            new_sha256=sha(refit/"frozen_source_assets"/(name+"_model.json")),
            max_weight_change=float(np.max(np.abs(new_model["weights"]-old_model["weights"]))),
            old_threshold=old_payload["candidate_threshold"],new_threshold=new_payload["candidate_threshold"],
            selected_before=len(old_selected),selected_after=len(selected),
            removed_ids=len(old_ids-new_ids),added_ids=len(new_ids-old_ids)))
    df=pd.DataFrame(table)
    df.to_csv(out / "source_effect_decomposition.csv",index=False)
    pd.DataFrame(candidate_rows).to_csv(out / "candidate_before_after_full_decisions.csv",index=False)
    differences=[]
    for (method,split,iou),g in df.groupby(["method","split","iou_threshold"]):
        g=g.set_index("stage")
        ordered=["historical","geometry_only_frozen_predictions","rebuilt_features_old_weights","source_refit_and_calibration"]
        for before,after in zip(ordered,ordered[1:]):
            differences.append(dict(method=method,split=split,iou_threshold=iou,before=before,after=after,
                **{"delta_"+k:g.loc[after,k]-g.loc[before,k] for k in ["tp","fp","fn","precision","recall","f1","mean_iou"]}))
    write_csv(out / "source_effect_incremental_deltas.csv",differences)
    write_csv(out / "model_asset_changes.csv",asset_changes)
    p0=pd.read_csv(ROOT / "results/my_method_boreas_train500_iou0p10/class_adaptive_confidence_gate_v1/selected_class_thresholds.csv").fillna("")
    p1=pd.read_csv(refit / "frozen_source_assets/selected_class_thresholds.csv").fillna("")
    report=dict(status="pass",original_selection_reproduced=original_validation,
        base_threshold_table_values_unchanged=p0.to_dict("records")==p1.to_dict("records"),
        candidate_ids_pool_order_budget_floor_unchanged=True,models=asset_changes,
        historical_outputs_untouched=True,non_algorithmic_change=True,
        effect_stage_definitions={"geometry_only_frozen_predictions":"Fixed boxes/scores/IDs; negate native-BEVFusion and GT CSV yaw according to full SDK audit; preserve LiRa yaw and source CSV rounding",
            "rebuilt_features_old_weights":"Rebuild yaw-dependent cross-source/GT geometry with native yaw precision; old trained weights and frozen admission threshold retained; reapply fixed postprocessing",
            "source_refit_and_calibration":"Original model family refit on 250, grid and selected threshold only100; original150 evaluated only after frozen assets exist"},
        source_refit=str(refit),source_feature_audit=str(geometry),
        limitations="These are evaluation/training-data integrity corrections, not an additional proposed algorithmic improvement")
    dump(out / "impact_audit.json",report)
    dump(out / "input_sha256.json",{str(p):sha(p) for p in [Path(__file__),S3/"features/candidates_causal_full_support.csv",geometry/"candidates_causal_full_support_canonical.csv",old_base_path]})
    dump(out / "output_sha256.json",{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
