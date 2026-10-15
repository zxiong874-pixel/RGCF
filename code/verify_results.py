"""Recompute frozen results; no training, sensor downloads or GPU invocation."""
from pathlib import Path
import argparse,json,sys,importlib.util
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'code'))
import jei_s1_evaluator_core as core

def compare(actual,expected,keys):
    a=actual.sort_values(keys).reset_index(drop=True)
    b=expected.sort_values(keys).reset_index(drop=True)
    assert len(a)==len(b),(len(a),len(b))
    for k in keys:assert a[k].astype(str).tolist()==b[k].astype(str).tolist(),k
    for k in ['tp','fp','fn','precision','recall','f1','matched_mean_iou','mean_iou']:
        if k in a and k in b:
            np.testing.assert_allclose(a[k],b[k],rtol=0,atol=1e-9,err_msg=k)

def quick(out):
    t=R/'data/target/results'
    ag=pd.read_csv(t/'aggregate_metrics.csv')
    pf=pd.read_csv(t/'per_frame_details.csv')
    keys=['scope','method','iou_threshold']
    counts=pf.groupby(keys,as_index=False)[['tp','fp','fn']].sum()
    compare(counts,ag,keys)
    assert len(ag)==36 and len(pf)==21492
    bg=pd.read_csv(R/'data/source/common_baseline_evaluation/holdout150_metrics.csv')
    for df in [ag,bg,pd.read_csv(R/'data/simulation/sensitivity/four_state_metrics.csv')]:
        np.testing.assert_allclose(df.f1,2*df.tp/(2*df.tp+df.fp+df.fn),rtol=0,atol=1e-12)
    truth=pd.read_csv(R/'data/gating_logic/gating_truth_table.csv')
    rad=(truth.qL<=truth.qLmax)&(truth.radar_support_points>0)
    cam=(truth.qL<=truth.qLmax)&(truth.qR<=.12)&(truth.qC_image>=.1)&(truth.qC_semantic>=.1)&(truth.mask_state!='absent')
    assert (rad==truth.radar_gate).all() and (cam==truth.camera_gate).all()
    proposed=ag[(ag.scope=='common_xy_roi')&(ag.method=='RGCF transfer*')].sort_values('iou_threshold')
    assert [round(x*100,2) for x in proposed.f1]==[54.39,52.23,47.80]
    proposed=bg[bg.method=='RGCF'].sort_values('iou_threshold')
    assert [round(x*100,2) for x in proposed.f1]==[79.60,79.27,76.63]
    return {'target_summary_rows':36,'target_frame_rows':21492,'truth_table_rows':len(truth),'abstract_F1_values_match':True,'scope':'Re-aggregation and predicate checks, not independent geometry rematching'}

def source(out):
    s=R/'data/source';b=s/'common_baseline_evaluation';f=s/'source_refit_20260909_173816'
    paths={'BEVFusion L':s/'bevfusion_lidar_canonical.csv','BEVFusion C+L':b/'predictions/bevfusion_camera_lidar_canonical.csv','LiRaFusion L+R':b/'predictions/lirafusion_calibrated_canonical.csv','Base LiDAR':f/'base_calibration/boreas_train500_class_adaptive_detections.csv','Simple fusion':f/'frozen_source_assets/SimpleGeometryCausal_detections.csv','RGCF':f/'frozen_source_assets/RGCF_B2_Causal_detections.csv'}
    gt,domain=core.load_gt(s/'boreas_train500_gt_canonical.csv')
    split=pd.read_csv(s/'split_manifest.csv',dtype=str)
    holdout=split.loc[split.split=='holdout_test','frame_key'].tolist()
    assert len(domain)==500 and len(holdout)==150
    rows=[]
    for method,path in paths.items():
        pred,unmapped,_=core.load_detections(path,domain);assert not unmapped
        for th in [.1,.25,.5]:
            p,g,frames=core.match_method(method,gt,pred,domain,th)
            frames=frames[frames.frame_key.isin(holdout)];p=p[p.frame_key.isin(holdout)]
            tp,fp,fn=[int(frames[k].sum()) for k in ['tp','fp','fn']]
            rows.append(dict(method=method,iou_threshold=th,**core.metrics(tp,fp,fn,p.loc[p.status=='TP','matched_iou'].tolist(),150)))
        print('Source rematched:',method,flush=True)
    data=pd.DataFrame(rows);compare(data,pd.read_csv(b/'holdout150_metrics.csv'),['method','iou_threshold'])
    data.to_csv(out/'source_recomputed.csv',index=False)
    return {'holdout_rows_rematched':len(rows),'frames':150,'methods':6,'match':True}

def target(out,gt_path):
    if not gt_path:raise ValueError('Target geometry rematching requires --gt-csv obtained locally with prepare_kradar_gt.py. No official K-Radar annotation is redistributed.')
    task=R/'data/target'
    # Reuse the recorded evaluator without modifying the frozen package.
    spec=importlib.util.spec_from_file_location('original',R/'code/original/recompute_target_metrics.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    gt,domain_gt=core.load_gt(gt_path)
    from decimal import Decimal,ROUND_HALF_UP
    manifest=pd.read_csv(task/'prepared/frame_manifest.csv',dtype=str)
    domain=[str(int((Decimal(x)/1000).to_integral_value(rounding=ROUND_HALF_UP))) for x in manifest.timestamp_ns]
    assert len(domain)==597 and sum(map(len,gt.values()))==1890
    rows=[];frames=[]
    for scope in ['full_annotated_domain','common_xy_roi']:
        inside=lambda box:abs(box.center_x)<=51.2 and abs(box.center_y)<=51.2
        sg={t:[x for x in boxes if scope=='full_annotated_domain' or inside(x)] for t,boxes in gt.items()}
        for method,stem in mod.METHODS.items():
            pred,unmapped,_=core.load_detections(task/'predictions'/(stem+'.csv'),domain);assert not unmapped
            pred={t:[x for x in boxes if scope=='full_annotated_domain' or inside(x)] for t,boxes in pred.items()}
            for th in [.1,.25,.5]:
                p,g,f=core.match_method(method,sg,pred,domain,th)
                tp,fp,fn=[int(f[k].sum()) for k in ['tp','fp','fn']]
                rows.append(dict(scope=scope,method=method,iou_threshold=th,**core.metrics(tp,fp,fn,p.loc[p.status=='TP','matched_iou'].tolist(),597)))
                f['scope']=scope;frames.append(f)
            print('Target rematched:',scope,method,flush=True)
    data=pd.DataFrame(rows);pf=pd.concat(frames,ignore_index=True)
    compare(data,pd.read_csv(task/'results/aggregate_metrics.csv'),['scope','method','iou_threshold'])
    compare(pf,pd.read_csv(task/'results/per_frame_details.csv',dtype={'frame_key':str}),['scope','method','iou_threshold','frame_key'])
    data.to_csv(out/'target_recomputed.csv',index=False)
    return {'summary_rows_rematched':len(data),'frame_rows_rematched':len(pf),'match':True,'gt_supplied_externally':True}

def simulation(out):
    sys.path.insert(0,str(R/'code/research/ros_src'))
    from bev_obstacle_detection.detection_metrics import Box2D,evaluate_detection_frames
    def load(path):
        df=pd.read_csv(path);frames={}
        for row in df.to_dict('records'):
            t=float(row['timestamp'])
            frames.setdefault(t,[]).append(Box2D(timestamp=t,center_x=float(row['center_x']),center_y=float(row['center_y']),size_x=float(row['size_x']),size_y=float(row['size_y'])))
        return frames
    root=R/'data/simulation/sensitivity';rows=[]
    expected=pd.read_csv(root/'four_state_metrics.csv')
    for row in expected.to_dict('records'):
        folder=root/row['scenario'];gt=load(folder/'gt.csv');pred=load(folder/f"predictions_qL_{row['qLmax']}.csv")
        # Preserve all frame timestamps, including empty prediction frames.
        times=pd.read_csv(folder/'frame_quality.csv').timestamp
        for t in times:gt.setdefault(float(t),[])
        m=evaluate_detection_frames(pred,gt,.1)
        rows.append(dict(scenario=row['scenario'],qLmax=row['qLmax'],**m))
    actual=pd.DataFrame(rows);compare(actual,expected,['scenario','qLmax'])
    actual.to_csv(out/'simulation_recomputed.csv',index=False)
    return {'settings_rematched':8,'frames_per_state':274,'match':True,'protocol':'axis-aligned Box2D, IoU-descending greedy matching'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['quick','source','target','simulation'],default='quick');p.add_argument('--gt-csv',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    result=target(a.out,a.gt_csv) if a.mode=='target' else globals()[a.mode](a.out)
    result.update(status='PASS',mode=a.mode,python=sys.version.split()[0])
    (a.out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
