"""Generate review charts for manuscript Figures 4–10 from packaged CSVs.
These are numerical reproductions, not pixel-identical publication figures.
"""
from pathlib import Path
import argparse
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):fig.savefig(a.out/(name+'.png'),dpi=160,bbox_inches='tight');plt.close(fig)
    def bars(ax,x,vals,label,offset=0,color=None):
        b=ax.bar(np.asarray(x)+offset,vals,width=.34,label=label,color=color);ax.bar_label(b,fmt='%.2f',padding=2,fontsize=8,rotation=0)
    ag=pd.read_csv(R/'data/target/results/aggregate_metrics.csv');dist=pd.read_csv(R/'data/target/results/distance_metrics.csv')
    methods=ag.method.drop_duplicates().tolist()
    fig,axs=plt.subplots(3,1,figsize=(8,9),layout='constrained')
    for ax,t in zip(axs,[.1,.25,.5]):
        v=ag[(ag.scope=='common_xy_roi')&np.isclose(ag.iou_threshold,t)].set_index('method').loc[methods]
        b=ax.bar(range(6),v.f1*100);ax.bar_label(b,fmt='%.2f',padding=2);ax.set_xticks(range(6),['BEV L','BEV C+L','LiRa L+R','Base','Simple','RGCF*']);ax.set_ylim(0,65);ax.set_ylabel('F1 (%)');ax.set_title(f'Rotated-BEV IoU {t:.2f}')
    save(fig,'figure04_target_F1')
    fig,axs=plt.subplots(2,1,figsize=(8,7),layout='constrained');x=np.arange(3)
    for off,scope,lab in [(-.17,'common_xy_roi','Common square'),(.17,'full_annotated_domain','Full annotations')]:
        v=ag[(ag.scope==scope)&(ag.method=='RGCF transfer*')].sort_values('iou_threshold');bars(axs[0],x,v.f1*100,lab,off)
    axs[0].set_xticks(x,['0.10','0.25','0.50']);axs[0].legend();axs[0].set_ylabel('F1 (%)');axs[0].set_ylim(0,70)
    for method in ['Base LiDAR','Simple fusion','RGCF transfer*']:
        v=dist[(dist.scope=='common_xy_roi')&(dist.method==method)&np.isclose(dist.iou_threshold,.1)]
        axs[1].plot(v.distance_bin,np.where(v.gt_count>0,v.f1*100,np.nan),'o-',label=method)
    axs[1].legend();axs[1].set_ylabel('F1 (%)');axs[1].set_xlabel('Distance bin (m); 0–3 m has no GT');save(fig,'figure05_scope_distance')
    pp=pd.read_csv(R/'data/target/results/per_prediction_details.csv');pp=pp[(pp.scope=='common_xy_roi')&(pp.method=='RGCF transfer*')]
    fig,axs=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    for ax,source in zip(axs,['radar_lirafusion','camera_lidar']):
        v=pp[pp.source==source].groupby(['iou_threshold','status']).size().unstack(fill_value=0).sort_index()
        b=ax.bar(x,v.TP,label='TP');c=ax.bar(x,v.FP,bottom=v.TP,label='FP');ax.bar_label(b,label_type='center');ax.bar_label(c,label_type='center');ax.set_xticks(x,['0.10','0.25','0.50']);ax.set_title(source);ax.set_ylim(0,float((v.TP+v.FP).max())*1.25);ax.legend()
    save(fig,'figure06_candidate_errors')
    summaries=R/'data/simulation/summaries';traj=pd.read_csv(summaries/'stage46_trajectory_consistency.csv');v=traj[traj.interval=='stable'].set_index('method').loc[['PX4','VINS']]
    fig,axs=plt.subplots(1,2,figsize=(9,4),layout='constrained');x=np.arange(2)
    for off,col,lab in [(-.17,'ate_rmse_m','ATE RMSE'),(.17,'ate_p95_m','ATE P95')]:
        b=axs[0].bar(x+off,v[col],.34,label=lab);axs[0].bar_label(b,fmt='%.4f',padding=3)
    for off,col,lab in [(-.17,'gt_path_length_m','GT path'),(.17,'estimated_path_length_m','Estimated path')]:bars(axs[1],x,v[col],lab,off)
    for ax in axs:ax.set_xticks(x,['PX4','VINS']);ax.legend(loc='upper left',bbox_to_anchor=(0,1.20));ax.set_ylabel('m')
    save(fig,'figure07_trajectory')
    radar=pd.read_csv(summaries/'stage47_radar_surface_overall.csv');camera=pd.read_csv(summaries/'stage49_camera_semantic_overall.csv')
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained');x=np.arange(3)
    for ax,df,conditions,ms in [(axs[0],radar,['normal','stage46_severe','stage45_holdout'],['baseline','radar_surface']),(axs[1],camera,['development_dual_failure','holdout_dual_failure','camera_failure'],['reference','full_method'])]:
        for off,m in zip([-.17,.17],ms):bars(ax,x,df[df.method==m].set_index('condition').loc[conditions].f1,m,off)
        ax.set_xticks(x,['Normal','Severe','Holdout'] if ax is axs[0] else ['Joint dev.','Joint holdout','Camera dropout']);ax.set_ylabel('F1');ax.set_ylim(0,.9);ax.legend()
    save(fig,'figure08_conditional_recovery')
    g=pd.read_csv(R/'data/simulation/sensitivity/four_state_metrics.csv');states=g.scenario.drop_duplicates().tolist();fig,ax=plt.subplots(figsize=(8,4),layout='constrained');x=np.arange(4)
    for off,t in [(-.17,.12),(.17,.20)]:bars(ax,x,g[np.isclose(g.qLmax,t)].set_index('scenario').loc[states].f1*100,f'qL threshold {t}',off)
    ax.set_xticks(x,states);ax.set_ylim(0,90);ax.set_ylabel('F1 (%)');ax.legend();save(fig,'figure09_thresholds')
    conditions=['normal','lidar_severe','development_dual_failure','holdout_dual_failure','camera_failure'];v=camera[camera.method=='full_method'].set_index('condition').loc[conditions]
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained');x=np.arange(5)
    for off,col,lab in [(-.17,'compute_mean_ms','Mean'),(.17,'compute_p95_ms','P95')]:bars(ax,x,v[col],lab,off)
    ax.set_xticks(x,['Normal','Severe LiDAR','Joint dev.','Joint holdout','Camera dropout']);ax.set_ylabel('Recorded computation (ms)');ax.set_title('XTDrone/PX4 component timing; not complete end-to-end latency');ax.legend();save(fig,'figure10_timing')
    print('Generated seven CSV-derived review charts')
if __name__=='__main__':main()
