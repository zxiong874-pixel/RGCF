# 中文投稿稿结果对应关系

本包方法名统一为 RGCF（可靠性引导的候选融合）。本次是更名与发布整理，不是重新训练或新增实验；所有预测、阈值和数值结果保持不变。

| 中文投稿稿位置 | 本包证据 |
| --- | --- |
| 表1 数据与用途 | data/source/split_manifest.csv；data/target/prepared/frame_manifest.csv；data/simulation/sensitivity/ |
| 表2 共同区域与全域F1 | data/target/results/aggregate_metrics.csv |
| 表3 检测计数 | data/target/results/aggregate_metrics.csv；per_frame_details.csv |
| 表4 四状态门控消融 | data/simulation/ablation/四状态指标.csv |
| 图3 距离段F1 | data/target/results/distance_metrics.csv |
| 图4 新增候选来源 | data/target/results/candidate_source_matches.csv；per_prediction_details.csv |
| 源域留出结果 | data/source/common_baseline_evaluation/holdout150_metrics.csv |
| 退化恢复与耗时 | data/simulation/summaries/ |
| 位姿一致性 | data/simulation/summaries/stage46_trajectory_consistency.csv |

中文稿图1、图2为结构示意图，不是数值实验图。本包 figures/ 目录保留既有结果图编号4—10，已重绘方法标签；不将它们冒充中文稿中同编号图片。CLAIM_TO_FILE_MAP.md 仍按既有实验稿位置定位，编号差异不代表结果差异。

原始数据及第三方权利边界见 DATA_SOURCES_AND_PERMISSIONS.md。本包不含未发表论文，不重新分发 K-Radar 原始传感数据及官方标注框。
