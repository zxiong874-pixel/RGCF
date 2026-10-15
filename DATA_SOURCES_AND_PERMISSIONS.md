# Data sources and sharing boundaries

## Author generated materials

Frozen model predictions, fitted admission parameters, experimental logs, evaluation summaries and the authors' scripts are provided for editorial and peer-review examination. The corresponding author is Zhuang Xiong, xiong239510@foxmail.com. This package does not grant a new blanket license over the authors' work or third-party materials; any wider reuse must respect the applicable rights and permissions. Nothing here restricts rights already granted by a third-party license.

## Boreas

Source: Burnett et al., *Boreas: A Multi-Season Autonomous Driving Dataset*, The International Journal of Robotics Research, 42(1–2), 33–42 (2023), DOI 10.1177/02783649231160195.

- Dataset and access: https://www.boreas.utias.utoronto.ca/
- Official data license: https://github.com/utiasASRL/pyboreas/blob/master/DATA_LICENSE.md
- License identified on the official page: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/

The included `data/source/boreas_train500_gt_canonical.csv` is a converted 500-anchor annotation subset. Its source is the official Boreas annotations, checked in the S6 geometry audit; the representation was converted to the evaluation's canonical box schema. Personal source-path strings were replaced where present. The dataset authors do not endorse this manuscript. Complete sensor data and the full original dataset are not redistributed.

## K Radar

Source: Paek, Kong and Wijaya, *K-Radar: 4D Radar Object Detection for Autonomous Driving in Various Weather Conditions*, NeurIPS Datasets and Benchmarks (2022), DOI 10.52202/068431-0276.

- Official repository, dataset access and license statement: https://github.com/kaist-avelab/K-Radar
- Frozen annotation version: commit f90c28993d1757ce0236632b4d912c92a14976b2, tools/revise_label/kradar_revised_label_v2_1.zip.
- The official repository identifies dataset licensing as CC BY-NC-ND and code licensing as Apache 2.0.

For conservative distribution, official sensor files and converted annotation boxes are excluded. Frame identifiers, the authors' predictions, measured matching outcomes and aggregate metrics are retained. Reviewers can obtain official annotations from their original channel and run the supplied converter locally. The archive should not be described as a newly licensed version of K-Radar. Confirm the source terms before any further distribution.

## Simulation and third party software

Gazebo-derived GT exports, candidate outputs and quality records are experimental results from the authors' XTDrone/PX4 workflow. The package does not distribute the full XTDrone, PX4, Gazebo, ROS, BEVFusion, LiRaFusion or VINS software distributions or their network weights. Obtain external dependencies from their original projects under their own terms. Research-code snapshots do not certify an independently installable ROS package; use the portable verification entry points for the tested CPU workflow.

Official license/source pages were checked on September 11, 2026. This source record documents packaging choices; it is not a legal determination of every downstream use.
