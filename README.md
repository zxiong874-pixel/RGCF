# RGCF

**Reliability-Guided Candidate Fusion（可靠性引导的候选融合）** for multisensor obstacle perception.

Authors: Qixue Guan and Zhuang Xiong, School of Information Science and Engineering, Shenyang Ligong University, China. Corresponding author: Zhuang Xiong, xiong239510@foxmail.com.

RGCF combines reliability-guided admission of auxiliary detections with conditional candidate recovery under sensor degradation. This repository provides source snapshots, frozen evaluation records and CPU verification tools supporting the work. Public-data rotated-BEV evaluation and controlled UAV simulation use their separately documented protocols.

## Download and reproduce

Download **RGCF_Public_Package_20261015.zip** and its matching **.zip.sha256** file from [Releases](https://github.com/zxiong874-pixel/RGCF/releases/tag/v1.0.0). If the release is not visible, publication of the data package is still pending.

Extract the release archive and run commands **inside the extracted RGCF_Public_Package_20261015 folder**. The browsable repository and GitHub's automatic Source code ZIP do not contain the experimental data. Do not run data-dependent checks from this lightweight repository tree.

```text
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe code/verify_integrity.py
.venv\Scripts\python.exe code/verify_results.py --mode quick --out checks_quick
.venv\Scripts\python.exe code/verify_results.py --mode source --out checks_source
.venv\Scripts\python.exe code/verify_results.py --mode simulation --out checks_simulation
.venv\Scripts\python.exe code/plot_results.py --out review_figures
```

Use Python 3.12. On Linux/macOS use `.venv/bin/python`. Each output directory must be new. Target rematching additionally requires official K-Radar labels obtained locally; see [REPRODUCTION.md](REPRODUCTION.md). They are not redistributed in this release.

## Verification of the renamed release

The October 15, 2026 verification reproduced 18 source-holdout metric rows, 36 target aggregate rows and 21,492 target per-frame rows, and eight simulation settings. The quick check also verified 7,500 quality-gate cases. All 154 archived CSV files retain their original values and row ordering, with only textual method identifiers and associated file references changed. Seven numerical result plots were regenerated with the RGCF labels.

| Evaluation | RGCF F1 at IoU 0.10 | IoU 0.25 | IoU 0.50 |
| --- | ---: | ---: | ---: |
| Boreas held-out source | 79.60% | 79.27% | 76.63% |
| K-Radar common region | 54.39% | 52.23% | 47.80% |

These are rotated-BEV detection F1 values, not official 3D AP or mAP. K-Radar uses 597 frozen-transfer anchors without pose input. Detailed verification reports are included in the release.

## Files and naming

- [REPRODUCTION.md](REPRODUCTION.md): setup and reproduction boundaries.
- [NAMING.md](NAMING.md): current method names and identifier conventions.
- [CHINESE_MANUSCRIPT_MAP.md](CHINESE_MANUSCRIPT_MAP.md): current Chinese manuscript results mapped to evidence.
- [CLAIM_TO_FILE_MAP.md](CLAIM_TO_FILE_MAP.md): archived experimental manuscript numbering.
- `code/`: verification tools and research snapshots. Archived orchestration requires its original external environments and inputs.
- `figures/`: seven numerical review plots; figure numbering follows the archived experimental manuscript, not the condensed Chinese manuscript.

This is a naming migration of the existing experimental snapshot, not a new experiment or a claim of journal acceptance. Current labels and filenames use RGCF. Historical source names and hashes remain explicitly identified in the release provenance to preserve traceability.

## Data and permissions

See [DATA_SOURCES_AND_PERMISSIONS.md](DATA_SOURCES_AND_PERMISSIONS.md). Original sensor streams, official K-Radar annotation boxes, detector weights, ROS bags and unpublished manuscripts are not included. This package does not reproduce detector training or the complete physical UAV pipeline. Third-party rights remain unchanged; no blanket software/data license is assigned to this mixed-material release.
