# RGCF public data and code package

**RGCF — Reliability-Guided Candidate Fusion（可靠性引导的候选融合）** is the current method name. This release renames the September 11, 2026 experimental evidence snapshot for the Chinese submission; it does not change any experimental values or claim a new experiment. Manuscript files are not distributed here. Authors: Qixue Guan and Zhuang Xiong, School of Information Science and Engineering, Shenyang Ligong University. Corresponding author: Zhuang Xiong, xiong239510@foxmail.com.

The package contains frozen predictions, evaluation records, source-fitted parameters, simulation records, research code, and portable CPU verification tools. It allows reviewers to trace reported F1 values to prediction and frame records, repeat matching, and regenerate numerical versions of the archived result figures. It does not retrain the public detectors or reproduce a physical UAV experiment.

Project: https://github.com/zxiong874-pixel/RGCF

Current method labels, filenames, code references and regenerated plots use RGCF. Authentic historical source identifiers remain only in provenance records so that the migration can be traced. `provenance/MANIFEST.json` validates current bytes; archived experiment hashes are historical, not current-file checksums.

## Start here

Use Python 3.12. The preparation run used Windows, Python 3.12.14, and the exact direct dependency versions recorded in requirements.txt. A short extraction path is recommended. Install dependencies into a separate environment; do not execute archived orchestration scripts as the first step.

```text
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe code/verify_integrity.py
.venv\Scripts\python.exe code/verify_results.py --mode quick --out checks_quick
.venv\Scripts\python.exe code/verify_results.py --mode source --out checks_source
.venv\Scripts\python.exe code/verify_results.py --mode simulation --out checks_simulation
.venv\Scripts\python.exe code/plot_results.py --out review_figures
```

On Linux/macOS, replace `.venv\Scripts\python.exe` with `.venv/bin/python`. The commands resolve input paths from the script location, not the current working directory. Each `--out` destination must not already exist, so a repeat run should use a new output name. They do not overwrite frozen input files. No PowerShell activation script is needed.

## What has been checked

- The quick check reconstructs 36 target-domain aggregate count rows from 21,492 per-frame rows, checks F1 arithmetic and 7,500 quality-gate truth-table cases, and checks the abstract's source/target F1 values.
- Source rematching uses all 500 archived source anchors and reports the same 150-frame held-out subset, across six methods and three IoU thresholds: 18 reported rows. Split identifiers, all six prediction files, and canonical Boreas GT are supplied.
- Target rematching preserves 597 anchors, both evaluation scopes, six methods, and all three thresholds. The test reproduces all 36 aggregate and 21,492 per-frame count rows. Official K-Radar annotations must be acquired and converted locally as described below.
- Simulation rematching reproduces eight state/threshold settings, with 274 frames in each state, from the supplied prediction and Gazebo-derived GT CSV files. It uses the original axis-aligned Box2D matching code, not the rotated public-data metric.

Checks executed for the renamed release are in `verification/rgcf_20261015/`. Other records in `verification/` were produced in September 2026. They should be distinguished from dated historical audit JSON files copied with experiment evidence. Historical preparation audits may contain stage-specific false/pending flags; these describe when those files were created and are not silently rewritten as current submission decisions.

## K Radar annotation access and local conversion

No original K-Radar sensor data or converted official annotation boxes are redistributed here. The official dataset is marked CC BY-NC-ND. Obtain its v2.1 annotation ZIP from the official repository at the experiment's recorded commit:

https://github.com/kaist-avelab/K-Radar/blob/f90c28993d1757ce0236632b4d912c92a14976b2/tools/revise_label/kradar_revised_label_v2_1.zip

Recorded ZIP SHA256: `8d8c1f3033ded1825ec740b4df45d4da2b505f5b489b902cfd4a723124788a02`.

```text
.venv\Scripts\python.exe code/prepare_kradar_gt.py --labels-zip /your/local/kradar_revised_label_v2_1.zip --out local_inputs/kradar_gt.csv
.venv\Scripts\python.exe code/verify_results.py --mode target --gt-csv local_inputs/kradar_gt.csv --out checks_target
```

The converter validates the annotation release hash, selects exactly the 597 supplied sequence-1 anchors, doubles stored half-dimensions, converts heading degrees to radians, and preserves all 1,890 annotated boxes and visibility labels. It does not translate the native OS2-64 coordinates, tune a threshold, select favorable frames, download data, or upload any file. Local tests verified that the converted annotation values match the experiment's original GT export (source-path text intentionally differs). See `DATA_SOURCES_AND_PERMISSIONS.md` for source attribution and access boundaries.

The existing evidence map and Figures 4–10 retain the numbering of the archived September manuscript; they are not the numbering of the condensed Chinese manuscript. See `CHINESE_MANUSCRIPT_MAP.md` for the current Chinese tables and figures.

## Package contents

- `data/source/`: corrected source geometry, predictions, fixed split, source-only fitted assets, calibration records, comparison metrics and candidate-change evidence.
- `data/target/`: frozen K-Radar predictions, feature records, copied source-fitted assets, anchor/synchronization manifests, per-prediction matches, per-frame statistics and aggregate/distance/source-decomposition results.
- `data/simulation/`: four-state threshold replay, Gazebo-derived GT, candidate pools, historical conditional-expert comparisons, trajectory summaries and recorded computation summaries.
- `data/gating_logic/`: quality, override and fallback tests and the archived production gate block.
- `code/`: portable verification, annotation conversion, numerical plotting and integrity checks. `research/` retains method implementations; `archived_orchestration/` is provenance/reference code requiring reconstruction of original paths, SDKs and ROS inputs, not a portable one-command pipeline.
- `provenance/`: original-to-package hashes, schema/row-count dictionary, and the file manifest.
- `figures/`: CSV-derived review plots for manuscript Figures 4–10. These reproduce plotted quantities, not the exact visual style of the publication figures. Conceptual Figures 1–3 are not distributed in this public package.

## Interpretation and reproduction scope

Public-data scores are rotated-BEV detection F1, not official 3D AP or mAP. Target scopes are the complete annotated domain (1,890 GT) and the common square with both absolute center coordinates at most 51.2 m (1,157 GT). Counts are integers; stored F1 values are fractions, and manuscript percentages multiply them by 100. The empty 0–3 m distance bin must remain unavailable, not a zero-performance observation.

K-Radar is a frozen offline transfer without pose input. One future Camera association and unverified raw Radar timestamps remain as documented; this package does not turn them into an online-causal test. Source calibration and train/holdout assignments are fixed. Do not select thresholds using target labels.

Simulation threshold replay controls admission from already generated candidate pools. A candidate admission indicator is not an unconditional gate-dispatch rate. Table 7's weighting-only/always-on/conditional comparison and the older online comparisons are distinct records; their F1 values must not be pooled. Camera dropout retains the recorded fallback for missing semantic-quality fields. ATE evaluates trajectory agreement, not obstacle-detection accuracy. Runtime summaries are recorded component computation, not complete end-to-end latency.

Raw sensor files, ROS bags, SDK-native pickles and public detector weights are not bundled. The trajectory and historical timing summaries can be inspected and replotted, but their raw-stream derivation is not rerun by these portable commands. Full model training, sensor feature extraction and ROS execution require the respective original environments and separately acquired inputs. These boundaries are listed against manuscript locations in `CLAIM_TO_FILE_MAP.md`.

Personal absolute workspace paths in selected text files have been replaced by neutral placeholders. Numerical records are unchanged. Both original and packaged hashes are preserved, and new manifests distinguish the changed bytes. No credentials, account exports, virtual environment, raw archive, or unrelated prior paper draft is included. No blanket open-source/data license is assigned by this preparation step.
