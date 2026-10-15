# Method naming

The method is **RGCF**, short for **Reliability-Guided Candidate Fusion** (可靠性引导的候选融合).

All active labels, file paths, model identifiers and code references in this release use RGCF. Source and transfer configurations keep their existing suffixes: for example, `RGCF_B2_Causal` and `RGCF_FrozenTransfer_NoPose`. The `transfer*` marker retains its original meaning: frozen transfer without pose input.

Historical aliases, source paths and original hashes appear only in clearly labelled provenance records. These records document a rename of the same method and experiment, not a different algorithm. Numerical records, source/target splits, geometry, thresholds, fitted weights, sample counts and dates have not been changed. Existing archived orchestration scripts retain their original external environment requirements.
