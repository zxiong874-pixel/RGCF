#!/usr/bin/env python3
"""Execute the actual node's AST gate block; do not import or alter ROS runtime."""
import ast
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src/bev_obstacle_detection/src"))
from native_geometry_audit import dump, sha, write_csv
from bev_obstacle_detection.bev_grid import BevGridSpec
from bev_obstacle_detection.camera_semantic_candidates import camera_semantic_candidates
from bev_obstacle_detection.radar_surface_candidates import radar_surface_candidates
from bev_obstacle_detection.multimodal_support_fusion import (
    adaptive_support_weights, fuse_bev_support_maps, TemporalModalityReliabilityController,
)
from bev_obstacle_detection.candidate_temporal_confirmation import (
    CandidateTemporalConfirmationTracker, _candidate_position,
)

NODE = ROOT / "src/bev_obstacle_detection/scripts/bev_obstacle_node.py"


def actual_block():
    source = NODE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for method in ast.walk(tree):
        if not isinstance(method, ast.FunctionDef):
            continue
        for start, node in enumerate(method.body):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
                    t.id == "radar_candidate_gate_active" for t in node.targets):
                for end in range(start, len(method.body)):
                    n = method.body[end]
                    if isinstance(n, ast.If) and isinstance(n.test, ast.Attribute) and n.test.attr == "_last_camera_candidate_gate_active":
                        body = method.body[start:end + 1]
                        block = ast.Module(body=body, type_ignores=[])
                        code = compile(ast.fix_missing_locations(block), str(NODE), "exec")
                        defaults = {}
                        for assignment in ast.walk(tree):
                            if not isinstance(assignment, ast.Assign) or len(assignment.targets) != 1:
                                continue
                            target = assignment.targets[0]
                            if not isinstance(target, ast.Attribute) or not isinstance(target.value, ast.Name) or target.value.id != "self":
                                continue
                            for call in ast.walk(assignment.value):
                                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "get_param" and len(call.args) == 2:
                                    try:
                                        defaults[target.attr] = ast.literal_eval(call.args[1])
                                    except (ValueError, TypeError):
                                        pass
                        snippet = "\n".join(source.splitlines()[body[0].lineno - 1:body[-1].end_lineno])
                        return code, defaults, snippet, (body[0].lineno, body[-1].end_lineno)
    raise RuntimeError("Could not locate exact production gate block")


def execute(code, defaults, ql, qr, qi, qs, points, mask, threshold=.12,
            enabled_r=True, enabled_c=True, force_r=False, force_c=False):
    state = SimpleNamespace(**defaults)
    state.grid_spec = None
    state.use_reliability_gated_radar_candidates = enabled_r
    state.use_reliability_gated_camera_candidates = enabled_c
    state.force_radar_candidate_generation = force_r
    state.force_camera_candidate_generation = force_c
    state.radar_candidate_max_lidar_quality = threshold
    state.camera_candidate_max_lidar_quality = threshold
    state._last_frame_modality_qualities = {"lidar": ql, "radar": qr}
    state._last_camera_raw_image_quality = qi
    state._last_camera_semantic_quality = qs
    state._last_camera_semantic_mask = mask
    state._last_candidate_radar_support_points = np.zeros((points, 3))
    calls = []
    def radar(*args, **kwargs):
        calls.append("Radar")
        return ["Radar"]
    def camera(*args, **kwargs):
        calls.append("Camera")
        return ["Camera"]
    scope = dict(self=state, obstacles=[], radar_surface_candidates=radar,
                 camera_semantic_candidates=camera,
                 merge_obstacle_candidates=lambda a, b, **kw: a + b)
    exec(code, scope)
    return (state._last_radar_candidate_gate_active,
            state._last_camera_candidate_gate_active, "+".join(calls))


def fallback_tests():
    records = []
    spec = BevGridSpec(-2, 15, -8, 8, -2, 3, .2)
    zero = np.zeros((10, 10), np.float32)
    for minimum in [0.0, .2]:
        q = dict(lidar=0., radar=0., camera=0.)
        w = adaptive_support_weights(q, min_lidar_weight=minimum)
        occupancy, score = fuse_bev_support_maps(zero, {"camera": zero, "radar": zero}, w)
        assert all(np.isfinite(list(w.values()))) and np.isfinite(score).all() and not occupancy.any()
        records.append(dict(test="all_modalities_unavailable_minL_" + str(minimum), status="pass", detail=str(w)))
    assert adaptive_support_weights({}) == {}
    assert camera_semantic_candidates(np.zeros((100,100),np.uint8), spec) == ()
    assert radar_surface_candidates(np.empty((0,3)), spec) == ()
    tracker = CandidateTemporalConfirmationTracker()
    assert tracker.update([], [], 1., world_from_body=None) == ((), ())
    assert _candidate_position(SimpleNamespace(center_x=2., center_y=3.), None) == (2.,3.,"body")
    records.append(dict(test="empty_candidates_empty_mask_no_pose",status="pass",detail="Empty tuples; no-pose coordinates remain body-relative"))
    ctrl = TemporalModalityReliabilityController(rise_alpha=1., fall_alpha=1.)
    ctrl.update({"radar": .12})
    assert ctrl.active["radar"]
    ctrl.update({"radar": .05})
    assert not ctrl.active["radar"]
    ctrl.update({"radar": .12})
    assert ctrl.active["radar"]
    ctrl.reset()
    assert not ctrl.active and not ctrl.smoothed_qualities
    weights = ctrl.update({"lidar": 0, "camera": 0, "radar": 0})
    assert weights == dict(camera=0., lidar=0., radar=0.)
    records.append(dict(test="hysteresis_threshold_equality_and_explicit_reset",status="pass",detail="<=failure exits; >=recovery enters; reset clears state"))
    records.append(dict(test="automatic_node_sequence_reset",status="not_implemented",detail="Production node has no reliability-controller.reset call; fresh process/explicit lifecycle reset required between independent sequences"))
    return records


def main():
    out = Path(__file__).resolve().parents[1] / "D_gating/actual_logic"
    out.mkdir(parents=True, exist_ok=False)
    code, defaults, snippet, lines = actual_block()
    (out / "production_gate_block.txt").write_text(snippet + "\n", encoding="utf-8")
    rows = []
    for threshold in [.12,.20]:
        qls = [0, threshold-1e-9, threshold, threshold+1e-9, 1]
        qrs = [0,.12-1e-9,.12,.12+1e-9,1]
        qcs = [0,.10-1e-9,.10,.10+1e-9,1]
        for ql, qr, qi, qs, points, mask_key in itertools.product(qls,qrs,qcs,qcs,[0,3],["absent","empty","foreground"]):
            mask = None if mask_key == "absent" else np.full((2,2), 0 if mask_key == "empty" else 255, np.uint8)
            actual_r, actual_c, order = execute(code,defaults,ql,qr,qi,qs,points,mask,threshold)
            expected_r = ql <= threshold and points > 0
            expected_c = ql <= threshold and qr <= .12 and qi >= .10 and qs >= .10 and mask is not None
            assert (actual_r,actual_c) == (expected_r,expected_c)
            assert order == "+".join(k for k,v in [("Radar",expected_r),("Camera",expected_c)] if v)
            rows.append(dict(qLmax=threshold,qL=ql,qR=qr,qC_image=qi,qC_semantic=qs,
                radar_support_points=points,mask_state=mask_key,radar_gate=actual_r,
                camera_gate=actual_c,dispatch_order=order,both_active=actual_r and actual_c))
    override_rows = []
    for er,ec,fr,fc,ql,qr,qi,qs,points,available in itertools.product(
            [False,True],[False,True],[False,True],[False,True],[0.,1.],[0.,1.],[0.,1.],[0.,1.],[0,3],[False,True]):
        mask = np.ones((2,2),np.uint8) if available else None
        r,c,order = execute(code,defaults,ql,qr,qi,qs,points,mask,.12,er,ec,fr,fc)
        expected = (er and (fr or ql <= .12) and points > 0,
                    ec and (fc or (ql <= .12 and qr <= .12)) and qi >= .1 and qs >= .1 and available)
        assert (r,c) == expected
        override_rows.append(dict(enabled_r=er,enabled_c=ec,force_r=fr,force_c=fc,qL=ql,qR=qr,
            qC_image=qi,qC_semantic=qs,radar_points=points,mask_available=available,radar_gate=r,camera_gate=c))
    write_csv(out / "gating_truth_table.csv",rows)
    write_csv(out / "force_disable_truth_table.csv",override_rows)
    write_csv(out / "fallback_tests.csv",fallback_tests())
    report = dict(status="pass", production_node=str(NODE), production_sha256=sha(NODE),
        source_lines=lines, gate_cases=len(rows), override_cases=len(override_rows),
        both_active_cases=sum(r["both_active"] for r in rows), source_code_modified=False,
        radar_formula="E_R AND (F_R OR q_L <= tau_LR) AND (N_R_filtered > 0)",
        camera_formula="E_C AND (F_C OR ((q_L <= tau_LC) AND (q_R <= tau_R))) AND (q_C_image >= tau_I) AND (q_C_semantic >= tau_S) AND (mask IS NOT None)",
        radar_availability="Nonempty candidate-support point array after optional consistency filtering, not merely a message and not a q_R minimum",
        camera_availability="Non-None semantic mask AND both raw-image/time quality and semantic quality passing inclusive thresholds",
        exclusive=False, order="Radar merge then Camera merge", expert_gate_has_hysteresis=False,
        weights_have_separate_hysteresis=True, gates_use="instantaneous _last_frame_modality_qualities, not smoothed weight quality",
        automatic_controller_segment_reset=False,
        scope="Actual production branch AST with instrumented candidate generators; real generator/weight/tracker empty-input functions separately tested; not an end-to-end ROS replay",
        snapshot_defaults={k:v for k,v in defaults.items() if "candidate" in k or "min_lidar_fusion_weight" in k})
    dump(out / "gating_audit.json",report)
    dump(out / "output_sha256.json",{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps({k:report[k] for k in ["status","gate_cases","override_cases","both_active_cases","exclusive","automatic_controller_segment_reset"]},indent=2))


if __name__ == "__main__":
    main()
