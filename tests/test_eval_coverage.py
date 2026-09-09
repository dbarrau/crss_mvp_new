"""Tests for the coverage audit (keeps quality_set.json a governed design).

The live gate is the point: the real set, against the real manifest, must have no
FAIL — so a future edit that breaks a floor or a ceiling fails CI instead of
quietly unbalancing the set. The detector cases pin each rule (and the CROSS
exemption we fixed) with tiny synthetic sets + tuned manifests.
"""
import json
from pathlib import Path

from scripts.audit_eval_coverage import audit, FAIL, REVIEW

ROOT = Path(__file__).resolve().parent.parent


def _fails(findings):
    return [f for f in findings if f["severity"] == FAIL]


def q(qid, route, regs, primary, diff):
    return {"id": qid, "coverage": {"route": route, "regs": regs, "primary_reg": primary, "difficulty": diff}}


# a permissive manifest; each detector tightens only the axis it exercises
def _mf(**over):
    base = {
        "routes": ["role", "definition", "cross_regulation"],
        "difficulties": ["E", "M", "H"],
        "route_floor": 2,
        "reg_floors": {"MDR": 1, "AIAct": 1},
        "difficulty_cell_ceiling": 3,
        "single_reg_buckets": ["MDR", "AIAct"],
        "single_reg_aggregate_soft": 5,
    }
    base.update(over)
    return base


def test_live_set_matches_its_manifest():
    questions = json.loads((ROOT / "eval" / "quality_set.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "eval" / "coverage_manifest.json").read_text(encoding="utf-8"))
    fails = _fails(audit(questions, manifest))
    assert not fails, "coverage failures (run scripts/audit_eval_coverage.py):\n" + \
        "\n".join(f"  {f['message']}" for f in fails)


def test_route_below_floor_fails():
    qs = [q("A", "role", ["MDR"], "MDR", "E"), q("B", "role", ["MDR"], "MDR", "M"),
          q("C", "definition", ["AIAct"], "AIAct", "E")]  # definition has only 1
    fails = _fails(audit(qs, _mf()))
    assert any("definition" in f["message"] for f in fails)


def test_reg_below_floor_fails():
    qs = [q("A", "definition", ["AIAct"], "AIAct", "E"), q("B", "definition", ["AIAct"], "AIAct", "M")]
    fails = _fails(audit(qs, _mf(reg_floors={"MDR": 2, "AIAct": 1})))  # MDR present 0×, floor 2
    assert any("MDR" in f["message"] for f in fails)


def test_single_reg_difficulty_ceiling_fails():
    qs = [q(f"R{i}", "role", ["MDR"], "MDR", "M") for i in range(4)]  # 4 in role×MDR×M
    fails = _fails(audit(qs, _mf(route_floor=1, reg_floors={"MDR": 1})))
    assert any("MDRxM" in f["message"] for f in fails)


def test_cross_bucket_exempt_from_ceiling():
    # five cross-reg cases all land in role_pair CROSS×M — must NOT fail (each spans a distinct pair)
    qs = [q(f"X{i}", "cross_regulation", ["MDR", "AIAct"], "CROSS", "M") for i in range(5)]
    findings = audit(qs, _mf(route_floor=1, reg_floors={"MDR": 1}))
    assert not _fails(findings)
    assert not [f for f in findings if f["severity"] == REVIEW]  # CROSS exempt from soft too


def test_malformed_coverage_fails():
    fails = _fails(audit([{"id": "NO_COV"}], _mf(route_floor=1, reg_floors={})))
    assert any("NO_COV" in f["message"] for f in fails)
