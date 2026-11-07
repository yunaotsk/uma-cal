"""Offline tests for the prediction cache. A labelled stub predictor stands in for UMA: no weights, no network."""

import subprocess

import pytest
from ase.build import bulk

import src.pred_cache as pc
from src.pred_cache import (CacheConfig, PredictionCache, SplitGuard, a0_gate_decision, a0_gate_summary, conformal_quantile,
                            predict_cached, record_a0_gate, wilson_interval)

VERS = {"torch_version": "t1", "fairchem_core_version": "f1", "ase_version": "a1"}
ALLOW = {f"sid{i}" for i in range(10)}  # explicit allow-list for stub structures (default guard would reject them)


class _StubPredictor:
    """Test double (energy = -3 eV per atom). Not a UMA prediction."""

    def __init__(self, checkpoint="uma-s-1p2p1", task="omat", inference_settings="batch", device="cpu"):
        self.checkpoint, self.task, self.inference_settings, self.device = checkpoint, task, inference_settings, device
        self.calls = []

    def predict_energy(self, atoms):
        self.calls.append(len(atoms))
        return -3.0 * len(atoms)


def _items(n):
    return [(f"sid{i}", bulk("Cu", "fcc", a=3.6, cubic=True)) for i in range(n)]


@pytest.fixture
def cache(tmp_path):
    c = PredictionCache(tmp_path / "c" / "p.sqlite", versions=VERS)
    yield c
    c.close()


def test_hits_are_not_recomputed(cache):
    p = _StubPredictor()
    first = predict_cached(p, _items(3), cache, allowed_sids=ALLOW)
    again = predict_cached(p, _items(3), cache, allowed_sids=ALLOW)
    assert first == again == {"sid0": -12.0, "sid1": -12.0, "sid2": -12.0}
    assert len(p.calls) == 3 and len(cache) == 3


def test_resumes_after_interruption(tmp_path):
    path = tmp_path / "p.sqlite"
    p = _StubPredictor()
    c1 = PredictionCache(path, versions=VERS)
    predict_cached(p, _items(2), c1, allowed_sids=ALLOW)  # "interrupted" after two structures
    c1.close()
    c2 = PredictionCache(path, versions=VERS)  # new process, same file
    p2 = _StubPredictor()
    out = predict_cached(p2, _items(5), c2, allowed_sids=ALLOW)
    assert len(out) == 5 and len(p2.calls) == 3  # only the three missing ones were computed
    c2.close()


def test_config_is_part_of_the_key(cache):
    predict_cached(_StubPredictor(inference_settings="batch"), _items(2), cache, allowed_sids=ALLOW)
    other = _StubPredictor(inference_settings="high-precision")
    predict_cached(other, _items(2), cache, allowed_sids=ALLOW)
    assert len(other.calls) == 2  # different setting: no cross-use of cached values
    assert len(cache) == 4


def test_metadata_recorded_and_versions_auditable(cache):
    predict_cached(_StubPredictor(), _items(1), cache, allowed_sids=ALLOW)
    (row,) = cache.rows()
    assert (row["checkpoint"], row["task"], row["inference_settings"], row["device"]) == ("uma-s-1p2p1", "omat", "batch", "cpu")
    assert (row["torch_version"], row["fairchem_core_version"], row["ase_version"]) == ("t1", "f1", "a1")
    assert row["natoms"] == 4 and row["seconds"] >= 0 and row["created_utc"]
    assert len(cache.distinct_versions()) == 1


def test_allowed_sids_guard_blocks_before_any_prediction(cache):
    p = _StubPredictor()
    items = _items(3)
    with pytest.raises(PermissionError):
        predict_cached(p, items, cache, allowed_sids={"sid0", "sid1"})
    assert p.calls == [] and len(cache) == 0
    assert set(predict_cached(p, items[:2], cache, allowed_sids={"sid0", "sid1"})) == {"sid0", "sid1"}


def test_config_id_stable_and_distinct():
    a = CacheConfig("m", "omat", "batch", "cpu")
    assert a.config_id == CacheConfig("m", "omat", "batch", "cpu").config_id
    assert a.config_id != CacheConfig("m", "omat", "batch", "cuda").config_id


def test_cache_files_are_gitignored():
    r = subprocess.run(["git", "check-ignore", "-q", "data/cache/predictions.sqlite"], capture_output=True)
    assert r.returncode == 0


# ------------------------------------------------------------------ A0 gate guard ---

def _write_splits(d, sizes=None):
    """Tiny synthetic split files: cal0.., a00.., A_.., B_.., C_.. sids."""
    d.mkdir(parents=True, exist_ok=True)
    for name in ("cal", "A0", "A", "B", "C"):
        (d / f"{name}.csv").write_text("sid\n" + "".join(f"{name}_{i}\n" for i in range(3)))
    return d


@pytest.fixture
def guard(tmp_path):
    return SplitGuard(_write_splits(tmp_path / "splits"), tmp_path / "cache" / "gate.json")


CFG = CacheConfig("uma-s-1p2p1", "omat", "batch", "cpu")


def _synthetic_scores(kind):
    """Deterministic absolute-error scores (no model): A0 exchangeable with Cal ('pass') or heavier-tailed ('fail')."""
    cal = [i / 3999 for i in range(4000)]
    a0 = [i / 1999 for i in range(2000)] if kind == "pass" else [1.25 * i / 1999 for i in range(2000)]
    return cal, a0


def _record(guard, tmp_path, kind, **kw):
    cal, a0 = _synthetic_scores(kind)
    args = dict(gate_path=guard.gate_path, report_path=tmp_path / "reports" / "a0_gate_result.md", versions=VERS,
                code_commit="abc1234", code_dirty=False, splits_dir=guard.splits_dir)
    args.update(kw)
    return record_a0_gate(cal, a0, CFG, **args)


def test_guard_permits_only_cal_and_a0_until_gate_passed(guard):
    allowed = guard.allowed_sids()
    assert allowed == {f"{n}_{i}" for n in ("cal", "A0") for i in range(3)}
    for locked in ("A_0", "B_1", "C_2"):
        assert locked not in allowed


def test_failed_gate_keeps_a_b_c_locked(guard, tmp_path):
    assert _record(guard, tmp_path, "fail") is False
    assert not guard.gate_passed()
    assert "B_0" not in guard.allowed_sids()


def test_passed_gate_unlocks_a_b_c(guard, tmp_path):
    assert _record(guard, tmp_path, "pass") is True
    assert guard.gate_passed()
    assert guard.allowed_sids() == {f"{n}_{i}" for n in ("cal", "A0", "A", "B", "C") for i in range(3)}
    import json
    rec = json.loads(guard.gate_path.read_text())
    assert rec["passed"] is True and rec["a0_coverage"] == pytest.approx(0.90, abs=0.002) and rec["n_a0"] == 2000 and rec["n_cal"] == 4000
    assert rec["decided_utc"] and rec["code_commit"] == "abc1234" and set(rec["levels"]) == {"0.80", "0.90", "0.95"}


@pytest.mark.parametrize("cov,expected", [(0.90, True), (0.88, True), (0.92, True), (0.8799, False), (0.9201, False), (0.5, False)])
def test_gate_decision_band_is_two_points(cov, expected):
    assert a0_gate_decision(cov) is expected


def test_default_predict_path_is_guarded(guard, monkeypatch, cache, tmp_path):
    """allowed_sids=None must use the gate-aware guard: A/B/C refused before the gate, accepted after."""
    monkeypatch.setattr(pc, "SplitGuard", lambda: guard)
    p = _StubPredictor()
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    assert predict_cached(p, [("cal_0", atoms), ("A0_1", atoms)], cache) == {"cal_0": -12.0, "A0_1": -12.0}
    with pytest.raises(PermissionError):
        predict_cached(p, [("cal_1", atoms), ("B_0", atoms)], cache)
    assert len(p.calls) == 2 and cache.get("cal_1", CacheConfig.from_predictor(p)) is None  # nothing predicted for the refused batch
    _record(guard, tmp_path, "pass")
    assert "B_0" in predict_cached(p, [("B_0", atoms)], cache)


def test_real_committed_splits_lock_a_b_c_without_gate(tmp_path):
    """On the real split files: with no gate record, only Cal and A0 sids are allowed."""
    import csv
    from pathlib import Path

    d = Path("data/splits")
    if not (d / "A0.csv").exists():
        pytest.skip("split files not present")
    g = SplitGuard(d, tmp_path / "no_gate.json")
    sids = lambda n: {r["sid"] for r in csv.DictReader(open(d / f"{n}.csv", newline=""))}  # noqa: E731
    allowed = g.allowed_sids()
    assert allowed == sids("cal") | sids("A0")
    for locked in ("A", "B", "C"):
        assert not (allowed & sids(locked))


# ----------------------------------------------------- gate maths and the committed report ---

def test_conformal_quantile_index_and_infinite_case():
    xs = list(range(1, 10))  # n = 9
    assert conformal_quantile(xs, 0.10) == 9  # k = ceil(10 * 0.9) = 9
    assert conformal_quantile(xs, 0.50) == 5  # k = ceil(10 * 0.5) = 5
    assert conformal_quantile(range(5), 0.10) == float("inf")  # k = ceil(6 * 0.9) = 6 > 5
    cal = [i / 3999 for i in range(4000)]
    assert conformal_quantile(cal, 0.10) == pytest.approx(3600 / 3999)  # k = 3601 for n = 4000


def test_wilson_interval_known_value():
    lo, hi = wilson_interval(90, 100)
    assert (lo, hi) == pytest.approx((0.8256, 0.9448), abs=1e-3)
    assert wilson_interval(0, 50)[0] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        wilson_interval(1, 0)


def test_summary_levels_k_and_exchangeable_coverage():
    cal, a0 = _synthetic_scores("pass")
    sm = a0_gate_summary(cal, a0)
    assert (sm["n_cal"], sm["n_a0"]) == (4000, 2000) and set(sm["levels"]) == {0.80, 0.90, 0.95}
    assert [sm["levels"][l]["k"] for l in (0.80, 0.90, 0.95)] == [3201, 3601, 3801]
    qs = [sm["levels"][l]["q_eV_per_atom"] for l in (0.80, 0.90, 0.95)]
    assert qs == sorted(qs)
    for level, r in sm["levels"].items():
        assert r["coverage"] == pytest.approx(level, abs=0.01)
        assert r["wilson95"][0] < r["coverage"] < r["wilson95"][1]
        assert r["se_composition_level"] == pytest.approx((r["coverage"] * (1 - r["coverage"]) / 2000) ** 0.5)


def test_report_written_on_pass_with_all_required_fields(guard, tmp_path):
    assert _record(guard, tmp_path, "pass") is True
    text = (tmp_path / "reports" / "a0_gate_result.md").read_text()
    for needle in ("**Result: PASS**", "n_cal = 4,000", "n_a0 = 2,000", "| 80% |", "| 90% |", "| 95% |", "Wilson 95%", "SE (composition level)",
                   "abc1234", "uma-s-1p2p1", "`omat`", "`batch`", "`cpu`", "torch_version: t1", "fairchem_core_version: f1", "ase_version: a1", "Decided 20"):
        assert needle in text, needle
    import hashlib
    for name in ("cal", "A0"):
        h = hashlib.sha256((guard.splits_dir / f"{name}.csv").read_bytes()).hexdigest()
        assert h in text


def test_report_is_written_even_when_the_gate_fails(guard, tmp_path):
    report = tmp_path / "reports" / "a0_gate_result.md"
    assert not report.exists()
    assert _record(guard, tmp_path, "fail") is False
    assert report.exists()
    text = report.read_text()
    assert "**Result: FAIL**" in text and "| 90% |" in text and "abc1234" in text and "torch_version: t1" in text
    assert not guard.gate_passed() and "B_0" not in guard.allowed_sids()  # A, B, C stay locked


def test_report_numbers_match_gate_record(guard, tmp_path):
    import json
    _record(guard, tmp_path, "pass")
    rec = json.loads(guard.gate_path.read_text())
    text = (tmp_path / "reports" / "a0_gate_result.md").read_text()
    assert f"{rec['levels']['0.90']['coverage']:.4f}" in text and f"{rec['levels']['0.90']['q']:.5f}" in text


def test_gate_is_decided_once_unless_overwrite(guard, tmp_path):
    _record(guard, tmp_path, "fail")
    with pytest.raises(FileExistsError):
        _record(guard, tmp_path, "pass")
    assert not guard.gate_passed()  # the refused call changed nothing
    assert _record(guard, tmp_path, "pass", overwrite=True) is True and guard.gate_passed()


def test_dirty_working_tree_is_flagged_in_report(guard, tmp_path):
    _record(guard, tmp_path, "pass", code_dirty=True)
    assert "uncommitted changes" in (tmp_path / "reports" / "a0_gate_result.md").read_text()


def test_empty_scores_rejected(guard, tmp_path):
    with pytest.raises(ValueError):
        record_a0_gate([], [0.1], CFG, gate_path=guard.gate_path, report_path=tmp_path / "r.md", versions=VERS, code_commit="x", code_dirty=False)
    assert not guard.gate_path.exists()
