"""Offline tests for the prediction cache. A labelled stub predictor stands in for UMA: no weights, no network."""

import subprocess

import pytest
from ase.build import bulk

import src.pred_cache as pc
from src.pred_cache import CacheConfig, PredictionCache, SplitGuard, a0_gate_decision, predict_cached, record_a0_gate

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


def test_guard_permits_only_cal_and_a0_until_gate_passed(guard):
    allowed = guard.allowed_sids()
    assert allowed == {f"{n}_{i}" for n in ("cal", "A0") for i in range(3)}
    for locked in ("A_0", "B_1", "C_2"):
        assert locked not in allowed


def test_failed_gate_keeps_a_b_c_locked(guard):
    assert record_a0_gate(0.80, n_a0=2000, n_cal=4000, gate_path=guard.gate_path) is False
    assert not guard.gate_passed()
    assert "B_0" not in guard.allowed_sids()


def test_passed_gate_unlocks_a_b_c(guard):
    assert record_a0_gate(0.905, n_a0=2000, n_cal=4000, gate_path=guard.gate_path) is True
    assert guard.gate_passed()
    assert guard.allowed_sids() == {f"{n}_{i}" for n in ("cal", "A0", "A", "B", "C") for i in range(3)}
    import json
    rec = json.loads(guard.gate_path.read_text())
    assert rec["passed"] is True and rec["a0_coverage"] == 0.905 and rec["n_a0"] == 2000 and rec["decided_utc"]


@pytest.mark.parametrize("cov,expected", [(0.90, True), (0.88, True), (0.92, True), (0.8799, False), (0.9201, False), (0.5, False)])
def test_gate_decision_band_is_two_points(cov, expected):
    assert a0_gate_decision(cov) is expected


def test_default_predict_path_is_guarded(guard, monkeypatch, cache):
    """allowed_sids=None must use the gate-aware guard: A/B/C refused before the gate, accepted after."""
    monkeypatch.setattr(pc, "SplitGuard", lambda: guard)
    p = _StubPredictor()
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    assert predict_cached(p, [("cal_0", atoms), ("A0_1", atoms)], cache) == {"cal_0": -12.0, "A0_1": -12.0}
    with pytest.raises(PermissionError):
        predict_cached(p, [("cal_1", atoms), ("B_0", atoms)], cache)
    assert len(p.calls) == 2 and cache.get("cal_1", CacheConfig.from_predictor(p)) is None  # nothing predicted for the refused batch
    record_a0_gate(0.91, 2000, 4000, gate_path=guard.gate_path)
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
