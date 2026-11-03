"""Offline tests for the prediction cache. A labelled stub predictor stands in for UMA: no weights, no network."""

import subprocess

import pytest
from ase.build import bulk

from src.pred_cache import CacheConfig, PredictionCache, predict_cached

VERS = {"torch_version": "t1", "fairchem_core_version": "f1", "ase_version": "a1"}


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
    first = predict_cached(p, _items(3), cache)
    again = predict_cached(p, _items(3), cache)
    assert first == again == {"sid0": -12.0, "sid1": -12.0, "sid2": -12.0}
    assert len(p.calls) == 3 and len(cache) == 3


def test_resumes_after_interruption(tmp_path):
    path = tmp_path / "p.sqlite"
    p = _StubPredictor()
    c1 = PredictionCache(path, versions=VERS)
    predict_cached(p, _items(2), c1)  # "interrupted" after two structures
    c1.close()
    c2 = PredictionCache(path, versions=VERS)  # new process, same file
    p2 = _StubPredictor()
    out = predict_cached(p2, _items(5), c2)
    assert len(out) == 5 and len(p2.calls) == 3  # only the three missing ones were computed
    c2.close()


def test_config_is_part_of_the_key(cache):
    predict_cached(_StubPredictor(inference_settings="batch"), _items(2), cache)
    other = _StubPredictor(inference_settings="high-precision")
    predict_cached(other, _items(2), cache)
    assert len(other.calls) == 2  # different setting: no cross-use of cached values
    assert len(cache) == 4


def test_metadata_recorded_and_versions_auditable(cache):
    predict_cached(_StubPredictor(), _items(1), cache)
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
