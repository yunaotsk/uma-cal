"""Checks on the frozen split files in data/splits (offline; no model, no raw data needed)."""

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from src import dataset as D

NAMES = ["cal", "A0", "A", "B", "C"]
SPLITS = Path("data/splits")
pytestmark = pytest.mark.skipif(not all((SPLITS / f"{n}.csv").exists() for n in NAMES), reason="split files not present")


@pytest.fixture(scope="module")
def sp():
    return {n: pd.read_csv(SPLITS / f"{n}.csv") for n in NAMES}


@pytest.mark.parametrize("col", ["comp_key", "parent_id", "sid"])
def test_all_five_splits_pairwise_disjoint(sp, col):
    for i, x in enumerate(NAMES):
        for y in NAMES[i + 1:]:
            assert not (set(sp[x][col]) & set(sp[y][col])), f"{x} and {y} share {col}"


def test_no_repeats_within_splits(sp):
    for n in NAMES:
        assert sp[n].comp_key.is_unique and sp[n].sid.is_unique, n
    for n in ("A", "B", "C"):
        assert sp[n].elements.is_unique, f"{n} repeats a chemical system"


def test_shift_definitions_hold_on_frozen_files(sp):
    f_block = D.F_BLOCK
    els = lambda n: {e for s in sp[n].elements for e in s.split()}  # noqa: E731
    cal_sys, cal_els = set(sp["cal"].elements), els("cal")
    assert not (cal_els & f_block) and not (els("A0") & f_block) and not (els("A") & f_block) and not (els("B") & f_block)
    assert sp["C"].elements.map(lambda s: bool(set(s.split()) & f_block)).all()
    assert set(sp["A"].elements) <= cal_sys
    assert not (set(sp["B"].elements) & cal_sys) and els("B") <= cal_els
    assert (sp["cal"].natoms <= 24).all() and all((sp[n].natoms <= 24).all() for n in NAMES)


def test_frozen_sizes_and_similarity_column(sp):
    assert {n: len(sp[n]) for n in NAMES} == {"cal": 4000, "A0": 2000, "A": 713, "B": 2000, "C": 2000}
    for n in NAMES:
        assert sp[n].max_jaccard_to_cal.between(0, 1).all()
    assert (sp["A"].max_jaccard_to_cal == 1.0).all() and (sp["B"].max_jaccard_to_cal < 1.0).all()


def test_hashes_match_split_freeze_document():
    doc = Path("reports/split_freeze.md")
    if not doc.exists():
        pytest.skip("freeze document not present")
    recorded = dict(re.findall(r"\| `(data/splits/[^`]+\.csv)` \| [\d,]+ \| `([0-9a-f]{64})` \|", doc.read_text()))
    assert set(recorded) == {f"data/splits/{n}.csv" for n in NAMES}
    for f, h in recorded.items():
        assert hashlib.sha256(Path(f).read_bytes()).hexdigest() == h, f"{f} changed since the freeze"


def test_benchmark_structures_are_all_in_cal(sp):
    b = Path("data/benchmark/cal_benchmark.json")
    if not b.exists():
        pytest.skip("benchmark file not present")
    sids = {r["sid"] for r in json.loads(b.read_text())["per_structure"]}
    assert len(sids) == 20 and sids <= set(sp["cal"].sid)
    assert not any(sids & set(sp[n].sid) for n in NAMES[1:])
