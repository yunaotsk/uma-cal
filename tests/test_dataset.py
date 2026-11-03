"""Offline tests for composition extraction, classification and split construction.

Split tests use a synthetic metadata table; no OMat24 files or model weights are needed.
"""

import numpy as np
import pandas as pd
import pytest
from ase.build import bulk

from src import dataset as D


def test_reduced_composition_gcd():
    assert D.reduced_composition(bulk("Al", "fcc", a=4.05, cubic=True).numbers) == {"Al": 1}
    # Dy4 N4 Si4 -> Dy1 N1 Si1 ; counts given as atomic numbers
    nums = [66] * 4 + [7] * 4 + [14] * 4
    assert D.reduced_composition(nums) == {"Dy": 1, "N": 1, "Si": 1}
    assert D.composition_key(D.reduced_composition(nums)) == "Dy1N1Si1"


def test_reduced_composition_not_reducible_and_peroxide():
    nums = [12] * 2 + [22] * 3 + [25] * 1 + [16] * 8  # Mg2 Ti3 Mn1 S8
    assert D.reduced_composition(nums) == {"Mg": 2, "Mn": 1, "S": 8, "Ti": 3}
    assert D.reduced_composition([3, 3, 8, 8]) == {"Li": 1, "O": 1}  # Li2O2 -> LiO


def test_composition_key_order_independent_and_roundtrip():
    a = D.composition_key({"O": 2, "Fe": 1})
    b = D.composition_key({"Fe": 1, "O": 2})
    assert a == b == "Fe1O2"
    assert D.parse_composition_key("Co1K1Os4") == {"Co": 1, "K": 1, "Os": 4}


def test_empty_structure_rejected():
    with pytest.raises(ValueError):
        D.reduced_composition([])


@pytest.mark.parametrize(
    "els,family",
    [
        (["Al"], "elemental"),
        (["Fe", "O"], "oxide"),
        (["Ca", "P", "O"], "oxide"),  # phosphate: P is not a strong anion
        (["Li", "F"], "halide"),
        (["Zn", "S"], "chalcogenide"),
        (["Ga", "N"], "nitride"),
        (["Ti", "O", "N"], "mixed-anion"),
        (["Cu", "Cl", "O"], "mixed-anion"),
        (["Zn", "P"], "pnictide"),
        (["Mg", "H"], "hydride"),
        (["Ti", "C"], "carbide"),
        (["Fe", "Si"], "silicide"),
        (["Fe", "B", "C"], "mixed-light-element"),
        (["Cu", "Zr", "Ag"], "intermetallic"),
        (["Ni", "Sb"], "intermetallic"),
    ],
)
def test_chemical_family_rules(els, family):
    assert D.chemical_family(els) == family


def test_tm_classes_and_hubbard_rule():
    assert D.tm_class(["Fe", "Zr"]) == "3d+4d"
    assert D.tm_class(["Fe", "Al"]) == "3d-only"
    assert D.tm_class(["Mo", "Al"]) == "4d-only"
    assert D.tm_class(["Al", "Mg"]) == "neither"
    assert D.hubbard_u_applies(["Fe", "O"])
    assert D.hubbard_u_applies(["Ni", "F", "K"])
    assert not D.hubbard_u_applies(["Fe", "S"])  # sulfide: no U
    assert not D.hubbard_u_applies(["Zn", "O"])  # Zn is not a U element
    assert not D.hubbard_u_applies(["Fe", "Co"])  # intermetallic


def test_binomial_se():
    assert D.binomial_se(0.9, 1000) == pytest.approx(0.0094868, rel=1e-4)
    assert D.binomial_se(0.9, 100) > D.binomial_se(0.9, 1000)


def test_assert_disjoint_detects_overlap():
    a = pd.DataFrame({"comp_key": ["A1B1", "C1D1"]})
    b = pd.DataFrame({"comp_key": ["C1D1", "E1F1"]})
    with pytest.raises(AssertionError):
        D.assert_disjoint(a, b)
    D.assert_disjoint(a, b.iloc[1:])


def _synthetic(seed=0, n=6000):
    rng = np.random.default_rng(seed)
    alphabet = ["Li", "Na", "K", "Mg", "Ca", "Al", "Ga", "In", "Zn", "Cu", "Ag", "Ni", "Zr", "Nb", "Ru",
                "S", "Se", "Cl", "Br", "N", "P", "Si", "Ge", "Sn", "Pb", "Bi", "Dy", "Er", "La", "Th", "U"]
    rows, parent = [], 0
    for _ in range(n):
        k = int(rng.choice([2, 3, 4], p=[0.2, 0.6, 0.2]))
        els = sorted(rng.choice(alphabet, size=k, replace=False))
        counts = {e: int(rng.integers(1, 4)) for e in els}
        key = D.composition_key(counts)
        parent += 1
        for _ in range(int(rng.choice([1, 2], p=[0.8, 0.2]))):  # rattled siblings share composition + parent
            i = len(rows)
            rows.append(dict(index=i, sid=f"s{i}", parent_id=f"p{parent}", comp_key=key, elements=" ".join(els),
                             n_elements=k, natoms=int(rng.integers(10, 30)), hubbard_u=bool(rng.random() < 0.02)))
    return pd.DataFrame(rows).drop_duplicates(["comp_key", "sid"])


@pytest.fixture(scope="module")
def synth():
    return _synthetic()


def test_eligibility_filters(synth):
    e = D.eligible_structures(synth, max_atoms=24)
    assert (e.natoms <= 24).all() and (~e.hubbard_u).all() and (e.n_elements >= 2).all()


def test_splits_disjoint_and_shift_properties(synth):
    s = D.build_splits(synth, seed=1, n_cal=300, n_a=80, n_b=80, n_c=80)
    names = list(s)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            assert not (set(s[x].comp_key) & set(s[y].comp_key))
            assert not (set(s[x].sid) & set(s[y].sid))
            assert not (set(s[x].parent_id) & set(s[y].parent_id))
    cal_sys, cal_els = set(s["cal"].elements), {e for x in s["cal"].elements for e in x.split()}
    assert set(s["A"].elements) <= cal_sys
    assert not (set(s["B"].elements) & cal_sys)
    assert not (cal_els & D.F_BLOCK)
    assert s["C"].elements.map(lambda x: bool(set(x.split()) & D.F_BLOCK)).all()
    assert len(s["A"]) > 0 and len(s["B"]) > 0 and len(s["C"]) > 0
    for name in "ABC":
        assert s[name].comp_key.is_unique


def test_splits_deterministic_and_seed_sensitive(synth):
    a = D.build_splits(synth, seed=7, n_cal=200, n_a=50, n_b=50, n_c=50)
    b = D.build_splits(synth, seed=7, n_cal=200, n_a=50, n_b=50, n_c=50)
    c = D.build_splits(synth, seed=8, n_cal=200, n_a=50, n_b=50, n_c=50)
    for k in a:
        assert a[k].sid.tolist() == b[k].sid.tolist()
    assert a["cal"].sid.tolist() != c["cal"].sid.tolist()


def test_invariant_checker_catches_leak(synth):
    s = D.build_splits(synth, seed=1, n_cal=200, n_a=50, n_b=50, n_c=50)
    leaked = dict(s)
    leaked["B"] = pd.concat([s["B"], s["cal"].iloc[:1]])
    with pytest.raises(AssertionError):
        D.check_split_invariants(leaked)
