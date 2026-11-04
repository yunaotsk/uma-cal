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
                             n_elements=k, natoms=int(rng.integers(10, 30)), hubbard_u=bool(rng.random() < 0.02),
                             family=D.chemical_family(els), tm_class=D.tm_class(els),
                             prototype_label=f"P{i % 7}:{'-'.join(els)}", parent_prototype_label=f"P{parent % 7}:{'-'.join(els)}",
                             energy_per_atom_eV=-4.0 + float(rng.normal()), max_abs_force_eV_A=float(abs(rng.normal()) * 5)))
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


# ---------------------------------------------------------------- ported code ---


def test_find_aselmdb_files_recursive_file_and_missing(tmp_path):
    nested = tmp_path / "extract" / "name"
    nested.mkdir(parents=True)
    (nested / "data.aselmdb").write_bytes(b"")
    (nested / "data.aselmdb-lock").write_bytes(b"")  # must not be picked up
    (nested / "metadata.npz").write_bytes(b"")
    assert D.find_aselmdb_files(tmp_path / "extract") == [str(nested / "data.aselmdb")]
    assert D.find_aselmdb_files(nested / "data.aselmdb") == [str(nested / "data.aselmdb")]
    with pytest.raises(FileNotFoundError):
        D.find_aselmdb_files(tmp_path / "extract" / "empty_dir_does_not_exist")


def test_load_metadata_from_nested_aselmdb(tmp_path):
    """Regression test: the tarball extracts to <dir>/<name>/data.aselmdb, which AseDBDataset cannot find on its own."""
    import ase.db
    from ase.calculators.singlepoint import SinglePointCalculator

    nested = tmp_path / "extract" / "rattled-1000-subsampled"
    nested.mkdir(parents=True)
    db = ase.db.connect(str(nested / "data.aselmdb"))
    for i, (sym, a0) in enumerate([("Cu", 3.6), ("Al", 4.05)]):
        at = bulk(sym, "fcc", a=a0, cubic=True)
        at.calc = SinglePointCalculator(at, energy=-4.0 * len(at), forces=[[0.0, 0.0, 0.0]] * len(at), stress=[0.0] * 6)
        db.write(at, data={"sid": f"sid{i}", "parent_id": f"par{i}", "calc_id": "c", "task_type": "Static",
                           "prototype_label": "A_cF4_225_a:" + sym, "parent_prototype_label": "A_cF4_225_a:" + sym,
                           "composition_reduced": sym + "1"})
    df = D.load_metadata(tmp_path / "extract")
    assert len(df) == 2
    assert df.sid.tolist() == ["sid0", "sid1"] and df.parent_id.tolist() == ["par0", "par1"]
    assert df.comp_key.tolist() == ["Cu1", "Al1"]
    assert df.energy_per_atom_eV.tolist() == pytest.approx([-4.0, -4.0])
    assert df.hubbard_u.sum() == 0


def test_coverage_se_levels():
    lv = D.coverage_se_levels(n_structures=1250, n_compositions=1000)
    assert lv["se_composition_level"] == pytest.approx(0.0094868, rel=1e-4)
    assert lv["se_structure_level"] < lv["se_composition_level"]  # structure level is the optimistic one
    assert D.coverage_se_levels(1000, 1000)["se_structure_level"] == pytest.approx(lv["se_composition_level"])
    with pytest.raises(ValueError):
        D.coverage_se_levels(0, 10)


def test_resolve_element_spec():
    assert D.resolve_element_spec(["Ru", "Rh"]) == {"Ru", "Rh"}
    assert D.resolve_element_spec(["f-block"]) == D.F_BLOCK
    assert D.resolve_element_spec(["lanthanides", "U"]) == D.LANTHANIDES | {"U"}
    for bad in (["Xx"], ["ru"], []):
        with pytest.raises(ValueError):
            D.resolve_element_spec(bad)


def test_family_rules_vs_alternative():
    phosphate, carbonate = ["Ca", "P", "O"], ["Ca", "C", "O"]
    assert D.chemical_family(phosphate) == "oxide" and D.chemical_family(carbonate) == "oxide"
    assert D.family_alternative(phosphate) == "mixed-anion" and D.family_alternative(carbonate) == "mixed-anion"
    assert D.chemical_family(["Fe", "Si"]) == "silicide"
    assert D.family_alternative(["Fe", "Si"]) == "metalloid-containing"
    assert D.family_alternative(["Fe", "Co"]) == "intermetallic"
    assert D.family_alternative(["He", "Na"]) == "other"


def test_min_comps_per_element_enforced_for_b(synth):
    kw = dict(seed=3, n_cal=300, n_a=50, n_b=200, n_c=50)
    loose = D.build_splits(synth, min_comps_per_element=1, **kw)
    strict = D.build_splits(synth, min_comps_per_element=40, **kw)
    cal_counts = pd.Series([e for s in strict["cal"].elements for e in s.split()]).value_counts()
    for s in strict["B"].elements:
        assert all(cal_counts.get(e, 0) >= 40 for e in s.split())
    assert len(strict["B"]) <= len(loose["B"])
    assert len(D.build_splits(synth, min_comps_per_element=10**6, **kw)["B"]) == 0


def test_propose_splits_cli_end_to_end(tmp_path, synth):
    import json
    import subprocess
    import sys

    meta = tmp_path / "meta.pkl"
    synth.to_pickle(meta)
    out = tmp_path / "splits"
    cmd = [sys.executable, "scripts/propose_splits.py", "--heldout-elements", "f-block", "Th",
           "--max-atoms", "24", "--seed", "5", "--n-cal", "200", "--n-a", "40", "--n-b", "40", "--n-c", "40",
           "--b-holdout-frac", "0.3", "--metadata", str(meta), "--out-dir", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    summary = json.loads((out / "split_summary.json").read_text())
    assert summary["config"]["b_holdout_frac"] == 0.3 and summary["config"]["max_atoms"] == 24
    for name in "ABC":
        se = summary[name]["coverage_se"]
        assert se["composition_level"]["n"] == summary[name]["compositions"]
        assert se["structure_level_secondary"]["se_if_all_used_assuming_independence"] <= se["composition_level"]["se"] + 1e-9
    sets = {n: pd.read_csv(out / f"{n}.csv") for n in ("cal", "A", "B", "C")}
    assert {"sid", "parent_id", "comp_key"} <= set(sets["A"].columns)  # source ids preserved
    sets["A0"] = pd.read_csv(out / "A0.csv")
    assert {"max_jaccard_to_cal"} <= set(sets["cal"].columns) and sets["B"].max_jaccard_to_cal.between(0, 1).all()
    for x in sets:
        for y in sets:
            if x < y:
                for col in ("comp_key", "parent_id", "sid"):
                    assert not set(sets[x][col]) & set(sets[y][col])
    r2 = subprocess.run([sys.executable, "scripts/propose_splits.py"], capture_output=True, text=True)
    assert r2.returncode == 2  # held-out elements and atom cap are required, no defaults


# ------------------------------------------------------------- A0 and similarity ---

ALL_FIVE = ["cal", "A0", "A", "B", "C"]


def test_five_splits_pairwise_disjoint_on_comp_parent_sid(synth):
    s = D.build_splits(synth, seed=11, n_cal=300, n_a=60, n_a0=60, n_b=60, n_c=60)
    assert set(s) == set(ALL_FIVE) and all(len(s[k]) > 0 for k in ALL_FIVE)
    for i, x in enumerate(ALL_FIVE):
        for y in ALL_FIVE[i + 1:]:
            for col in ("comp_key", "parent_id", "sid"):
                assert not (set(s[x][col]) & set(s[y][col])), (x, y, col)


def test_a0_is_drawn_jointly_with_cal_not_from_leftovers(synth):
    """Cal+A0 is one prefix of a single random permutation, so growing Cal by A0's size must reproduce Cal u A0."""
    kw = dict(seed=4, n_a=10, n_b=10, n_c=10)
    split = D.build_splits(synth, n_cal=150, n_a0=100, **kw)
    merged = D.build_splits(synth, n_cal=250, n_a0=0, **kw)
    assert set(split["cal"].sid) | set(split["A0"].sid) == set(merged["cal"].sid)
    assert len(split["A0"]) == 100 and len(merged["A0"]) == 0


def test_a0_properties_familiar_pool(synth):
    s = D.build_splits(synth, seed=2, n_cal=300, n_a=40, n_a0=80, n_b=40, n_c=40)
    assert not ({e for x in s["A0"].elements for e in x.split()} & D.F_BLOCK)
    assert (s["A0"].natoms <= 24).all() and (~s["A0"].hubbard_u).all() and (s["A0"].n_elements >= 2).all()
    assert s["A0"].comp_key.is_unique


def test_invariants_catch_parent_and_sid_leaks_into_a0(synth):
    s = D.build_splits(synth, seed=2, n_cal=200, n_a=20, n_a0=40, n_b=20, n_c=20)
    bad_parent = dict(s)
    bad_parent["A0"] = s["A0"].assign(parent_id=s["cal"].parent_id.iloc[0])
    with pytest.raises(AssertionError):
        D.check_split_invariants(bad_parent)
    bad_sid = dict(s)
    bad_sid["A0"] = s["A0"].assign(sid=s["cal"].sid.iloc[0])
    with pytest.raises(AssertionError):
        D.check_split_invariants(bad_sid)


def test_max_jaccard_matches_bruteforce_and_exclude_self():
    cal = pd.DataFrame({"elements": ["Al Cu", "Cu Zn", "Al Cu Zn"], "comp_key": ["a", "b", "c"]})
    test = pd.DataFrame({"elements": ["Al Cu", "Fe Ni", "Cu"], "comp_key": ["x", "y", "z"]})
    got = D.max_jaccard_to_cal(test, cal)
    assert got == pytest.approx([1.0, 0.0, 0.5])  # {Cu} vs {Cu,Zn}=1/2, vs {Al,Cu}=1/2
    loo = D.max_jaccard_to_cal(cal, cal, exclude_self=True)
    assert loo == pytest.approx([2 / 3, 2 / 3, 2 / 3])  # never 1.0 from comparing a composition with itself
    assert D.max_jaccard_to_cal(cal, cal)[0] == 1.0


def _cli(tmp_path, synth, extra):
    import subprocess
    import sys

    meta = tmp_path / "meta.pkl"
    synth.to_pickle(meta)
    out = tmp_path / "splits"
    cmd = [sys.executable, "scripts/propose_splits.py", "--heldout-elements", "f-block", "--max-atoms", "24", "--seed", "5",
           "--n-cal", "200", "--n-a0", "50", "--n-b", "40", "--n-c", "40", "--metadata", str(meta), "--out-dir", str(out), *extra]
    return subprocess.run(cmd, capture_output=True, text=True), out


def test_cli_n_a_all_takes_every_available_a(tmp_path, synth):
    import json

    r, out = _cli(tmp_path, synth, ["--n-a", "all"])
    assert r.returncode == 0, r.stderr[-600:]
    summary = json.loads((out / "split_summary.json").read_text())
    assert summary["targets"]["A"] == "all available"
    assert summary["A"]["compositions"] == summary["capacity_with_same_cal"]["A"] > 0


def test_cli_stops_without_writing_when_a_target_is_unreachable(tmp_path, synth):
    r, out = _cli(tmp_path, synth, ["--n-a", "all", "--n-b", "100000"])
    assert r.returncode != 0 and "STOP" in r.stderr and "B" in r.stderr
    assert not out.exists() or not any(out.iterdir())  # nothing written, no size lowered silently
