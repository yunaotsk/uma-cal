"""OMat24 dataset ingestion, composition utilities, and chemical classification.

No UMA inference happens here. Labels are the raw DFT total energies stored in the
ASE calculator results (eV, total per cell); the MP2020-corrected energy stored in
``atoms.info`` is kept as a separate column and is NOT the default label.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from ase.data import atomic_numbers, chemical_symbols

# Extraction directory of the archive; the .aselmdb files are discovered recursively beneath it.
DEFAULT_LMDB_DIR = Path("data/raw/omat24_val_rattled-1000-subsampled")

# --------------------------------------------------------------------------- #
# Element groups (explicit rules; atomic-number ranges)
# --------------------------------------------------------------------------- #
def _z(symbols: str) -> set[str]:
    return set(symbols.split())


TM_3D = {chemical_symbols[z] for z in range(21, 31)}  # Sc..Zn
TM_4D = {chemical_symbols[z] for z in range(39, 49)}  # Y..Cd
TM_5D = {chemical_symbols[z] for z in range(72, 81)}  # Hf..Hg
LANTHANIDES = {chemical_symbols[z] for z in range(57, 72)}  # La..Lu
ACTINIDES = {chemical_symbols[z] for z in range(89, 104)}  # Ac..Lr
# Elements that receive Hubbard U in OMat24 (oxides/fluorides only), per the OMat24 paper.
HUBBARD_ELEMENTS = _z("Co Cr Fe Mn Mo Ni V W")
HUBBARD_ANIONS = _z("O F")

STRONG_ANION_CLASS = {
    **{e: "halide" for e in _z("F Cl Br I")},
    "O": "oxide",
    **{e: "chalcogenide" for e in _z("S Se Te")},
    "N": "nitride",
}
WEAK_ANION_CLASS = [  # applied in this order only if no strong anion is present
    (_z("P As"), "pnictide"),
    (_z("H"), "hydride"),
    (_z("C"), "carbide"),
    (_z("B"), "boride"),
    (_z("Si"), "silicide"),
]


def reduced_composition(numbers) -> dict[str, int]:
    """Reduced composition (element -> integer count after dividing by the gcd)."""
    counts = Counter(chemical_symbols[int(z)] for z in numbers)
    if not counts:
        raise ValueError("empty structure")
    g = math.gcd(*counts.values())
    return {el: n // g for el, n in sorted(counts.items())}


def composition_key(reduced: dict[str, int]) -> str:
    """Canonical string, elements alphabetical, counts always explicit: 'Dy1N1Si1'."""
    return "".join(f"{el}{n}" for el, n in sorted(reduced.items()))


def parse_composition_key(key: str) -> dict[str, int]:
    import re

    return {el: int(n) for el, n in re.findall(r"([A-Z][a-z]?)(\d+)", key)}


def chemical_family(elements) -> str:
    """Anion-based family. Strong anions: halogens, O, chalcogens, N.

    - exactly one strong-anion class -> that class (H/B/C/Si/P/As act as cations/network formers)
    - two or more strong-anion classes -> 'mixed-anion' (kept separate, not forced)
    - no strong anion: P/As -> pnictide; H -> hydride; C -> carbide; B -> boride; Si -> silicide
      (if more than one of those weak classes matches -> 'mixed-light-element')
    - single element -> 'elemental'; otherwise 'intermetallic' (includes Ge, Sb, Bi, Sn, ...)
    """
    els = set(elements)
    if len(els) == 1:
        return "elemental"
    strong = {STRONG_ANION_CLASS[e] for e in els if e in STRONG_ANION_CLASS}
    if len(strong) == 1:
        return next(iter(strong))
    if len(strong) > 1:
        return "mixed-anion"
    weak = [name for group, name in WEAK_ANION_CLASS if els & group]
    if len(weak) == 1:
        return weak[0]
    if len(weak) > 1:
        return "mixed-light-element"
    return "intermetallic"


def tm_flags(elements) -> tuple[bool, bool]:
    els = set(elements)
    return bool(els & TM_3D), bool(els & TM_4D)


def tm_class(elements) -> str:
    has3, has4 = tm_flags(elements)
    return {(True, True): "3d+4d", (True, False): "3d-only", (False, True): "4d-only", (False, False): "neither"}[
        (has3, has4)
    ]


def hubbard_u_applies(elements) -> bool:
    """True if OMat24's stated PBE+U rule (oxide/fluoride containing Co,Cr,Fe,Mn,Mo,Ni,V,W) applies."""
    els = set(elements)
    return bool(els & HUBBARD_ELEMENTS) and bool(els & HUBBARD_ANIONS)


def order_name(n_elements: int) -> str:
    return {1: "unary", 2: "binary", 3: "ternary"}.get(n_elements, "quaternary+")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def find_aselmdb_files(path: str | Path) -> list[str]:
    """Sorted ``*.aselmdb`` files under ``path`` (a file, or a directory searched recursively).

    ``AseDBDataset`` only globs ``<dir>/*`` (non-recursive) and the OMat24 tarball extracts to
    ``<dir>/<name>/data.aselmdb``, so the database files must be located explicitly.
    """
    path = Path(path)
    if path.is_file():
        return [str(path)]
    files = sorted(str(f) for f in path.rglob("*.aselmdb"))
    if not files:
        raise FileNotFoundError(f"No .aselmdb files under {path}")
    return files


def open_dataset(path: str | Path = DEFAULT_LMDB_DIR):
    """Open the OMat24 LMDB(s) with fairchem's supported ``AseDBDataset`` reader."""
    from fairchem.core.datasets import AseDBDataset

    return AseDBDataset(config={"src": find_aselmdb_files(path)})


def iter_omat24_atoms(path: str | Path = DEFAULT_LMDB_DIR) -> Iterator[tuple[int, object]]:
    """Yield ``(index, Atoms)``; the index is stable for a fixed archive."""
    ds = open_dataset(path)
    for i in range(len(ds)):
        yield i, ds.get_atoms(i)


def load_metadata(lmdb_dir: str | Path = DEFAULT_LMDB_DIR) -> pd.DataFrame:
    """Read every structure into a metadata table (source ids, parent ids, labels, chemistry flags)."""
    return pd.DataFrame([atoms_to_row(atoms, index=i) for i, atoms in iter_omat24_atoms(lmdb_dir)])


def atoms_to_row(atoms, index: int) -> dict:
    info = atoms.info
    red = reduced_composition(atoms.numbers)
    els = sorted(red)
    has3, has4 = tm_flags(els)
    energy = float(atoms.get_potential_energy())
    n = len(atoms)
    return {
        "index": index,
        "sid": info["sid"],
        "parent_id": info.get("parent_id", ""),
        "calc_id": info.get("calc_id", ""),
        "task_type": info.get("task_type", ""),
        "prototype_label": info.get("prototype_label", ""),
        "parent_prototype_label": info.get("parent_prototype_label", ""),
        "composition_reduced_stored": info.get("composition_reduced", ""),
        "comp_key": composition_key(red),
        "elements": " ".join(els),
        "n_elements": len(els),
        "natoms": n,
        "energy_total_eV": energy,
        "energy_per_atom_eV": energy / n,
        "energy_corrected_mp2020_eV": float(info["energy_corrected_mp2020"])
        if "energy_corrected_mp2020" in info
        else float("nan"),
        "max_abs_force_eV_A": float(np.abs(atoms.get_forces()).max()),
        "volume_per_atom_A3": atoms.get_volume() / n,
        "family": chemical_family(els),
        "has_3d": has3,
        "has_4d": has4,
        "tm_class": tm_class(els),
        "hubbard_u": hubbard_u_applies(els),
    }


def structural_prototype(label: str) -> str:
    """Prototype label without the element decoration, e.g. 'ABC_tI12_139_e_e_e:Dy-N-Si' -> 'ABC_tI12_139_e_e_e'."""
    return label.split(":")[0]


def get_atoms_by_index(index: int, lmdb_dir: str | Path = DEFAULT_LMDB_DIR):
    return open_dataset(lmdb_dir).get_atoms(index)


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def assert_disjoint(df_a: pd.DataFrame, df_b: pd.DataFrame, col: str = "comp_key") -> None:
    overlap = set(df_a[col]) & set(df_b[col])
    if overlap:
        raise AssertionError(f"{len(overlap)} shared {col} values, e.g. {sorted(overlap)[:5]}")


def binomial_se(p: float, n: int) -> float:
    """Standard error of an empirical coverage estimate, treating test points as independent."""
    return math.sqrt(p * (1 - p) / n)


def coverage_se_levels(n_structures: int, n_compositions: int, p: float = 0.9) -> dict:
    """Binomial SE of a coverage estimate at the two levels.

    ``composition_level`` is the headline: structures sharing a reduced composition (rattled copies,
    polymorphs) are correlated, so the number of distinct compositions bounds the effective sample size.
    ``structure_level`` is secondary and optimistic whenever n_structures > n_compositions.
    """
    if n_structures <= 0 or n_compositions <= 0:
        raise ValueError("counts must be positive")
    return {
        "n_compositions": n_compositions,
        "se_composition_level": binomial_se(p, n_compositions),
        "n_structures": n_structures,
        "se_structure_level": binomial_se(p, n_structures),
    }


def sample_by_group(df: pd.DataFrame, group_col: str, max_per_group: int, seed: int) -> pd.DataFrame:
    """Cap rows per group (e.g. per reduced composition) with a seeded RNG; order-stable."""
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in df.sort_values("index").groupby(group_col, sort=True):
        if len(g) > max_per_group:
            g = g.iloc[np.sort(rng.choice(len(g), max_per_group, replace=False))]
        parts.append(g)
    return pd.concat(parts).sort_values("index").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Split construction (pure metadata operations; no model inference)
# --------------------------------------------------------------------------- #
F_BLOCK = LANTHANIDES | ACTINIDES
NOBLE_GASES = _z("He Ne Ar Kr Xe Rn")
DEFAULT_SEED = 20241220

ELEMENT_GROUPS = {
    "f-block": F_BLOCK,
    "lanthanides": LANTHANIDES,
    "actinides": ACTINIDES,
    "3d": TM_3D,
    "4d": TM_4D,
    "5d": TM_5D,
}


def resolve_element_spec(tokens) -> set[str]:
    """Turn CLI tokens (element symbols and/or group names such as 'f-block') into a symbol set."""
    out: set[str] = set()
    for t in tokens:
        if t.lower() in ELEMENT_GROUPS:
            out |= ELEMENT_GROUPS[t.lower()]
        elif t in atomic_numbers and t != "X":
            out.add(t)
        else:
            raise ValueError(f"Unknown element or group: {t!r}")
    if not out:
        raise ValueError("empty element specification")
    return out


def eligible_structures(df: pd.DataFrame, max_atoms: int = 24) -> pd.DataFrame:
    """Structure-level eligibility, label-independent.

    Drops: > max_atoms atoms (CPU budget); Hubbard-U-treated structures (different DFT
    treatment); unary structures; noble-gas-containing structures (<=3 compositions).
    Nothing here looks at energies or forces.
    """
    els = df["elements"].str.split()
    keep = (
        (df["natoms"] <= max_atoms)
        & (~df["hubbard_u"])
        & (df["n_elements"] >= 2)
        & (~els.map(lambda x: bool(set(x) & NOBLE_GASES)))
    )
    return df[keep].copy()


def one_per_composition(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """One randomly chosen structure per reduced composition (sibling/polymorph de-duplication)."""
    d = df.sort_values("index").reset_index(drop=True)
    order = rng.permutation(len(d))
    return d.iloc[order].drop_duplicates("comp_key").sort_values("index").reset_index(drop=True)


def _pairs(elements: str) -> set[tuple[str, str]]:
    return set(itertools.combinations(sorted(elements.split()), 2))


def _pick_one_per_system(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    d = df.sort_values("index").reset_index(drop=True)
    d = d.iloc[rng.permutation(len(d))].drop_duplicates("elements")
    return d.sort_values("index").reset_index(drop=True)


def _take(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(df) <= n:
        return df.sort_values("index").reset_index(drop=True)
    idx = np.sort(rng.choice(len(df), n, replace=False))
    return df.sort_values("index").reset_index(drop=True).iloc[idx].reset_index(drop=True)


def build_splits(
    df: pd.DataFrame,
    seed: int = DEFAULT_SEED,
    n_cal: int = 2000,
    n_a: int = 1000,
    n_a0: int = 1000,
    n_b: int = 1000,
    n_c: int = 1000,
    max_atoms: int = 24,
    b_holdout_frac: float = 0.25,
    held_out_elements: set[str] = F_BLOCK,
    min_comps_per_element: int = 5,
) -> dict[str, pd.DataFrame]:
    """Return {'cal','A0','A','B','C'} DataFrames (one structure per reduced composition).

    Familiar pool P = eligible compositions with no held-out element.
    * B-reserved systems: a random ``b_holdout_frac`` of the systems (exact element sets) of P is
      reserved for B and can never appear in Cal, A0 or A. The remaining compositions form Q.
    * **Cal and A0 are drawn jointly**: one random permutation of Q, first ``n_cal`` -> Cal, next
      ``n_a0`` -> A0, remainder -> leftover. Cal and A0 are therefore exchangeable draws from Q
      (A0 is the pipeline control: coverage should be ~nominal).
    * A (system-matched): leftover compositions whose chemical system is present in Cal; one per system.
    * B: compositions of B-reserved systems whose every element pair co-occurs in some Cal composition
      and whose every element appears in >= ``min_comps_per_element`` Cal compositions; one per system.
    * C: one composition per system among compositions with >= 1 held-out element.
    A's size is limited by how many leftover systems also occur in Cal (see ``capacity``).
    """
    rng = np.random.default_rng(seed)
    base = eligible_structures(df, max_atoms)
    u = one_per_composition(base, rng)
    held = u["elements"].map(lambda s: bool(set(s.split()) & held_out_elements))
    pool, c_pool = u[~held], u[held]

    systems = sorted(pool["elements"].unique())
    perm = rng.permutation(len(systems))
    n_hold = int(round(b_holdout_frac * len(systems)))
    b_systems = {systems[i] for i in perm[:n_hold]}
    q = pool[~pool["elements"].isin(b_systems)].sort_values("index").reset_index(drop=True)

    order = rng.permutation(len(q))
    cal = q.iloc[np.sort(order[:n_cal])].reset_index(drop=True)
    a0 = q.iloc[np.sort(order[n_cal : n_cal + n_a0])].reset_index(drop=True)
    leftover = q.iloc[order[n_cal + n_a0 :]]

    cal_sys_set = set(cal["elements"])
    a = _take(_pick_one_per_system(leftover[leftover["elements"].isin(cal_sys_set)], rng), n_a, rng)

    cal_pairs = set().union(*[_pairs(s) for s in cal_sys_set]) if cal_sys_set else set()
    cal_el_counts = Counter(e for s in cal["elements"] for e in s.split())
    b_cand = pool[pool["elements"].isin(b_systems)]
    b_cand = b_cand[
        b_cand["elements"].map(
            lambda s: _pairs(s) <= cal_pairs
            and s not in cal_sys_set
            and all(cal_el_counts[e] >= min_comps_per_element for e in s.split())
        )
    ]
    b = _take(_pick_one_per_system(b_cand, rng), n_b, rng)
    c = _take(_pick_one_per_system(c_pool, rng), n_c, rng)
    c = c.assign(n_held_out_elements=c["elements"].map(lambda s: len(set(s.split()) & held_out_elements)))

    out = {"cal": cal, "A0": a0, "A": a, "B": b, "C": c}
    check_split_invariants(out, held_out_elements)
    return out


def check_split_invariants(splits: dict[str, pd.DataFrame], held_out_elements: set[str] = F_BLOCK) -> None:
    """Raise AssertionError if any leakage invariant is violated.

    Pairwise disjointness on reduced composition, ``parent_id`` and ``sid`` across every pair of splits
    present, plus the shift properties of A, B, C and the held-out-element rule.
    """
    names = list(splits)
    for i, x in enumerate(names):
        for y in names[i + 1 :]:
            for col in ("comp_key", "sid", "parent_id"):
                assert_disjoint(splits[x], splits[y], col)
    cal = splits["cal"]
    cal_sys = set(cal["elements"])
    cal_els = {e for s in cal["elements"] for e in s.split()}
    assert not (cal_els & held_out_elements), "Cal contains held-out elements"
    if "A" in splits:
        assert set(splits["A"]["elements"]) <= cal_sys, "A systems must be present in Cal"
    if "B" in splits:
        assert not (set(splits["B"]["elements"]) & cal_sys), "B systems must be absent from Cal"
        b_els = {e for s in splits["B"]["elements"] for e in s.split()}
        assert b_els <= cal_els, "B contains an element unseen in Cal"
    for name in ("A0", "A", "B"):
        if name in splits:
            els = {e for s in splits[name]["elements"] for e in s.split()}
            assert not (els & held_out_elements), f"{name} contains held-out elements"
    if "C" in splits:
        assert splits["C"]["elements"].map(lambda s: bool(set(s.split()) & held_out_elements)).all(), "C lacks held-out element"
    for name, sp in splits.items():
        assert sp["comp_key"].is_unique, f"{name} has repeated compositions"
    for name in ("A", "B", "C"):
        if name in splits:
            assert splits[name]["elements"].is_unique, f"{name} has repeated systems"


def _element_matrix(element_strings, vocab: dict[str, int]) -> np.ndarray:
    m = np.zeros((len(element_strings), len(vocab)), dtype=np.float32)
    for i, s in enumerate(element_strings):
        for e in s.split():
            m[i, vocab[e]] = 1.0
    return m


def max_jaccard_to_cal(test: pd.DataFrame, cal: pd.DataFrame, exclude_self: bool = False) -> np.ndarray:
    """Per test structure: max Jaccard similarity between its element set and any Cal element set.

    ``exclude_self=True`` (for scoring Cal itself) ignores the structure's own composition, so a Cal
    structure is compared with the *other* Cal compositions (system-mates still count).
    """
    vocab = {e: i for i, e in enumerate(sorted({e for s in list(test["elements"]) + list(cal["elements"]) for e in s.split()}))}
    t, c = _element_matrix(test["elements"], vocab), _element_matrix(cal["elements"], vocab)
    inter = t @ c.T
    union = t.sum(1)[:, None] + c.sum(1)[None, :] - inter
    jac = inter / union
    if exclude_self:
        same = test["comp_key"].to_numpy()[:, None] == cal["comp_key"].to_numpy()[None, :]
        jac = np.where(same, -1.0, jac)
    return jac.max(axis=1)


# --------------------------------------------------------------------------- #
# Alternative family rules (sensitivity check only; NOT used for splitting)
# --------------------------------------------------------------------------- #
_ALT_ANION_GROUPS = {
    "hydride": _z("H"),
    "carbide": _z("C"),
    "nitride": _z("N"),
    "oxide": _z("O"),
    "halide": _z("F Cl Br I"),
    "chalcogenide": _z("S Se Te"),
    "pnictide": _z("P As"),
}
_ALT_METALLOIDS = _z("B Si Ge Sb")


def family_alternative(elements) -> str:
    """The rules from the ``claude/vibrant-rubin-1v8122`` branch: every H/C/N/O/halogen/chalcogen/P/As
    counts as an anion (so phosphates and carbonates become 'mixed-anion'); B/Si/Ge/Sb form one
    'metalloid-containing' bucket; noble-gas compounds are 'other'."""
    els = set(elements)
    if els & NOBLE_GASES:
        return "other"
    if len(els) == 1:
        return "elemental"
    present = [name for name, g in _ALT_ANION_GROUPS.items() if els & g]
    if len(present) == 1:
        return present[0]
    if len(present) > 1:
        return "mixed-anion"
    if els & _ALT_METALLOIDS:
        return "metalloid-containing"
    return "intermetallic"
