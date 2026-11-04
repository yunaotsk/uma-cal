"""Resumable on-disk cache of UMA energy predictions, keyed by structure id (``sid``).

SQLite (stdlib), one commit per prediction, so an interrupted run resumes where it stopped.
The inference configuration (checkpoint, task, inference setting, device) is part of the key, so
predictions made under different settings are never mixed; library versions are stored alongside
every row and can be audited with ``distinct_versions``. The cache file is git-ignored (``data/cache/``).
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_CACHE_PATH = Path("data/cache/predictions.sqlite")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    sid TEXT NOT NULL,
    config_id TEXT NOT NULL,
    checkpoint TEXT NOT NULL,
    task TEXT NOT NULL,
    inference_settings TEXT NOT NULL,
    device TEXT NOT NULL,
    energy_total_eV REAL NOT NULL,
    natoms INTEGER NOT NULL,
    seconds REAL,
    torch_version TEXT,
    fairchem_core_version TEXT,
    ase_version TEXT,
    created_utc TEXT NOT NULL,
    PRIMARY KEY (sid, config_id)
)
"""


@dataclass(frozen=True)
class CacheConfig:
    checkpoint: str
    task: str
    inference_settings: str
    device: str

    @property
    def config_id(self) -> str:
        raw = "|".join((self.checkpoint, self.task, self.inference_settings, self.device))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @classmethod
    def from_predictor(cls, predictor) -> "CacheConfig":
        return cls(predictor.checkpoint, predictor.task, predictor.inference_settings, predictor.device)


def library_versions() -> dict[str, str]:
    import ase
    import fairchem.core
    import torch

    return {"torch_version": torch.__version__, "fairchem_core_version": fairchem.core.__version__, "ase_version": ase.__version__}


class PredictionCache:
    def __init__(self, path: str | Path = DEFAULT_CACHE_PATH, versions: dict[str, str] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._versions = versions  # injectable for tests; looked up lazily otherwise
        self._db = sqlite3.connect(self.path)
        self._db.execute(_SCHEMA)
        self._db.commit()

    def versions(self) -> dict[str, str]:
        if self._versions is None:
            self._versions = library_versions()
        return self._versions

    def get(self, sid: str, cfg: CacheConfig) -> float | None:
        row = self._db.execute(
            "SELECT energy_total_eV FROM predictions WHERE sid = ? AND config_id = ?", (sid, cfg.config_id)
        ).fetchone()
        return None if row is None else float(row[0])

    def put(self, sid: str, cfg: CacheConfig, energy_total_eV: float, natoms: int, seconds: float | None = None) -> None:
        v = self.versions()
        self._db.execute(
            "INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (sid, cfg.config_id, cfg.checkpoint, cfg.task, cfg.inference_settings, cfg.device, float(energy_total_eV), int(natoms),
             seconds, v.get("torch_version"), v.get("fairchem_core_version"), v.get("ase_version"),
             datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        self._db.commit()

    def __len__(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    def rows(self, cfg: CacheConfig | None = None) -> list[dict]:
        q, args = "SELECT * FROM predictions", ()
        if cfg is not None:
            q, args = q + " WHERE config_id = ?", (cfg.config_id,)
        cur = self._db.execute(q + " ORDER BY sid", args)
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def distinct_versions(self) -> list[tuple]:
        """Distinct (config_id, torch, fairchem, ase) tuples: more than one per config means mixed library versions."""
        return self._db.execute(
            "SELECT DISTINCT config_id, torch_version, fairchem_core_version, ase_version FROM predictions"
        ).fetchall()

    def close(self) -> None:
        self._db.close()


GATE_PATH = Path("data/cache/a0_gate.json")
SPLITS_DIR = Path("data/splits")
GATE_BAND = 0.02  # A0 coverage must be within about 2 points of nominal (reports/analysis_plan.md)


class SplitGuard:
    """Decides which structures may be predicted.

    Cal and A0 are always allowed. A, B and C stay **locked until the A0 gate is recorded as passed**
    (``record_a0_gate``). The gate decision is made on Cal and A0 predictions only, before any A, B or C
    prediction exists, which is what keeps the analysis pre-specified. This is a procedural guard against
    accidents, not a security boundary: the gate file can be edited by hand, and doing so would be a protocol violation.
    """

    def __init__(self, splits_dir: str | Path = SPLITS_DIR, gate_path: str | Path = GATE_PATH):
        self.splits_dir, self.gate_path = Path(splits_dir), Path(gate_path)

    def _sids(self, name: str) -> set[str]:
        import csv

        with open(self.splits_dir / f"{name}.csv", newline="") as f:
            return {row["sid"] for row in csv.DictReader(f)}

    def gate_passed(self) -> bool:
        import json

        if not self.gate_path.exists():
            return False
        return json.loads(self.gate_path.read_text()).get("passed") is True

    def allowed_sids(self) -> set[str]:
        allowed = self._sids("cal") | self._sids("A0")
        if self.gate_passed():
            allowed |= self._sids("A") | self._sids("B") | self._sids("C")
        return allowed


def a0_gate_decision(coverage: float, nominal: float = 0.90, band: float = GATE_BAND) -> bool:
    """True iff A0 coverage is within ``band`` of nominal (inclusive, rounded to avoid float edge noise)."""
    return round(abs(coverage - nominal), 9) <= band


def record_a0_gate(coverage: float, n_a0: int, n_cal: int, gate_path: str | Path = GATE_PATH,
                   nominal: float = 0.90, band: float = GATE_BAND) -> bool:
    """Record the A0 gate decision (to be called once, with Cal-calibrated coverage on A0 only). Returns ``passed``."""
    import json

    passed = a0_gate_decision(coverage, nominal, band)
    gate_path = Path(gate_path)
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps({
        "passed": passed, "a0_coverage": coverage, "nominal": nominal, "band": band, "n_a0": n_a0, "n_cal": n_cal,
        "decided_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rule": "pass iff |coverage - nominal| <= band (reports/analysis_plan.md); decided before any A, B or C prediction",
    }, indent=2))
    return passed


def predict_cached(predictor, items: Iterable[tuple[str, object]], cache: PredictionCache,
                   allowed_sids: Iterable[str] | None = None) -> dict[str, float]:
    """Predict total energies (eV) for ``(sid, Atoms)`` pairs, skipping cache hits.

    Hard guard: any sid outside the allowed set raises ``PermissionError`` *before* any prediction is made.
    ``allowed_sids=None`` means the default ``SplitGuard()``: Cal and A0 only, with A, B and C locked until the
    A0 gate has been recorded as passed. Pass an explicit set to narrow it further (e.g. Cal only).
    Returns ``{sid: energy_total_eV}`` for every item (hits and new predictions).
    """
    items = list(items)
    allowed = SplitGuard().allowed_sids() if allowed_sids is None else set(allowed_sids)
    bad = [sid for sid, _ in items if sid not in allowed]
    if bad:
        raise PermissionError(f"{len(bad)} structure(s) not allowed, e.g. {bad[:3]}; refusing to predict "
                              "(A, B and C stay locked until the A0 gate is recorded as passed)")
    cfg = CacheConfig.from_predictor(predictor)
    out: dict[str, float] = {}
    for sid, atoms in items:
        hit = cache.get(sid, cfg)
        if hit is None:
            t0 = time.perf_counter()
            e = predictor.predict_energy(atoms)
            cache.put(sid, cfg, e, len(atoms), time.perf_counter() - t0)
            hit = e
        out[sid] = hit
    return out
