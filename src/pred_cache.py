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


def predict_cached(predictor, items: Iterable[tuple[str, object]], cache: PredictionCache,
                   allowed_sids: set[str] | None = None) -> dict[str, float]:
    """Predict total energies (eV) for ``(sid, Atoms)`` pairs, skipping cache hits.

    ``allowed_sids`` is a hard guard: if given, any sid outside it raises *before* any prediction is made.
    Returns ``{sid: energy_total_eV}`` for every item (hits and new predictions).
    """
    items = list(items)
    if allowed_sids is not None:
        bad = [sid for sid, _ in items if sid not in allowed_sids]
        if bad:
            raise PermissionError(f"{len(bad)} structure(s) not in the allowed set, e.g. {bad[:3]}; refusing to predict")
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
