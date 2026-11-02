"""Offline unit tests: a stub calculator is injected explicitly; no weights loaded."""

import pytest
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes

from src.inference import UMAPredictor


class _StubCalculator(Calculator):
    """Deterministic test double (energy = 2 eV per atom). Not a UMA prediction."""

    implemented_properties = ["energy"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 2.0 * len(atoms)}


@pytest.fixture
def predictor():
    return UMAPredictor(calculator=_StubCalculator())


def test_predict_energy_returns_float(predictor):
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    e = predictor.predict_energy(atoms)
    assert isinstance(e, float)
    assert e == pytest.approx(8.0)


def test_input_not_mutated(predictor):
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)
    predictor.predict_energy(atoms)
    assert atoms.calc is None


def test_predict_energies_batch(predictor):
    structs = [bulk("Al", "fcc", a=4.05), bulk("Cu", "fcc", a=3.6, cubic=True)]
    assert predictor.predict_energies(structs) == pytest.approx([2.0, 8.0])


def test_empty_structure_rejected(predictor):
    from ase import Atoms

    with pytest.raises(ValueError):
        predictor.predict_energy(Atoms())


def test_defaults_are_pinned():
    p = UMAPredictor(calculator=_StubCalculator())
    assert (p.checkpoint, p.task, p.device) == ("uma-s-1p2p1", "omat", "cpu")


def test_import_does_not_load_fairchem():
    import subprocess, sys

    out = subprocess.run(
        [sys.executable, "-c", "import sys, src.inference; print('fairchem' in sys.modules)"],
        capture_output=True, text=True,
    )
    assert out.stdout.strip() == "False"
