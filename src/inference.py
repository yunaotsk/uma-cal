"""Reusable UMA energy prediction interface (CPU, ``omat`` task).

The checkpoint is loaded once per ``UMAPredictor`` instance; subsequent calls
reuse the loaded model.
"""

from __future__ import annotations

from collections.abc import Sequence

from ase import Atoms

DEFAULT_CHECKPOINT = "uma-s-1p2p1"
DEFAULT_TASK = "omat"
DEFAULT_DEVICE = "cpu"
# "batch": full FP32, MOLE experts unmerged (safe for changing compositions).
# The library's "default" preset selects reduced-precision "nitro" mode.
DEFAULT_INFERENCE_SETTINGS = "batch"


class UMAPredictor:
    """Wraps FAIRChemCalculator to predict total energies for ASE structures."""

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        task: str = DEFAULT_TASK,
        device: str = DEFAULT_DEVICE,
        inference_settings: str = DEFAULT_INFERENCE_SETTINGS,
        calculator=None,
    ):
        self.checkpoint = checkpoint
        self.task = task
        self.device = device
        self.inference_settings = inference_settings
        if calculator is not None:  # explicit injection (used by unit tests)
            self._calc = calculator
        else:
            self._calc = self._load_calculator()

    def _load_calculator(self):
        # Imported lazily so importing this module does not pull in fairchem.
        from fairchem.core import FAIRChemCalculator, pretrained_mlip

        predict_unit = pretrained_mlip.get_predict_unit(
            self.checkpoint,
            inference_settings=self.inference_settings,
            device=self.device,
        )
        return FAIRChemCalculator(predict_unit, task_name=self.task)

    def predict_energy(self, atoms: Atoms) -> float:
        """Total energy in eV. Does not mutate the input structure."""
        if len(atoms) == 0:
            raise ValueError("Cannot predict energy for an empty structure.")
        work = atoms.copy()
        work.calc = self._calc
        return float(work.get_potential_energy())

    def predict_energies(self, structures: Sequence[Atoms]) -> list[float]:
        return [self.predict_energy(a) for a in structures]
