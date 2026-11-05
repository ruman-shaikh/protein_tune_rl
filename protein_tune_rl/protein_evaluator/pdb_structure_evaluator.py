from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.SASA import ShrakeRupley


BACKBONE_ATOMS = ("N", "CA", "C")


class PDBStructureEvaluator:
    """Compare a predicted antibody PDB against a reference structure.

    Region bounds are one-based and inclusive sequence positions within the
    selected chain.  Framework backbone atoms are used for rigid alignment;
    the reported region RMSD therefore measures loop geometry after framework
    superposition.  Region RMSD is undefined when loop lengths differ.
    """

    def __init__(self, backbone_atoms: Iterable[str] = BACKBONE_ATOMS):
        self.backbone_atoms = tuple(backbone_atoms)
        self.parser = PDBParser(QUIET=True)
        self.sasa = ShrakeRupley()

    def _load(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        structure = self.parser.get_structure(path.stem, str(path))
        return structure, next(structure.get_models())

    @staticmethod
    def _residues(model, chain_id):
        if chain_id not in model:
            available = ", ".join(chain.id for chain in model)
            raise ValueError(
                f"chain {chain_id!r} not found; available chains: {available}"
            )
        residues = [r for r in model[chain_id] if is_aa(r, standard=False)]
        if not residues:
            raise ValueError(f"chain {chain_id!r} has no amino-acid residues")
        return residues

    @staticmethod
    def _validate_region(region: Tuple[int, int], length: int):
        start, end = region
        if start < 1 or end < start or end > length:
            raise ValueError(
                f"region {region} is outside chain sequence positions 1..{length}"
            )
        return start - 1, end

    def _paired_atoms(self, fixed_residues, moving_residues, indices):
        fixed_atoms, moving_atoms = [], []
        for fixed_idx, moving_idx in indices:
            fixed_residue = fixed_residues[fixed_idx]
            moving_residue = moving_residues[moving_idx]
            for atom_name in self.backbone_atoms:
                if atom_name in fixed_residue and atom_name in moving_residue:
                    fixed_atoms.append(fixed_residue[atom_name])
                    moving_atoms.append(moving_residue[atom_name])
        if len(fixed_atoms) < 3:
            raise ValueError("fewer than three paired backbone atoms were found")
        return fixed_atoms, moving_atoms

    @staticmethod
    def _fit(fixed_atoms, moving_atoms):
        fixed = np.asarray([a.coord for a in fixed_atoms], dtype=float)
        moving = np.asarray([a.coord for a in moving_atoms], dtype=float)
        fixed_center = fixed.mean(axis=0)
        moving_center = moving.mean(axis=0)
        covariance = (moving - moving_center).T @ (fixed - fixed_center)
        u, _, vt = np.linalg.svd(covariance)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        translation = fixed_center - moving_center @ rotation
        aligned = moving @ rotation + translation
        rmsd = np.sqrt(np.mean(np.sum((fixed - aligned) ** 2, axis=1)))
        return rotation, translation, float(rmsd)

    @staticmethod
    def _rmsd_after_transform(fixed_atoms, moving_atoms, rotation, translation):
        fixed = np.asarray([a.coord for a in fixed_atoms], dtype=float)
        moving = np.asarray([a.coord for a in moving_atoms], dtype=float)
        aligned = moving @ rotation + translation
        return float(np.sqrt(np.mean(np.sum((fixed - aligned) ** 2, axis=1))))

    def _sasa_metrics(self, structure, residues, region_slice):
        self.sasa.compute(structure, level="R")
        residue_sasa = np.asarray(
            [float(getattr(residue, "sasa", 0.0)) for residue in residues]
        )
        start, end = region_slice
        return float(residue_sasa.sum()), float(residue_sasa[start:end].sum())

    @staticmethod
    def _mean_b_factor(residues, region_slice):
        start, end = region_slice
        atoms = [atom for residue in residues[start:end] for atom in residue]
        if not atoms:
            return float("nan")
        return float(np.mean([atom.bfactor for atom in atoms]))

    def evaluate(
        self,
        reference_pdb,
        predicted_pdb,
        reference_region: Tuple[int, int],
        predicted_region: Optional[Tuple[int, int]] = None,
        reference_chain: str = "H",
        predicted_chain: str = "H",
    ) -> Dict[str, float]:
        predicted_region = predicted_region or reference_region
        reference_structure, reference_model = self._load(reference_pdb)
        predicted_structure, predicted_model = self._load(predicted_pdb)
        reference_residues = self._residues(reference_model, reference_chain)
        predicted_residues = self._residues(predicted_model, predicted_chain)
        ref_slice = self._validate_region(reference_region, len(reference_residues))
        pred_slice = self._validate_region(predicted_region, len(predicted_residues))

        ref_start, ref_end = ref_slice
        pred_start, pred_end = pred_slice
        prefix_count = min(ref_start, pred_start)
        suffix_count = min(
            len(reference_residues) - ref_end,
            len(predicted_residues) - pred_end,
        )
        framework_pairs = [(i, i) for i in range(prefix_count)]
        framework_pairs.extend(
            (
                len(reference_residues) - suffix_count + i,
                len(predicted_residues) - suffix_count + i,
            )
            for i in range(suffix_count)
        )
        fixed_framework, moving_framework = self._paired_atoms(
            reference_residues, predicted_residues, framework_pairs
        )
        rotation, translation, framework_rmsd = self._fit(
            fixed_framework, moving_framework
        )

        reference_region_length = ref_end - ref_start
        predicted_region_length = pred_end - pred_start
        region_rmsd = float("nan")
        if reference_region_length == predicted_region_length:
            region_pairs = [
                (ref_start + i, pred_start + i)
                for i in range(reference_region_length)
            ]
            fixed_region, moving_region = self._paired_atoms(
                reference_residues, predicted_residues, region_pairs
            )
            region_rmsd = self._rmsd_after_transform(
                fixed_region, moving_region, rotation, translation
            )

        reference_total_sasa, reference_region_sasa = self._sasa_metrics(
            reference_structure, reference_residues, ref_slice
        )
        predicted_total_sasa, predicted_region_sasa = self._sasa_metrics(
            predicted_structure, predicted_residues, pred_slice
        )

        return {
            "reference_chain_length": len(reference_residues),
            "predicted_chain_length": len(predicted_residues),
            "reference_region_length": reference_region_length,
            "predicted_region_length": predicted_region_length,
            "region_length_difference": (
                predicted_region_length - reference_region_length
            ),
            "framework_backbone_rmsd": framework_rmsd,
            "region_backbone_rmsd": region_rmsd,
            "reference_total_sasa": reference_total_sasa,
            "predicted_total_sasa": predicted_total_sasa,
            "reference_region_sasa": reference_region_sasa,
            "predicted_region_sasa": predicted_region_sasa,
            "predicted_region_mean_b_factor": self._mean_b_factor(
                predicted_residues, pred_slice
            ),
        }
