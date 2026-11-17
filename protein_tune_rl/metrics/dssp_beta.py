import os
from typing import Dict

import numpy as np
from Bio.PDB import PDBParser

from protein_tune_rl.metrics.structure import StructureBasedMetric

BACKBONE = ("N", "CA", "C", "O")


class DSSPBetaStrand(StructureBasedMetric):
    """Predicted beta-strand fraction, assigned from coordinates.

    Folds the antibody with IgFold and runs the DSSP hydrogen-bond algorithm
    (via pydssp, which needs no system mkdssp) on the predicted backbone.
    Unlike ``ss_perc_sheet`` -- a fixed per-residue amino-acid lookup -- this
    reads secondary structure off the geometry, so it cannot be satisfied by
    amino-acid composition alone.

    ``region_only=True`` scores just the infilled span, which is what the
    policy controls; it requires ``seq_pre_mask``/``seq_post_mask`` in the
    chains dict. Whole-chain scoring is dominated by the frozen framework and
    responds only weakly to the generated loop.

    Normalisation constants must match the chosen scope. Measured over 20
    native held-out antibodies (see results/c4a_structural_attackability.csv):
    region mean 0.419, std 0.108.
    """

    def __init__(
        self,
        folding_tool: str = "igfold",
        options: Dict = None,
        mean: float = 0.0,
        std: float = 1.0,
        fail_value: float = 0.0,
        region_only: bool = True,
        chain: str = "H",
        mode: str = "fraction",
    ):
        if options is None:
            options = {"do_refine": False, "use_openmm": False, "do_renum": False}
        if mode not in {"fraction", "count"}:
            raise ValueError("mode must be 'fraction' or 'count'")
        super().__init__(folding_tool, options, mean, std, fail_value)
        self.region_only = region_only
        self.chain = chain
        self.mode = mode

    def _assign(self, pdb_path, start, length):
        import pydssp

        model = next(
            PDBParser(QUIET=True).get_structure("s", pdb_path).get_models()
        )
        residues = [r for r in model[self.chain] if all(a in r for a in BACKBONE)]
        if not residues:
            return None
        coords = np.array(
            [[r[a].get_coord() for a in BACKBONE] for r in residues],
            dtype=np.float32,
        )
        ss = pydssp.assign(coords, out_type="c3")

        if not self.region_only:
            scope = ss
        else:
            scope = ss[start : start + length]
            if len(scope) == 0:
                return None

        if self.mode == "count":
            # Count, not fraction. A fraction is a ratio over the generated
            # span, so truncating the span to a few framework-adjacent
            # residues raises it (measured: the policy collapsed HCDR3 to a
            # 3-5 residue stub, see Path_Forward.md section 6). A count cannot
            # be gamed that way: a 5-residue stub caps at 5, below the ~5.3
            # strand residues a native-length loop already provides.
            return float(np.sum(scope == "E"))
        return float(np.mean(scope == "E"))

    def __call__(self, chains: Dict):
        name = f"fold{str(self.count)}"
        output_pdb_file, out = self._fold(chains, name)

        score = None
        if out is not None and os.path.exists(output_pdb_file):
            pre = str(chains.get("seq_pre_mask") or "")
            post = str(chains.get("seq_post_mask") or "")
            heavy = str(chains["H"])
            length = len(heavy) - len(pre) - len(post)
            try:
                score = self._assign(output_pdb_file, len(pre), length)
            except Exception as e:
                print(f"Error in DSSPBetaStrand assignment: {e}")
                score = None

        self._safe_remove(output_pdb_file)
        self._safe_remove(self.workspace + name + ".fasta")

        if score is None:
            # Folding or assignment failed; fall back rather than kill the run.
            return (self.fail_value - self.mean) / self.std
        return (score - self.mean) / self.std

    def __repr__(self):
        scope = "generated region" if self.region_only else "whole chain"
        return f"DSSP beta-strand fraction ({scope})"
