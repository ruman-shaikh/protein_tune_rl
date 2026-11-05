import os
from typing import Dict

from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

from protein_tune_rl.metrics.structure import StructureBasedMetric


class SASA(StructureBasedMetric):
    def __init__(
        self,
        folding_tool: str,
        options: Dict = None,
        mean: float = 0.0,
        std: float = 1.0,
    ):
        if options is None:
            options = {"do_renum": False}
        super().__init__(folding_tool, options, mean, std)

        self.parser = PDBParser(QUIET=1)
        self.sr = ShrakeRupley()

    def __call__(self, chains: Dict):
        name = f"fold{str(self.count)}"
        output_pdb_file, out = self._fold(chains, name)

        if out is None or not os.path.exists(output_pdb_file):
            # Folding failed after retries; return the fallback score.
            self._safe_remove(output_pdb_file)
            self._safe_remove(self.workspace + name + ".fasta")
            return (self.fail_value - self.mean) / self.std

        struct = self.parser.get_structure(name, output_pdb_file)
        self.sr.compute(struct, level="S")

        self._safe_remove(output_pdb_file)
        self._safe_remove(self.workspace + name + ".fasta")

        return (struct.sasa - self.mean) / self.std

    def __repr__(self):
        return "SASA"
