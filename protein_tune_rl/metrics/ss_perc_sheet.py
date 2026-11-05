from Bio.SeqUtils.ProtParam import ProteinAnalysis


class PercBetaSheet:
    """Beta-sheet propensity reward.

    Note: this is Biopython's compositional secondary-structure *fraction*
    (a fixed per-residue amino-acid lookup), not a structural prediction.

    ``region_only=True`` scores only the infilled span of the heavy chain
    (requires ``seq_pre_mask``/``seq_post_mask`` in the chains dict). This
    removes the dilution from the frozen framework + light chain, so the reward
    responds to what the policy actually controls. The normalization constants
    (mean/std) must be recomputed for the region-only distribution.
    """

    def __init__(self, mean=0.0, std=1.0, region_only=False):
        self.mean = mean
        self.std = std
        self.region_only = region_only

    def _target_sequence(self, chains):
        if (
            self.region_only
            and chains.get("seq_pre_mask") is not None
            and chains.get("seq_post_mask") is not None
        ):
            heavy = str(chains["H"])
            start = len(str(chains["seq_pre_mask"]))
            end = len(heavy) - len(str(chains["seq_post_mask"]))
            return heavy[start:end]
        return str(chains["H"]) + str(chains["L"])

    def __call__(self, chains):
        seq = self._target_sequence(chains)
        if len(seq) == 0:
            return (0.0 - self.mean) / self.std
        X = ProteinAnalysis(seq)
        return (X.secondary_structure_fraction()[2] - self.mean) / self.std
