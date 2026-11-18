from typing import Dict


class NativeIdentity:
    """Fraction of the generated span matching the native region, position-wise.

    A deliberate NEGATIVE control for the degeneracy detector (C8). Unlike the
    endpoints characterised in C4, this reward cannot be satisfied by any
    constant output: the target differs per context, so a policy that improves
    must track the prompt rather than emit a fixed motif. A converging run
    against it should therefore trip none of the detector's signatures, which
    is what gives the detector a false-positive rate.

    Requires the collator to pass `masked_seq` (the native region) through the
    chains dict alongside `seq_pre_mask`/`seq_post_mask`.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def __call__(self, chains: Dict):
        native = str(chains.get("masked_seq") or "")
        heavy = str(chains["H"])
        pre = str(chains.get("seq_pre_mask") or "")
        post = str(chains.get("seq_post_mask") or "")
        gen = heavy[len(pre): len(heavy) - len(post)] if post else heavy[len(pre):]

        if not native:
            return (0.0 - self.mean) / self.std

        # Position-wise matches, normalised by the native length. Generating a
        # longer span cannot inflate the score, and truncating cannot either.
        matches = sum(1 for a, b in zip(gen, native) if a == b)
        score = matches / len(native)
        return (score - self.mean) / self.std

    def __repr__(self):
        return "native HCDR3 identity (position-wise fraction)"
