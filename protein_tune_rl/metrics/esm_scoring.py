from typing import Dict, Optional

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


class ESMSequenceScoring:
    """Score protein sequences with ESM masked pseudo-log-likelihood.

    Each selected residue is masked in turn and scored under the frozen ESM
    model.  Higher scores indicate that ESM assigns greater probability to the
    observed residues.  ``region_only=True`` restricts scoring to the generated
    span described by ``seq_pre_mask`` and ``seq_post_mask``.
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        reduction: str = "mean",
        region_only: bool = False,
        mask_batch_size: int = 32,
        mean: float = 0.0,
        std: float = 1.0,
        device: Optional[str] = None,
        local_files_only: bool = False,
        model=None,
        tokenizer=None,
    ):
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        if mask_batch_size < 1:
            raise ValueError("mask_batch_size must be positive")
        if std == 0:
            raise ValueError("std must be non-zero")

        self.reduction = reduction
        self.region_only = region_only
        self.mask_batch_size = mask_batch_size
        self.mean = mean
        self.std = std
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.model = model or AutoModelForMaskedLM.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.model.to(self.device).eval()

        if self.tokenizer.mask_token_id is None:
            raise ValueError("ESM tokenizer does not define a mask token")

    @staticmethod
    def _region_bounds(chains: Dict, sequence_length: int):
        if "seq_pre_mask" not in chains or "seq_post_mask" not in chains:
            raise ValueError(
                "region_only scoring requires seq_pre_mask and seq_post_mask"
            )
        start = len(chains["seq_pre_mask"])
        end = sequence_length - len(chains["seq_post_mask"])
        if start < 0 or end <= start or end > sequence_length:
            raise ValueError("invalid generated-region boundaries")
        return start, end

    def _residue_token_positions(self, encoding):
        special = encoding["special_tokens_mask"][0].bool()
        attended = encoding["attention_mask"][0].bool()
        return torch.nonzero(attended & ~special, as_tuple=False).flatten()

    @torch.inference_mode()
    def score(self, sequence: str, region=None):
        encoding = self.tokenizer(
            sequence,
            return_tensors="pt",
            return_special_tokens_mask=True,
            add_special_tokens=True,
        )
        residue_positions = self._residue_token_positions(encoding)
        if len(residue_positions) != len(sequence):
            raise ValueError(
                "ESM tokenization did not produce one token per residue; "
                f"got {len(residue_positions)} tokens for {len(sequence)} residues"
            )

        selected = residue_positions
        if region is not None:
            start, end = region
            selected = residue_positions[start:end]
        if len(selected) == 0:
            raise ValueError("no residues selected for ESM scoring")

        input_ids = encoding["input_ids"][0]
        attention_mask = encoding["attention_mask"][0]
        observed_ids = input_ids[selected]
        token_logps = []

        for offset in range(0, len(selected), self.mask_batch_size):
            positions = selected[offset : offset + self.mask_batch_size]
            batch_size = len(positions)
            masked_ids = input_ids.repeat(batch_size, 1)
            masked_attention = attention_mask.repeat(batch_size, 1)
            rows = torch.arange(batch_size)
            masked_ids[rows, positions] = self.tokenizer.mask_token_id

            logits = self.model(
                input_ids=masked_ids.to(self.device),
                attention_mask=masked_attention.to(self.device),
            ).logits
            logps = torch.log_softmax(logits, dim=-1).cpu()
            targets = observed_ids[offset : offset + batch_size]
            token_logps.append(logps[rows, positions, targets])

        token_logps = torch.cat(token_logps)
        if self.reduction == "mean":
            return token_logps.mean().item()
        return token_logps.sum().item()

    def __call__(self, chains: Dict):
        sequence = str(chains["H"])
        region = None
        if self.region_only:
            region = self._region_bounds(chains, len(sequence))
        raw_score = self.score(sequence, region=region)
        return (raw_score - self.mean) / self.std

    def __repr__(self):
        scope = "generated region" if self.region_only else "heavy chain"
        return f"ESM masked pseudo-log-likelihood ({scope}, {self.reduction})"
