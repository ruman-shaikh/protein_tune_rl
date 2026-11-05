from types import SimpleNamespace

import numpy as np
import pytest
import torch

from protein_tune_rl.metrics.esm_scoring import ESMSequenceScoring
from protein_tune_rl.protein_evaluator.pdb_structure_evaluator import (
    PDBStructureEvaluator,
)


class FakeESMTokenizer:
    mask_token_id = 2

    def __init__(self):
        self.ids = {aa: i + 3 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

    def __call__(
        self,
        sequence,
        return_tensors=None,
        return_special_tokens_mask=False,
        add_special_tokens=True,
    ):
        residue_ids = [self.ids[aa] for aa in sequence]
        return {
            "input_ids": torch.tensor([[0, *residue_ids, 1]]),
            "attention_mask": torch.ones((1, len(sequence) + 2), dtype=torch.long),
            "special_tokens_mask": torch.tensor(
                [[1, *([0] * len(sequence)), 1]], dtype=torch.long
            ),
        }


class FakeESMModel:
    def __init__(self, target_ids, boosts, vocab_size=32):
        self.target_ids = torch.tensor([0, *target_ids, 1])
        self.boosts = torch.tensor([0.0, *boosts, 0.0])
        self.vocab_size = vocab_size
        self.seen_mask_positions = []

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, self.vocab_size, device=input_ids.device)
        mask_positions = torch.nonzero(input_ids == 2, as_tuple=False)
        for row, position in mask_positions.tolist():
            self.seen_mask_positions.append(position)
            target = int(self.target_ids[position])
            logits[row, position, target] = self.boosts[position]
        return SimpleNamespace(logits=logits)


def _expected_logp(boost, vocab_size=32):
    return boost - np.log(np.exp(boost) + vocab_size - 1)


def test_esm_scores_every_residue_and_batches_masks():
    tokenizer = FakeESMTokenizer()
    sequence = "ACDE"
    target_ids = [tokenizer.ids[aa] for aa in sequence]
    model = FakeESMModel(target_ids, boosts=[1.0, 2.0, 3.0, 4.0])
    metric = ESMSequenceScoring(
        model=model,
        tokenizer=tokenizer,
        mask_batch_size=2,
        reduction="mean",
        device="cpu",
    )

    score = metric({"H": sequence})
    expected = np.mean([_expected_logp(x) for x in [1.0, 2.0, 3.0, 4.0]])

    assert score == pytest.approx(expected)
    assert sorted(model.seen_mask_positions) == [1, 2, 3, 4]


def test_esm_region_only_scores_generated_span():
    tokenizer = FakeESMTokenizer()
    sequence = "ACDE"
    target_ids = [tokenizer.ids[aa] for aa in sequence]
    model = FakeESMModel(target_ids, boosts=[1.0, 2.0, 3.0, 4.0])
    metric = ESMSequenceScoring(
        model=model,
        tokenizer=tokenizer,
        region_only=True,
        reduction="sum",
        device="cpu",
    )

    score = metric(
        {"H": sequence, "seq_pre_mask": "A", "seq_post_mask": "E"}
    )
    expected = _expected_logp(2.0) + _expected_logp(3.0)

    assert score == pytest.approx(expected)
    assert sorted(model.seen_mask_positions) == [2, 3]


def test_esm_region_only_requires_boundaries():
    tokenizer = FakeESMTokenizer()
    model = FakeESMModel([tokenizer.ids["A"]], boosts=[1.0])
    metric = ESMSequenceScoring(
        model=model, tokenizer=tokenizer, region_only=True, device="cpu"
    )
    with pytest.raises(ValueError, match="requires seq_pre_mask"):
        metric({"H": "A"})


def _write_pdb(
    path,
    residue_origins,
    bfactor=80.0,
    rotation=None,
    translation=None,
    residue_displacements=None,
):
    serial = 1
    lines = []
    atom_offsets = {
        "N": np.array([0.0, 0.0, 0.0]),
        "CA": np.array([0.9, 0.5, 0.2]),
        "C": np.array([1.8, 0.1, 0.6]),
        "O": np.array([2.3, -0.2, 1.0]),
    }
    rotation = np.eye(3) if rotation is None else np.asarray(rotation)
    translation = np.zeros(3) if translation is None else np.asarray(translation)
    residue_displacements = residue_displacements or {}
    for residue_number, origin in enumerate(residue_origins, start=1):
        for atom_name, offset in atom_offsets.items():
            coordinate = (np.asarray(origin) + offset) @ rotation + translation
            coordinate += np.asarray(
                residue_displacements.get(residue_number, np.zeros(3))
            )
            x, y, z = coordinate
            element = atom_name[0]
            lines.append(
                f"ATOM  {serial:5d} {atom_name:^4s} ALA H{residue_number:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{bfactor:6.2f}"
                f"          {element:>2s}  "
            )
            serial += 1
    lines.extend(["TER", "END"])
    path.write_text("\n".join(lines) + "\n")


def test_pdb_evaluator_aligns_framework_and_measures_region(tmp_path):
    reference_origins = [
        [0.0, 0.0, 0.0],
        [3.0, 1.0, 0.5],
        [5.0, 2.0, 1.0],
        [7.0, 0.5, 1.5],
        [9.0, 1.5, 0.2],
    ]
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    reference = tmp_path / "reference.pdb"
    predicted = tmp_path / "predicted.pdb"
    _write_pdb(reference, reference_origins, bfactor=20.0)
    _write_pdb(
        predicted,
        reference_origins,
        bfactor=85.0,
        rotation=rotation,
        translation=np.array([4, 8, 2]),
        residue_displacements={3: np.array([0.0, 0.0, 2.0])},
    )

    result = PDBStructureEvaluator().evaluate(
        reference, predicted, reference_region=(3, 3)
    )

    assert result["framework_backbone_rmsd"] < 1e-3
    assert result["region_backbone_rmsd"] > 1.0
    assert result["region_length_difference"] == 0
    assert result["predicted_region_sasa"] > 0
    assert result["predicted_region_mean_b_factor"] == pytest.approx(85.0)


def test_pdb_evaluator_reports_length_mismatch_without_region_rmsd(tmp_path):
    reference_origins = [[i * 3.0, (i % 2), 0.2 * i] for i in range(5)]
    predicted_origins = [[i * 3.0, (i % 2), 0.2 * i] for i in range(6)]
    reference = tmp_path / "reference.pdb"
    predicted = tmp_path / "predicted.pdb"
    _write_pdb(reference, reference_origins)
    _write_pdb(predicted, predicted_origins)

    result = PDBStructureEvaluator().evaluate(
        reference,
        predicted,
        reference_region=(3, 3),
        predicted_region=(3, 4),
    )

    assert result["region_length_difference"] == 1
    assert np.isnan(result["region_backbone_rmsd"])
