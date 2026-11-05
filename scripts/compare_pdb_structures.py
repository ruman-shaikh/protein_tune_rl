import argparse
import json

from protein_tune_rl.protein_evaluator.pdb_structure_evaluator import (
    PDBStructureEvaluator,
)


def parse_region(value):
    try:
        start, end = (int(part) for part in value.split(":"))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("region must be START:END") from exc
    return start, end


def main():
    parser = argparse.ArgumentParser(
        description="Compare a predicted antibody PDB with a reference PDB."
    )
    parser.add_argument("reference_pdb")
    parser.add_argument("predicted_pdb")
    parser.add_argument("--reference-region", required=True, type=parse_region)
    parser.add_argument("--predicted-region", type=parse_region)
    parser.add_argument("--reference-chain", default="H")
    parser.add_argument("--predicted-chain", default="H")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    metrics = PDBStructureEvaluator().evaluate(
        args.reference_pdb,
        args.predicted_pdb,
        reference_region=args.reference_region,
        predicted_region=args.predicted_region,
        reference_chain=args.reference_chain,
        predicted_chain=args.predicted_chain,
    )
    payload = json.dumps(metrics, indent=2, allow_nan=True)
    print(payload)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(payload + "\n")


if __name__ == "__main__":
    main()
