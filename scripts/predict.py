"""Load a teacher/head-free bundle and write predictions for new CSV rows."""

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refutab.pipeline import ReFuTabPipeline


def predict_csv(model: ReFuTabPipeline, input_path: Path, output_path: Path, *, id_column: str | None = None):
    id_column = id_column or model.schema.id_column
    dtype = {id_column: str} if id_column and isinstance(model.training["train_ids"][0], str) else None
    frame = pd.read_csv(input_path, dtype=dtype)
    expected = set(model.schema.feature_columns) | ({id_column} if id_column else set())
    if set(frame.columns) != expected:
        raise ValueError("Input CSV must contain exactly the saved features and the declared row ID column")
    if id_column:
        frame.index = frame.pop(id_column)
    values = model.predict(frame.loc[:, list(model.schema.feature_columns)])
    names = ["prediction"] if model.schema.task == "regression" else [f"probability_{i}" for i in range(len(model.class_order))]
    result = pd.DataFrame(values, index=frame.index, columns=names)
    result.index.name = id_column or "row_id"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, mode="x")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New private output CSV; never overwritten")
    parser.add_argument("--model-path", type=Path, default=ROOT / "artifacts/e5-small-v2")
    parser.add_argument("--id-column")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    model = ReFuTabPipeline.load(args.bundle, model_path=args.model_path)
    predict_csv(model, args.input, args.output, id_column=args.id_column)
    print(f"Saved predictions to {args.output}; class order: {list(model.class_order)}")


if __name__ == "__main__":
    main()
