"""Deterministic original fixtures for architecture checks, not benchmark claims."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DatasetSchema, DatasetTable
from .splits import SplitAssignments


def make_synthetic(task: str = "binary", *, seed: int = 42, train_rows: int = 60,
                   validation_rows: int = 12, test_rows: int = 12) -> tuple[DatasetTable, SplitAssignments]:
    """Generate independent latent numeric/text factors and a synthetic target.

    Partitions are declared before fitting. The test partition is reserved by
    the smoke runner. Tiny tests may transform it to verify inference only.
    """
    if task not in {"regression", "binary", "multiclass"}:
        raise ValueError("Unsupported synthetic task")
    if train_rows < 30 or validation_rows < 3 or test_rows < 3:
        raise ValueError("Synthetic fixture needs at least 30 train and 3 validation/test rows")
    rng = np.random.default_rng(seed)
    count = train_rows + validation_rows + test_rows
    words = ("river forest orange silver table window garden stone bridge paper apple cloud "
             "field train quiet bright clear warm gentle crisp fresh calm vivid lively "
             "blue green amber violet copper flower tree path story song light lake "
             "pencil glass water wind leaf branch grass ocean mountain valley stone "
             "simple useful smooth soft sturdy bold quick steady open round deep wide").split()
    first = rng.choice(words, size=(count, 9))
    second = rng.choice(words, size=(count, 7))
    numeric = rng.normal(size=(count, 2))
    text_latent = np.isin(first[:, 0], words[:len(words) // 2]).astype(float) * 2 - 1
    score = .8 * numeric[:, 0] - .3 * numeric[:, 1] + .6 * text_latent
    if task == "regression":
        target = score + .1 * rng.normal(size=count)
    elif task == "binary":
        target = np.where(score > 0, "positive", "negative")
    else:
        target = np.asarray(["low", "middle", "high"])[np.digitize(score, [-.5, .5])]
    ids = [f"synthetic-{i:04d}" for i in range(count)]
    category = rng.choice(["north", "south", "east"], size=count).astype(object)
    category[::17] = None
    category[train_rows] = "unseen-validation-category"
    values = numeric.copy()
    values[::19, 0] = np.nan
    frame = pd.DataFrame({"row_id": ids, "description": [" ".join(row) for row in first],
                          "summary": [" ".join(row) for row in second],
                          "value": values[:, 0], "context": values[:, 1], "region": category,
                          "target": target, "fold_marker": ["reserved-metadata"] * count})
    schema = DatasetSchema("refutab_synthetic_v1", task, "target", ("description", "summary"),
                           ("value", "context"), ("region",), "row_id", forbidden_columns=("fold_marker",))
    table = DatasetTable(frame, schema)
    splits = SplitAssignments(tuple(ids[:train_rows]), tuple(ids[train_rows:train_rows + validation_rows]),
                              tuple(ids[train_rows + validation_rows:]), seed=seed,
                              dataset_fingerprint=table.fingerprint(),
                              metadata={"protocol": "synthetic_fixed_partition_v1", "seed": seed})
    splits.validate(table)
    return table, splits
