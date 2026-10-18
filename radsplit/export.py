"""Write partitions in the formats expected by common training frameworks."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def monai_datalist(
    data: pd.DataFrame, *, image: str, label: str | None = None, split: str = "split"
) -> dict[str, list[dict[str, object]]]:
    """Decathlon style datalist: one list of ``{"image": ..., "label": ...}`` per partition."""
    out: dict[str, list[dict[str, object]]] = {}
    for name, part in data.groupby(split, sort=False):
        columns = [image] + ([label] if label else [])
        renamed = part[columns].rename(columns={image: "image", **({label: "label"} if label else {})})
        out[str(name)] = renamed.to_dict(orient="records")
    return out


def nnunet_splits(data: pd.DataFrame, *, case: str, fold: str = "split") -> list[dict[str, list[str]]]:
    """nnU-Net ``splits_final.json`` content from cross validation folds.

    Each fold in turn is the validation set and all other folds form the
    training set.
    """
    cases = data[[case, fold]].drop_duplicates()
    names = sorted(cases[fold].astype(str).unique())
    splits = []
    for name in names:
        held = cases[fold].astype(str) == name
        splits.append(
            {
                "train": sorted(cases.loc[~held, case].astype(str)),
                "val": sorted(cases.loc[held, case].astype(str)),
            }
        )
    return splits


def write_json(content: object, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(content, indent=2), encoding="utf-8")
    return target
