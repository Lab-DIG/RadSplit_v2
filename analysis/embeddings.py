"""Image embeddings for the downstream experiment.

Frontal radiographs of the COVID-19 Image Data Collection are passed through
a DenseNet-121 pretrained on other public chest radiograph data sets
(TorchXRayVision, weights "densenet121-res224-all"); the 1024 dimensional
pooled feature vector of each image (center crop, 224 by 224 pixels) is saved. No fine tuning is performed.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cohorts
import numpy as np
import pandas as pd
import torch
import torchxrayvision as xrv
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"
RESULTS = ROOT / "results"
WEIGHTS = "densenet121-res224-all"


def main() -> None:
    cache = Path.home() / ".torchxrayvision" / "models_data"
    cache.mkdir(parents=True, exist_ok=True)
    local = EXTERNAL / "densenet121all.pt"
    name = ("nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-"
            "sc15-seed0-best.pt")
    if local.exists() and not (cache / name).exists():
        shutil.copy(local, cache / name)
    torch.set_num_threads(1)
    model = xrv.models.DenseNet(weights=WEIGHTS).eval()
    records = cohorts.load("covid").records
    frontal = records[records["view"].isin(["PA", "AP", "AP Supine"])
                      & records["folder"].eq("images")
                      & records["finding"].ne("unlabeled")].reset_index(drop=True)
    features, keep = [], []
    for index, row in frontal.iterrows():
        path = EXTERNAL / "covidchestxray" / "images" / row["image"]
        try:
            image = np.asarray(Image.open(path).convert("F"))
        except Exception as error:  # unreadable files are listed, not silently dropped
            print(f"skipped {row['image']}: {error}")
            continue
        height, width = image.shape
        side = min(height, width)
        top, left = (height - side) // 2, (width - side) // 2
        crop = Image.fromarray(image[top:top + side, left:left + side])
        small = np.asarray(crop.resize((224, 224), Image.BILINEAR))
        small = xrv.datasets.normalize(small, max(float(image.max()), 255.0))
        with torch.no_grad():
            vector = model.features2(torch.from_numpy(small)[None, None].float())[0].numpy()
        features.append(vector)
        keep.append(index)
        if len(keep) % 100 == 0:
            print(len(keep), flush=True)
    table = frontal.loc[keep].reset_index(drop=True)
    columns = [f"f{j}" for j in range(len(features[0]))]
    out = pd.concat([table, pd.DataFrame(np.vstack(features), columns=columns)], axis=1)
    out.to_parquet(RESULTS / "covidembeddings.parquet", index=False)
    print(out.shape, out["patient_id"].nunique(), out["covid19"].mean())


if __name__ == "__main__":
    main()
