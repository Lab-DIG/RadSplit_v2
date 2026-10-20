"""Download the public metadata tables used in the benchmark.

Only tabular metadata are downloaded (no images, except for the optional
radiograph files of the COVID-19 Image Data Collection used in the downstream
experiment). Every source is pinned to a commit so that the analysis can be
repeated on identical inputs. Obtain CBIS-DDSM case description tables
separately using the instructions in README.md; no datasets are redistributed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"

TORCHXRAYVISION = "bd41a6ce24c4f69fd6e7fd6c8b263166a6bbd737"
COVID = "78543292f8b01d5e0ed1d0e15dce71949f0657bb"
IDC = "24.2.2"  # also the pinned version of the idc-index-data package
PYLIDC = "0.2.3"
CXR8 = "77ad689c9364bceb791c7d913fa429b9e3413030"

FILES = {
    "nihchestxray14.csv.gz": (
        f"https://raw.githubusercontent.com/mlmed/torchxrayvision/{TORCHXRAYVISION}"
        "/torchxrayvision/data/Data_Entry_2017_v2020.csv.gz"
    ),
    "padchest.csv.gz": (
        f"https://raw.githubusercontent.com/mlmed/torchxrayvision/{TORCHXRAYVISION}"
        "/torchxrayvision/data/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz"
    ),
    "nihtestlist.txt": (
        f"https://raw.githubusercontent.com/TRKuan/cxr8/{CXR8}/dataset/test_list.txt"
    ),
    "idcctindex.parquet": (
        f"https://github.com/ImagingDataCommons/idc-index-data/releases/download/{IDC}/ct_index.parquet"
    ),
    "covidmetadata.csv": (
        f"https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/{COVID}/metadata.csv"
    ),
}
WEIGHTS = (
    "https://github.com/mlmed/torchxrayvision/releases/download/v1/"
    "nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", action="store_true",
                        help="also fetch COVID-19 collection radiographs and model weights")
    args = parser.parse_args()
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    record = {}
    for name, url in FILES.items():
        target = EXTERNAL / name
        if not target.exists():
            print(f"Downloading {name}")
            urllib.request.urlretrieve(url, target)
        record[name] = {"url": url, "sha256": sha256(target), "bytes": target.stat().st_size}
    lidc = EXTERNAL / "pylidc.sqlite"
    if not lidc.exists():
        # nodule annotations of LIDC-IDRI as distributed in the pylidc package
        import tempfile
        import zipfile

        with tempfile.TemporaryDirectory() as folder:
            subprocess.run(["python", "-m", "pip", "download", f"pylidc=={PYLIDC}", "--no-deps",
                            "-q", "-d", folder], check=True)
            wheel = next(Path(folder).glob("*.whl"))
            with zipfile.ZipFile(wheel) as archive:
                lidc.write_bytes(archive.read("pylidc/pylidc.sqlite"))
    record["pylidc.sqlite"] = {"source": f"pylidc {PYLIDC} (PyPI)", "sha256": sha256(lidc),
                               "bytes": lidc.stat().st_size}
    if args.images:
        repo = EXTERNAL / "covidchestxray"
        if not repo.exists():
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                            "https://github.com/ieee8023/covid-chestxray-dataset.git", str(repo)],
                           check=True)
            subprocess.run(["git", "-C", str(repo), "sparse-checkout", "set", "images"], check=True)
            subprocess.run(["git", "-C", str(repo), "checkout", COVID], check=True)
        weights = EXTERNAL / "densenet121all.pt"
        if not weights.exists():
            print("Downloading pretrained chest radiograph weights")
            urllib.request.urlretrieve(WEIGHTS, weights)
        record["densenet121all.pt"] = {"url": WEIGHTS, "sha256": sha256(weights),
                                       "bytes": weights.stat().st_size}
    for path in sorted((ROOT / "data" / "cbisddsm").glob("*.csv")):
        record[f"cbisddsm/{path.name}"] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    (ROOT / "data" / "sources.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
