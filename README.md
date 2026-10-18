# RadSplit

Patient-level data splits, covariate balance, acquisition domain hold-outs, and split audits for medical imaging AI.

Imaging datasets usually contain several records per patient (images, series, lesions, or examinations). RadSplit keeps every record of a patient in the same set, balances the characteristics you choose across sets, holds out a scanner, kernel, site, or visit when you want to test generalization, and audits splits made by any other tool. Every run writes the assignments, a balance report, and the settings and hash needed to reproduce it.

RadSplit is research software. It is not a medical device. Lower covariate imbalance does not by itself establish better diagnostic performance.

## Installation

Python 3.10 or newer is required.

```bash
git clone https://github.com/Lab-DIG/RadSplit-package.git
cd RadSplit-package
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -e '.[manifest]'
radsplit --help
```

`[manifest]` adds the DICOM, NIfTI, MetaImage, and NRRD readers. Use `[plot]` for balance plots or `[dev]` for tests.

## Quick start

You need a folder of images and a `labels.csv` with a `patient_id` column plus the characteristics to balance. If you already have a table with one row per record, skip step 1.

```bash
# 1. Scan: read image headers into one row per series and join the labels
radsplit scan /data/ct --labels labels.csv --output manifest.csv

# 2. Balance: patient-disjoint train/tune/test split with balanced characteristics
radsplit split manifest.csv --target label --balance site --balance sex --balance age --output mysplit

# 3. Hold out: use one scanner, site, or kernel as an unseen test set
radsplit holdout manifest.csv --by site --value "site C" --balance label --balance age --output holdout

# 4. Audit: check any table with a set column (from any tool) for shared patients and imbalance
radsplit audit splits.csv --patient patient_id --split split --variable label --variable site --strict
```

Each command prints a report and writes its outputs to the `--output` folder. To try it without your own data, generate a synthetic cohort and start at step 2:

```bash
python -c "from radsplit.synthetic import make_cohort; make_cohort(1000, seed=0).to_csv('cohort.csv', index=False)"
```

## Learn more

The [guide](GUIDE.md) covers:

- [the input table format](GUIDE.md#input-data)
- [each step in detail, with Python equivalents](GUIDE.md#run-it-on-your-own-dataset)
- [a worked example on a real dataset (CBIS-DDSM)](GUIDE.md#example-on-a-real-dataset-cbis-ddsm)
- [cross-validation, official test sets, leave-one-group-out, and nnU-Net/MONAI/PyTorch/scikit-learn exports](GUIDE.md#use-cases)
- [output files, reading the balance report, and every option](GUIDE.md#outputs-and-reproducibility)

## Testing

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

The tests use generated data and cover splitting, hold-outs, audits, exports, framework adapters, and header readers.

## Repository layout

| Path | Content |
|---|---|
| `radsplit/` | The package |
| `tests/` | Automated tests on generated data |
| `examples/` | End-to-end run on a real public dataset (see the [guide](GUIDE.md#example-on-a-real-dataset-cbis-ddsm)) |
| `GUIDE.md` | Full documentation |

## Citation and license

Software: Apache License 2.0 (see `LICENSE`). Dataset licenses and pretrained-weight terms are separate, and no dataset is redistributed here.

Until the manuscript is published, cite the repository and the exact commit you used:

```text
Panjwani D, Pham HT, Pudhota A, Aguilera SM, Kanagala A, Powell K, Murad D, Bodduluri S. RadSplit: an
open-source tool for patient-level splits and acquisition domain hold-outs for radiology AI.
https://github.com/Lab-DIG/RadSplit-package
```

The manuscript used public, deidentified datasets. Users remain responsible for institutional review requirements and for the conditions of each dataset.
