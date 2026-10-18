# RadSplit guide

Everything beyond the quick start in the [README](README.md): input data, the full four-step workflow, a worked example on a real dataset, every use case, outputs, and options.

## Contents

1. [Input data](#input-data)
2. [Run it on your own dataset](#run-it-on-your-own-dataset)
3. [Example on a real dataset: CBIS-DDSM](#example-on-a-real-dataset-cbis-ddsm)
4. [Use cases](#use-cases)
   - [A. Balanced training, tuning, and test sets](#a-balanced-training-tuning-and-test-sets)
   - [B. Custom set names and fractions](#b-custom-set-names-and-fractions)
   - [C. Cross-validation folds](#c-cross-validation-folds)
   - [D. Keep an official test set](#d-keep-an-official-test-set)
   - [E. Hold out a scanner, kernel, site, or visit](#e-hold-out-a-scanner-kernel-site-or-visit)
   - [F. Rotate over every group (leave one group out)](#f-rotate-over-every-group-leave-one-group-out)
   - [G. Audit an existing split](#g-audit-an-existing-split)
   - [H. Check splits automatically in a pipeline](#h-check-splits-automatically-in-a-pipeline)
   - [I. Start from a folder of images](#i-start-from-a-folder-of-images)
   - [J. Use the split in nnU-Net, MONAI, PyTorch, or scikit-learn](#j-use-the-split-in-nnu-net-monai-pytorch-or-scikit-learn)
5. [Outputs and reproducibility](#outputs-and-reproducibility)
6. [Reading the balance report](#reading-the-balance-report)
7. [Parameter reference](#parameter-reference)

## Input data

RadSplit works on a table with **one row per record**. A record is whatever your model sees as one sample: an image, a series, a lesion, or an examination. The table needs:

- a **patient identifier** column (required, no missing values), and
- the **characteristics to balance or compare**, such as label, age, sex, view, site, manufacturer, or kernel.

Any other columns (file paths, accession numbers) are carried through unchanged.

| patient_id | examination_id | image | label | site | scanner | age | sex |
|---|---|---|---|---|---|---|---|
| P0001 | P0001E0 | images/P0001E0.nii.gz | 1 | site 2 | scanner 1 | 64 | F |
| P0001 | P0001E1 | images/P0001E1.nii.gz | 1 | site 2 | scanner 3 | 64 | F |
| P0002 | P0002E0 | images/P0002E0.nii.gz | 0 | site 1 | scanner 2 | 71 | M |

The Python interface accepts a pandas DataFrame. The command line accepts a CSV or Parquet file, or a folder of images (use case I).

Numeric columns with more than 10 distinct values are treated as continuous; all others are treated as categorical. Override this with `continuous=[...]` or `categorical=[...]`. Characteristics that never change within a patient (such as sex) are balanced per patient; the others (such as view) are balanced per record.

To try the examples below without your own data, create a synthetic cohort with repeated examinations per patient:

```bash
python -c "from radsplit.synthetic import make_cohort; make_cohort(1000, seed=0).to_csv('cohort.csv', index=False)"
```

## Run it on your own dataset

This section takes you from your own data to a checked split in four steps: **scan → balance → hold-out → audit**. Each step shows the command-line version first, then the Python version. The use cases below explain each step in more detail.

You need:

- **Images** in one folder, with one subfolder per patient for NIfTI, MetaImage, and NRRD (for example `/data/ct/P001/scan.nii.gz`). DICOM files can be arranged any way, because the patient comes from the header.
- **A labels CSV** with one row per patient, a `patient_id` column that matches those folder names or the DICOM PatientID, and the characteristics you want to balance (for example `label, age, sex, site`).

If you already have a table with one row per record (see [Input data](#input-data)), skip step 1 and use that table wherever these steps use `manifest.csv`.

Install with the image readers:

```bash
python -m pip install -e '.[manifest]'
```

### Step 1. Scan the image folder

```bash
radsplit scan /data/ct --labels labels.csv --labels-on patient_id --output manifest.csv
```

This reads every image header and writes `manifest.csv`, with one row per series and your label columns joined in. It also prints an integrity summary. Check that the `patients` count is what you expect, and look at any `issues` before you continue. For example, `duplicate_checksum` means two files have identical content.

### Step 2. Make a balanced, patient-disjoint split

```bash
radsplit split manifest.csv --patient patient_id --target label \
  --balance site --balance sex --balance age \
  --ratios "train=0.7,tune=0.1,test=0.2" --seed 0 --output mysplit
```

The report shows the size of each set, confirms that `Patients in more than one partition` is 0, and lists the remaining imbalance (SMD) for each characteristic. `mysplit/assignments.csv` gives the set of each patient. `mysplit/split.json` holds the settings and the hash you need to reproduce the split. To make cross-validation folds instead, replace `--ratios` with `--folds 5`.

### Step 3. Hold out a scanner, site, or kernel (optional)

To test generalization, use one acquisition group as the test set:

```bash
radsplit holdout manifest.csv --patient patient_id --by site --value "site C" \
  --balance label --balance sex --balance age --mixed exclude --seed 0 --output holdout_siteC
```

`--by` is any column from the manifest, such as `site`, `manufacturer`, `scanner_model`, or `convolution_kernel`. Repeat `--value` to hold out several groups. The remaining patients are split into balanced training and tuning sets. The `holdout` command has no `--target` option, so pass your label with `--balance`. Before training, drop records marked `excluded`. See use case E for what `--mixed` does.

### Step 4. Audit the split

An audit needs a table with one row per record and a column that gives the set of each record. To audit the split from step 2, join the assignments back onto the manifest:

```bash
python -c "import pandas as pd; pd.read_csv('manifest.csv').merge(pd.read_csv('mysplit/assignments.csv'), on='patient_id').to_csv('splits.csv', index=False)"

radsplit audit splits.csv --patient patient_id --split split \
  --variable label --variable site --variable sex --variable age \
  --output audit.html --strict
```

`--strict` makes the command exit with status 1 if any patient is in more than one set or any characteristic has an absolute SMD above 0.1. The same command audits a split made by any other tool, or one that came with the dataset: point it at a table that has a set column.

### All four steps in Python

```python
import pandas as pd
from radsplit import audit, holdout, manifest, split

# 1. Scan
dataset = manifest.scan("/data/ct").join(pd.read_csv("labels.csv"), on="patient_id")
print(dataset.validate())                     # integrity issues, if any
records = dataset.series()                    # one row per series

# 2. Balance
result = split(records, patient="patient_id", target="label",
               balance=["site", "sex", "age"], seed=0)
print(result.report())
result.save("mysplit")

# 3. Hold out
held = holdout(records, patient="patient_id", by="site", values=["site C"],
               target="label", balance=["sex", "age"], seed=0)
print(held.audit.summary["holdout"])
held.save("holdout_siteC")

# 4. Audit
report = audit(result.data, patient="patient_id", split="split",
               variables=["label", "site", "sex", "age"])
print("Patient disjoint:", report.disjoint, "| Balanced:", report.balanced)
open("audit.html", "w").write(report.to_html())
```

`result.data` is your table with a `split` column added. Use it, or `result["train"]`, to build your training data loaders (see use case J).

## Example on a real dataset: CBIS-DDSM

[examples/cbisddsm.py](examples/cbisddsm.py) runs all four steps on a subset of [CBIS-DDSM](https://www.cancerimagingarchive.net/collection/cbis-ddsm/), a public mammography collection. CBIS-DDSM is a useful test case for two reasons:

- **Its DICOM PatientID is not the patient.** It holds an image key such as `Mass-Training_P_00001_LEFT_CC`, and the real patient is `P_00001`. The script rebuilds patient identifiers after the scan. Check for this in your own data too.
- **Its official split leaks when the tables are combined.** The mass and calcification tables each come with their own train/test split. Combined, some patients appear in both train and test, and some mammograms appear twice, once in each table.

The example needs about 4 GB of disk space for 40 patients. Run every command from the repository root.

**Step 1. Install.** This installs RadSplit with the DICOM readers and the TCIA downloader.

```bash
source .venv/bin/activate
python -m pip install -e '.[manifest]' tcia_utils
```

**Step 2. Add the label tables.** From the Data Access table on the [CBIS-DDSM page](https://www.cancerimagingarchive.net/collection/cbis-ddsm/), download the four "Description (csv)" files (not the images) and move them into `data/cbisddsm/`.

```bash
mkdir -p data/cbisddsm
mv ~/Downloads/mass_case_description_{train,test}_set.csv ~/Downloads/calc_case_description_{train,test}_set.csv data/cbisddsm/
ls data/cbisddsm
```

**Step 3. Choose the patients (dry run).** This picks 40 patients and prints the number of mammograms and the download size without downloading anything. The `Calling getSeries` line can take a few minutes.

```bash
python examples/cbisddsm.py fetch --dry-run
```

**Step 4. Download the images.** This downloads the full mammograms of those 40 patients (about 4 GB) into `data/cbisddsm/dicom/`. If it stops partway, run it again.

```bash
python examples/cbisddsm.py fetch
```

**Step 5. Check the download.** You should see one folder per mammogram, the number the dry run printed.

```bash
ls data/cbisddsm/dicom | wc -l
du -sh data/cbisddsm/dicom
```

**Step 6. Run scan → balance → hold-out → audit.** This prints a report for each step and writes the outputs to `out/cbisddsm/`. Open the two audit reports to compare the official split with the RadSplit split.

```bash
python examples/cbisddsm.py run
open out/cbisddsm/4_audit/official.html out/cbisddsm/4_audit/radsplit.html
```

| Path in `out/cbisddsm/` | Content |
|---|---|
| `1_scan/` | Manifest of every DICOM file and `records.csv`, one row per mammogram with its patient and labels |
| `2_balance/` | The RadSplit train/tune/test split: assignments, balance table, `split.json`, and audit report |
| `3_holdout/` | Calcification images held out as the test set, with the same files |
| `4_audit/` | Audit reports (`.html`, `.md`) of the official split and of the RadSplit split |

What to expect: the RadSplit split and the hold-out share no patients between sets. The official split shares any patient whose mass and calcification images are in different official sets.

## Use cases

### A. Balanced training, tuning, and test sets

The most common case: a 70/10/20 split with no patient in two sets and with the label and other characteristics balanced.

Python:

```python
import pandas as pd
from radsplit import split

records = pd.read_csv("cohort.csv")
result = split(
    records,
    patient="patient_id",
    target="label",                               # label column(s), listed first in the report
    balance=["site", "scanner", "sex", "age"],    # other characteristics to balance
    ratios={"train": 0.7, "tune": 0.1, "test": 0.2},
    seed=0,
)

print(result.report())                  # sizes, overlap, and balance summary
train = result["train"]                 # records of one set
for name, part in result.parts.items():
    print(name, len(part))
result.save("mysplit")                  # assignments, balance table, settings, reports
```

Command line:

```bash
radsplit split cohort.csv --patient patient_id --target label \
  --balance site --balance scanner --balance sex --balance age \
  --ratios "train=0.7,tune=0.1,test=0.2" --seed 0 --output mysplit
```

`result.data` is your table with a `split` column added. `result.assignments` has one row per patient.

### B. Custom set names and fractions

Use any names and any number of sets. Fractions must be positive and sum to 1.

```python
result = split(records, patient="patient_id", target="label",
               ratios={"development": 0.8, "holdout": 0.2}, seed=0)
```

```bash
radsplit split cohort.csv --target label --ratios "development=0.8,holdout=0.2"
```

### C. Cross-validation folds

Create K balanced, patient-disjoint folds named `fold1` to `foldK`.

```python
result = split(records, patient="patient_id", target="label",
               balance=["site", "age"], folds=5, seed=0)
print(result.data["split"].value_counts())
```

```bash
radsplit split cohort.csv --target label --balance site --balance age --folds 5 --output cvfolds
```

To export the folds to nnU-Net or to use them as a scikit-learn splitter, see use case J.

### D. Keep an official test set

When a dataset ships with an official test set, keep it and balance only the remaining patients. Put the set name in a column for the patients whose assignment is fixed and leave it empty for everyone else.

```python
import numpy as np

official_test_ids = set(records.loc[records["site"] == "site 4", "patient_id"])   # your official list
records["fixed"] = np.where(records["patient_id"].isin(official_test_ids), "test", None)
share = records.loc[records["fixed"] == "test", "patient_id"].nunique() / records["patient_id"].nunique()

result = split(
    records,
    patient="patient_id",
    target="label",
    balance=["sex", "age"],
    ratios={"train": 0.875 * (1 - share), "tune": 0.125 * (1 - share), "test": share},
    fixed="fixed",
    seed=0,
)
```

Set the test fraction equal to the share of patients in the fixed test set, as above, so the fixed patients fill it exactly. The same approach places a whole site in the test set. A patient cannot have two different fixed sets.

### E. Hold out a scanner, kernel, site, or visit

To test generalization to an acquisition group the model has not seen, put every record from that group in the test set and split the remaining patients into balanced training and tuning sets (default 87.5% and 12.5%).

```python
from radsplit import holdout
from radsplit.shift import shift_table

result = holdout(
    records,
    patient="patient_id",
    by="scanner",                 # column that defines the group
    values=["scanner 3"],         # one or more values to hold out
    target="label",
    balance=["site", "sex", "age"],
    mixed="exclude",              # how to handle patients scanned on several scanners
    seed=0,
)
print(result.report())
print(result.audit.summary["holdout"])        # test size, mixed patients, excluded records
print(shift_table(result).head(10))           # how the test set differs from training
```

```bash
radsplit holdout cohort.csv --patient patient_id --by scanner --value "scanner 3" \
  --balance site --balance sex --balance age --mixed exclude --output scanner3
```

**Mixed patients** have records both inside and outside the held-out group, for example examinations on two scanners. Choose one policy:

| `mixed=` | What happens | Use when |
|---|---|---|
| `"exclude"` (default) | Patient goes to the test set; their records from other groups are marked `excluded` | The test set should contain only held-out group records and no shared patients |
| `"test"` | All of the patient's records go to the test set | You want to keep every record and accept other groups in the test set |
| `"train"` | Patient stays in development; their held-out group records are marked `excluded` | Development must contain no held-out group record and a smaller test set is acceptable |

Excluded records keep the label `excluded` in `result.data["split"]`; drop them before training.

The **shift table** lists every characteristic by its standardized mean difference (SMD) between training and test sets. A hold-out test set often differs in more than the held-out variable (for example section thickness or disease prevalence). These differences are reported, not minimized, so that a scanner hold-out can be described accurately rather than read as a controlled comparison of one factor.

### F. Rotate over every group (leave one group out)

Hold out each group in turn, for example to report performance per scanner or per site. Groups with fewer than `min_patients` patients are skipped.

```python
from radsplit import leave_one_group_out

for scanner, fold in leave_one_group_out(records, patient="patient_id", by="scanner",
                                         target="label", balance=["site", "age"], min_patients=20):
    sizes = fold.audit.sizes.set_index("partition")["patients"].to_dict()
    print(scanner, sizes)
    fold.save(f"logo/{scanner}")
```

### G. Audit an existing split

Check a split made by any other tool or released with a dataset: patient overlap, overlap on other keys, set sizes, and balance.

From a table with a set column:

```python
from radsplit import audit

report = audit(
    records,                      # must contain a column with the set of each record
    patient="patient_id",
    split="split",                # name of that column
    variables=["label", "site", "scanner", "sex", "age"],
    also_disjoint=["examination_id"],    # other keys that must not be shared
)
print("Patient disjoint:", report.disjoint)
print("All terms with absolute SMD at most 0.1:", report.balanced)
print(report.overlap)
print(report.table())             # characteristics by set, ready for a manuscript table
open("audit.html", "w").write(report.to_html())
```

From index lists, for example the output of a scikit-learn splitter or a PyTorch `Subset`:

```python
from radsplit.frameworks import audit_indices

report = audit_indices(records, patient="patient_id",
                       partitions={"train": train_rows, "test": test_rows},
                       variables=["label", "age"])
```

Command line:

```bash
radsplit audit splits.csv --patient patient_id --split split \
  --variable label --variable site --variable age \
  --also-disjoint examination_id --output audit.html
```

When two separately released tables (for example mass and calcification tables) are combined, audit the combined table: each table can be patient-disjoint on its own while the combination is not.

### H. Check splits automatically in a pipeline

With `--strict`, the audit exits with status 1 when any patient appears in more than one set or any term has an absolute SMD above 0.1. Use it in continuous integration or before training starts.

```bash
radsplit audit data/splits.csv --patient patient_id --split split \
  --variable label --variable site --strict || { echo "Split check failed"; exit 1; }
```

In Python, check `report.disjoint` and `report.balanced`. To use a threshold other than 0.1, pass `threshold=` to `audit`.

### I. Start from a folder of images

`scan` reads image headers into a table with one row per file, groups DICOM instances into series, pairs masks with images, and checks integrity. Supported formats: DICOM (`.dcm`, `.dicom`, `.ima`, and files without a suffix that carry the DICOM signature), NIfTI (`.nii`, `.nii.gz`), MetaImage (`.mha`, `.mhd`), and NRRD (`.nrrd`, `.nhdr`). Requires the `manifest` extra.

```bash
radsplit scan /data/ct --output manifest.csv --labels labels.csv --labels-on patient_id
radsplit split /data/ct --labels labels.csv --target label --balance manufacturer --output ctsplit
```

```python
from radsplit import manifest, split

dataset = manifest.scan("/data/ct").join(pd.read_csv("labels.csv"), on="patient_id")
print(dataset.validate())                 # integrity issues, if any
series = dataset.series()                 # one row per series: use this for splitting
result = split(series, patient="patient_id", target="label", balance=["manufacturer"])
dataset.save("manifest")                  # manifest, metadata, and provenance
```

Split on `series()`, not on the file table, so that one record is one series rather than one slice. For DICOM, the patient identifier comes from the header. For NIfTI, MetaImage, and NRRD files, it is the first folder below the scanned folder (`/data/ct/P001/scan.nii.gz` gives `P001`); check identifiers derived from paths before use. Columns from the labels file fill the matching empty manifest columns; a column that already holds values must be renamed before joining.

### J. Use the split in nnU-Net, MONAI, PyTorch, or scikit-learn

**nnU-Net** (`splits_final.json` from cross-validation folds):

```python
from radsplit.export import nnunet_splits, write_json

folds = split(records, patient="patient_id", target="label", folds=5, seed=0)
write_json(nnunet_splits(folds.data, case="examination_id"),
           "nnUNet_preprocessed/Dataset001/splits_final.json")
```

`case` is the column holding the nnU-Net case identifier. Each fold in turn becomes the validation set.

**MONAI** (Decathlon-format data list):

```python
from radsplit.export import monai_datalist, write_json

datalist = monai_datalist(result.data, image="image", label="mask")   # column names
write_json(datalist, "datalist.json")
# monai.data.load_decathlon_datalist("datalist.json", data_list_key="train")
```

**PyTorch** (row positions for `Subset` or `SubsetRandomSampler`):

```python
from torch.utils.data import Subset
from radsplit.frameworks import indices, file_lists

index = indices(result)                      # {"train": array([...]), "tune": ..., "test": ...}
train_set = Subset(my_dataset, index["train"])  # my_dataset must be indexed like the input table
paths = file_lists(result, "image")          # or lists of file paths per set
```

**scikit-learn** (a patient-disjoint, balanced splitter for `cv=`):

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from radsplit.frameworks import BalancedGroupKFold

cv = BalancedGroupKFold(n_splits=5, covariates=records[["site", "sex"]], seed=0)
scores = cross_validate(LogisticRegression(), records[["age"]], records["label"],
                        groups=records["patient_id"], cv=cv)
search = GridSearchCV(LogisticRegression(), {"C": [0.1, 1, 10]}, cv=cv)
search.fit(records[["age"]], records["label"], groups=records["patient_id"])
```

`groups` must hold the patient identifier. The label `y` is balanced automatically; `covariates` adds further columns, one row per sample in the same order as `X`.

## Outputs and reproducibility

`result.save(folder)` and the command-line `--output` write:

| File | Content |
|---|---|
| `assignments.csv` | One row per patient with its set |
| `balance.csv` | One row per term with pooled and per-set means and SMDs |
| `split.json` | All settings, the summary, and the split hash |
| `audit.md`, `audit.html` | Readable report of sizes, overlap, and balance |

The same data, settings, and seed always give the same assignment and the same hash (`result.split_hash`, for example `rs:sha256:...`). Report the RadSplit version, the seed, and the hash in a paper so that reviewers can regenerate the split. `split.json` contains the information on data partitioning requested by the Checklist for Artificial Intelligence in Medical Imaging (CLAIM).

## Reading the balance report

Each numeric column and each level of a categorical column is a **term**. For each term, RadSplit computes the standardized mean difference (SMD) between sets:

- continuous term: (mean in set a minus mean in set b) / standard deviation in the full cohort
- categorical level: (proportion in set a minus proportion in set b) / sqrt(p(1 minus p)), where p is the proportion in the full cohort

With three or more sets, the SMD of a term is the largest absolute pairwise value. The report gives the mean and maximum absolute SMD across terms and the number of terms above 0.1. An SMD above 0.1 is a common convention for meaningful imbalance; it is not a statistical test. Plot the terms with `result.audit.loveplot()` (requires the `plot` extra).

Balancing applies only to measured characteristics that you include. Unmeasured differences between sets are not controlled.

## Parameter reference

`split(data, *, patient, ...)`

| Parameter | Default | Meaning |
|---|---|---|
| `patient` | required | Patient identifier column |
| `target` | none | Label column or columns; balanced and listed first |
| `balance` | none | Other columns to balance |
| `ratios` | `{"train": 0.7, "tune": 0.1, "test": 0.2}` | Set names and fractions |
| `folds` | none | Number of cross-validation folds (replaces `ratios`) |
| `fixed` | none | Column with assignments to keep (for example an official test set) |
| `continuous`, `categorical` | automatic | Override type detection |
| `weights` | equal | Relative weight per column in the objective |
| `balance_records` | `True` | Also balance the number of records per patient |
| `shape_bins` | `4` | Quantile bins used to balance the shape of continuous columns; 0 balances means only |
| `rare` | `0.01` | Levels seen in fewer than this fraction of units are pooled |
| `seed` | `0` | Random seed |
| `restarts` | `1` | Random starts; the best result is kept |
| `max_swaps` | `20000` | Maximum accepted patient exchanges per start |
| `column` | `"split"` | Name of the added set column |

`holdout(data, *, patient, by, values, ...)` also takes `development` (default `{"train": 0.875, "tune": 0.125}`), `test` (name of the test set, default `"test"`), and `mixed` (`"exclude"`, `"test"`, or `"train"`). Other options are passed to `split`.

`audit(data, *, patient, split, variables, ...)` also takes `partitions` (order of set names), `also_disjoint` (other keys to check), and `threshold` (default 0.1).

Run `radsplit split --help`, `radsplit holdout --help`, `radsplit audit --help`, or `radsplit scan --help` for the command-line options.

