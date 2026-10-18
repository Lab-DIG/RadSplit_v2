"""RadSplit demo on CBIS-DDSM, a public mammography dataset: scan, balance, hold out, audit.

Put the four CBIS-DDSM case description CSVs in data/cbisddsm/ (see GUIDE.md), then:

    python examples/cbisddsm.py fetch --dry-run   # show what will be downloaded
    python examples/cbisddsm.py fetch             # download mammograms of 40 patients (~4 GB)
    python examples/cbisddsm.py run               # run RadSplit; outputs go to out/cbisddsm/
"""

import sys
from pathlib import Path

import pandas as pd

from radsplit import audit, holdout, manifest, split

DATA = Path("data/cbisddsm")
OUT = Path("out/cbisddsm")


def load_labels() -> pd.DataFrame:
    """One row per image from the four CSVs, which have one row per lesion."""
    frames = []
    for path in sorted(DATA.glob("*_case_description_*_set.csv")):
        frame = pd.read_csv(path)
        frame.columns = frame.columns.str.strip().str.lower().str.replace(" ", "_")  # tables differ
        frame["official"] = "train" if "train" in path.name else "test"
        frames.append(frame)
    if not frames:
        sys.exit("Put the four CBIS-DDSM case description CSVs in data/cbisddsm/ (see GUIDE.md)")
    cases = pd.concat(frames)
    cases["image_key"] = cases["image_file_path"].str.split("/").str[0]
    cases["malignant"] = cases["pathology"].eq("MALIGNANT").astype(int)
    return cases.groupby("image_key", as_index=False).agg(
        patient_id=("patient_id", "first"), malignant=("malignant", "max"),
        abnormality=("abnormality_type", "first"), density=("breast_density", "first"),
        view=("image_view", "first"), official=("official", "first"))


def fetch(dry_run: bool) -> None:
    """Download the full mammograms of 20 patients with both mass and calcification
    images (needed for the hold-out) and 20 patients with one type."""
    from tcia_utils import nbia

    labels = load_labels()
    types = labels.groupby("patient_id")["abnormality"].nunique()
    patients = list(types[types > 1].index[:20]) + list(types[types == 1].index[:20])
    keys = labels.loc[labels["patient_id"].isin(patients), "image_key"]
    series = nbia.getSeries(collection="CBIS-DDSM", format="df")
    full = series[series["SeriesDescription"].str.contains("full mammogram", na=False)
                  & series["PatientID"].isin(keys)]
    print(f"{len(full)} mammograms of {len(patients)} patients, "
          f"{full['FileSize'].astype(float).sum() / 1e9:.1f} GB")
    if not dry_run:
        nbia.downloadSeries(full["SeriesInstanceUID"].tolist(), input_type="list",
                            path=str(DATA / "dicom"))


def heading(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def scan() -> pd.DataFrame:
    """Step 1: read every DICOM header into one row per mammogram and attach the labels."""
    heading("STEP 1. SCAN: read the DICOM headers")
    dataset = manifest.scan(DATA / "dicom")
    folder = dataset.save(OUT / "1_scan")
    # In CBIS-DDSM the DICOM PatientID is an image key (Mass-Training_P_00001_LEFT_CC);
    # the labels map it to the real patient (P_00001).
    records = dataset.series().rename(columns={"patient_id": "image_key"})
    records = records.merge(load_labels(), on="image_key")
    records.to_csv(folder / "records.csv", index=False)
    print(f"Files read          {len(dataset.table)}")
    print(f"Mammograms          {len(records)}")
    print(f"DICOM PatientIDs    {records['image_key'].nunique()}")
    print(f"Real patients       {records['patient_id'].nunique()}")
    print(f"Integrity issues    {dataset.validate()['counts'] or 'none'}")
    print(f"\nSaved to {folder}/ (manifest, records.csv)")
    return records


def balance(records: pd.DataFrame):
    """Step 2: patient-disjoint train/tune/test sets with balanced characteristics."""
    heading("STEP 2. BALANCE: patient-disjoint train/tune/test split")
    result = split(records, patient="patient_id", target="malignant",
                   balance=["density", "view", "abnormality"], seed=0)
    folder = result.save(OUT / "2_balance")
    print(result.report())
    print(f"\nSaved to {folder}/ (assignments, balance table, split.json, audit report)")
    return result


def hold_out(records: pd.DataFrame) -> None:
    """Step 3: calcification images as an unseen test set."""
    heading("STEP 3. HOLD-OUT: calcification images as the test set")
    result = holdout(records, patient="patient_id", by="abnormality", values=["calcification"],
                     target="malignant", balance=["density", "view"], seed=0)
    folder = result.save(OUT / "3_holdout")
    print(result.report())
    info = result.audit.summary["holdout"]
    print(f"\nPatients with both types   {info['mixed_patients']}")
    print(f"Images excluded            {info['excluded_records']} (their mass images)")
    print(f"\nSaved to {folder}/ (assignments, balance table, split.json, audit report)")


def audit_splits(records: pd.DataFrame, result) -> None:
    """Step 4: audit the official split and the RadSplit split."""
    heading("STEP 4. AUDIT: official split vs RadSplit split")
    folder = OUT / "4_audit"
    folder.mkdir(parents=True, exist_ok=True)
    print(f"{'Split':<10}{'Shared patients':>18}{'Max |SMD|':>12}")
    for name, data, column in [("official", records, "official"), ("radsplit", result.data, "split")]:
        report = audit(data, patient="patient_id", split=column,
                       variables=["malignant", "density", "view", "abnormality"])
        (folder / f"{name}.html").write_text(report.to_html(), encoding="utf-8")
        (folder / f"{name}.md").write_text(report.to_markdown(), encoding="utf-8")
        summary = report.summary
        print(f"{name:<10}{summary['patients_in_multiple_partitions']:>18}{summary['max_abs_smd']:>12.3f}")
    print(f"\nSaved to {folder}/ (official.html, radsplit.html)")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "fetch":
        fetch(dry_run="--dry-run" in sys.argv)
    elif command == "run":
        records = scan()
        result = balance(records)
        hold_out(records)
        audit_splits(records, result)
    else:
        sys.exit(__doc__)
