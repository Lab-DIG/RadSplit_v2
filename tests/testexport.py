import json

from typer.testing import CliRunner

from radsplit import split
from radsplit.cli import app
from radsplit.export import monai_datalist, nnunet_splits, write_json
from radsplit.synthetic import make_cohort

runner = CliRunner()


def test_monai_datalist_and_nnunet_splits(tmp_path):
    frame = make_cohort(60, seed=0)
    frame["image"] = frame["examination_id"] + ".nii.gz"
    result = split(frame, patient="patient_id", target="label", folds=3)
    datalist = monai_datalist(result.data, image="image", label="label")
    assert set(datalist) == {"fold1", "fold2", "fold3"}
    assert set(datalist["fold1"][0]) == {"image", "label"}
    splits = nnunet_splits(result.data, case="examination_id")
    assert len(splits) == 3
    for fold in splits:
        assert not set(fold["train"]) & set(fold["val"])
    path = write_json(splits, tmp_path / "splits.json")
    assert json.loads(path.read_text())[0]["val"]


def test_command_line_split_then_audit(tmp_path):
    table = tmp_path / "cohort.csv"
    make_cohort(600, seed=4).to_csv(table, index=False)
    out = tmp_path / "out"
    made = runner.invoke(app, ["split", str(table), "--patient", "patient_id", "--target", "label",
                               "--balance", "site", "--balance", "age", "--output", str(out)])
    assert made.exit_code == 0, made.output
    assert (out / "assignments.csv").exists()
    import pandas as pd

    merged = pd.read_csv(table).merge(pd.read_csv(out / "assignments.csv"), on="patient_id")
    merged.to_csv(tmp_path / "merged.csv", index=False)
    checked = runner.invoke(app, ["audit", str(tmp_path / "merged.csv"), "--patient", "patient_id",
                                  "--variable", "label", "--variable", "site", "--strict",
                                  "--output", str(tmp_path / "audit.html")])
    assert checked.exit_code == 0, checked.output
    assert (tmp_path / "audit.html").read_text().startswith("<!doctype html>")


def test_command_line_split_from_image_folder(tmp_path):
    import nibabel as nib
    import numpy as np
    import pandas as pd

    for index in range(15):
        folder = tmp_path / "images" / f"case{index:02d}"
        folder.mkdir(parents=True)
        nib.save(nib.Nifti1Image(np.ones((3, 3, 3 + index % 2), dtype=np.uint8), np.eye(4)),
                 folder / "mr.nii.gz")
    labels = tmp_path / "labels.csv"
    pd.DataFrame({"patient_id": [f"case{i:02d}" for i in range(15)],
                  "tumor": [i % 3 == 0 for i in range(15)]}).to_csv(labels, index=False)
    out = tmp_path / "out"
    made = runner.invoke(app, ["split", str(tmp_path / "images"), "--target", "tumor",
                               "--balance", "depth", "--labels", str(labels),
                               "--output", str(out)])
    assert made.exit_code == 0, made.output
    assert len(pd.read_csv(out / "assignments.csv")) == 15
