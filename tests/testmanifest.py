import shutil

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from pydicom.data import get_testdata_file

from radsplit import manifest
from radsplit.manifest import Manifest


def save(path, shape=(4, 5, 6), affine=None, fill=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    affine = np.diag([1, 2, 3, 1]) if affine is None else affine
    nib.save(nib.Nifti1Image(np.full(shape, fill, dtype=np.uint8), affine), path)


def test_scan_summary_fingerprint_and_persistence(tmp_path):
    save(tmp_path / "case001" / "ct.nii.gz")
    save(tmp_path / "case001" / "ct_mask.nii.gz")
    dataset = manifest.scan(tmp_path)
    summary = dataset.summary()
    assert (summary["patients"], summary["images"], summary["masks"]) == (1, 1, 1)
    assert dataset.masks.iloc[0]["mask_for"] == "case001/ct.nii.gz"
    assert dataset.validate()["valid"]
    fingerprint = dataset.fingerprint()
    dataset.save(tmp_path / "saved.manifest")
    loaded = Manifest.load(tmp_path / "saved.manifest")
    assert loaded.fingerprint() == fingerprint
    assert loaded.provenance[0]["operation"] == "scan"


def test_fingerprint_ignores_absolute_location(tmp_path):
    for name in ("one", "two"):
        save(tmp_path / name / "same.nii.gz")
    assert manifest.scan(tmp_path / "one").fingerprint() == manifest.scan(tmp_path / "two").fingerprint()


def test_scan_public_dicom_sample(tmp_path):
    source = get_testdata_file("CT_small.dcm")
    (tmp_path / "patient").mkdir()
    shutil.copy(source, tmp_path / "patient" / "slice.dcm")
    dataset = manifest.scan(tmp_path)
    row = dataset.table.iloc[0]
    assert row["file_type"] == "dicom" and row["modality"] == "CT"
    assert pd.notna(row["patient_id"])
    assert dataset.validate()["valid"]


@pytest.mark.parametrize(
    "mask_kwargs, field",
    [
        (dict(shape=(4, 5, 7)), "depth"),
        (dict(affine=np.diag([1, 2, 4, 1])), "spacing_z"),
        (dict(affine=np.array([[1, 0, 0, 9], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 1.0]])), "affine"),
    ],
)
def test_image_mask_geometry_mismatch_is_an_error(tmp_path, mask_kwargs, field):
    save(tmp_path / "case" / "mr.nii.gz")
    save(tmp_path / "case" / "mr_seg.nii.gz", **mask_kwargs)
    report = manifest.scan(tmp_path).validate()
    assert not report["valid"]
    mismatch = [i for i in report["issues"] if i["check"] == "image_mask_geometry_mismatch"]
    assert field in mismatch[0]["fields"]


def test_warnings_for_empty_unpaired_and_duplicate_files(tmp_path):
    save(tmp_path / "a" / "ct.nii.gz")
    save(tmp_path / "a" / "ct_mask.nii.gz", fill=0)
    save(tmp_path / "b" / "one.nii.gz")
    save(tmp_path / "b" / "two.nii.gz")
    save(tmp_path / "b" / "two_mask.nii.gz", fill=2)
    counts = manifest.scan(tmp_path).validate()["counts"]
    assert counts["empty_mask"] == 1
    assert counts["unpaired_mask"] == 1
    assert counts["duplicate_checksum"] >= 1


def test_unreadable_and_packaging_files(tmp_path):
    save(tmp_path / "a" / "ct.nii.gz")
    (tmp_path / "a" / "broken.nii.gz").write_bytes(b"not a nifti file")
    (tmp_path / "a" / "._ct.nii.gz").write_bytes(b"appledouble")
    dataset = manifest.scan(tmp_path)
    assert dataset.metadata["ignored_packaging_artifacts"] == ["a/._ct.nii.gz"]
    report = dataset.validate()
    assert not report["valid"] and report["counts"]["unreadable_file"] == 1


def test_manifest_split_integration(tmp_path):
    for index in range(12):
        save(tmp_path / f"case{index:02d}" / "ct.nii.gz", shape=(4, 5, 6 + index % 3))
    dataset = manifest.scan(tmp_path)
    result = dataset.split(target="depth", seed=0)
    assert result.audit.disjoint
    assert dataset.table["split"].notna().all()
    assert dataset.provenance[-1]["operation"] == "split"


def test_metaimage_and_nrrd_files_are_scanned(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    image = sitk.GetImageFromArray(np.ones((6, 5, 4), dtype=np.uint8))
    image.SetSpacing((0.5, 0.5, 2.0))
    for name in ("case01/ct.mha", "case02/ct.nrrd"):
        (tmp_path / name).parent.mkdir()
        sitk.WriteImage(image, str(tmp_path / name))
    dataset = manifest.scan(tmp_path)
    table = dataset.table.set_index("patient_id")
    assert set(table["file_type"]) == {"metaimage", "nrrd"}
    assert table.loc["case01", "spacing_z"] == 2.0
    assert (table.loc["case01", "rows"], table.loc["case01", "depth"]) == (4, 6)
    assert dataset.validate()["valid"]


def test_dicom_without_suffix_and_series_table(tmp_path):
    source = get_testdata_file("CT_small.dcm")
    (tmp_path / "p").mkdir()
    shutil.copy(source, tmp_path / "p" / "IM0001")
    shutil.copy(source, tmp_path / "p" / "slice.dcm")
    (tmp_path / "p" / "notes").write_text("not an image")
    dataset = manifest.scan(tmp_path)
    assert len(dataset.table) == 2
    series = dataset.series()
    assert len(series) == 1 and series.iloc[0]["files"] == 2


def test_join_labels_then_split(tmp_path):
    for index in range(12):
        save(tmp_path / f"case{index:02d}" / "ct.nii.gz")
    labels = pd.DataFrame({"patient_id": [f"case{i:02d}" for i in range(12)],
                           "diagnosis": [i % 2 for i in range(12)]})
    dataset = manifest.scan(tmp_path).join(labels)
    assert dataset.table["diagnosis"].notna().all()
    result = dataset.split(target="diagnosis")
    assert result.audit.disjoint
    with pytest.raises(ValueError):
        dataset.join(pd.concat([labels, labels]))


def test_join_fills_empty_placeholder_columns(tmp_path):
    for index in range(6):
        save(tmp_path / f"case{index:02d}" / "ct.nii.gz")
    labels = pd.DataFrame({"patient_id": [f"case{i:02d}" for i in range(6)],
                           "label": [i % 2 for i in range(6)]})
    dataset = manifest.scan(tmp_path).join(labels)
    assert "label_x" not in dataset.table and dataset.table["label"].notna().all()
    result = dataset.split(target="label", ratios={"train": 0.5, "test": 0.5})
    assert result.audit.disjoint
    with pytest.raises(ValueError):
        dataset.join(labels)


def test_acquisition_parameters_are_read_from_dicom_headers(tmp_path):
    import pydicom

    source = pydicom.dcmread(get_testdata_file("CT_small.dcm"))
    source.ConvolutionKernel = "B70f"
    source.MagneticFieldStrength = None
    (tmp_path / "p").mkdir()
    source.save_as(tmp_path / "p" / "a.dcm")
    row = manifest.scan(tmp_path).series().iloc[0]
    assert row["convolution_kernel"] == "B70f"
    assert row["kvp"] == 120.0
    assert row["slice_thickness"] == 5.0
    assert row["manufacturer"] == "GE MEDICAL SYSTEMS"
