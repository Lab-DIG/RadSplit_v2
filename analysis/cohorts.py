"""Harmonized record tables for the four public benchmark cohorts.

Each loader returns a :class:`Cohort` whose ``records`` table has one row per
record (lesion view record or radiograph), a ``patient_id`` column, and the
covariates listed in ``variables``.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

NIH_FINDINGS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", "Emphysema",
    "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
]
PADCHEST_NOT_FINDINGS = {"unchanged", "chronic changes", "exclude", "suboptimal study", ""}


@dataclass
class Cohort:
    key: str
    name: str
    modality: str
    record_unit: str
    records: pd.DataFrame
    variables: list[str]
    stratify: str
    official: str | None = None

    @property
    def n_patients(self) -> int:
        return int(self.records["patient_id"].nunique())


def cbisddsm() -> Cohort:
    frames = []
    for file, kind, official in [
        ("masstrain.csv", "mass", "train"), ("masstest.csv", "mass", "test"),
        ("calctrain.csv", "calcification", "train"), ("calctest.csv", "calcification", "test"),
    ]:
        path = DATA / "cbisddsm" / file
        if not path.exists():
            # file names as downloaded from TCIA, e.g. mass_case_description_train_set.csv
            prefix = "mass" if kind == "mass" else "calc"
            path = DATA / "cbisddsm" / f"{prefix}_case_description_{official}_set.csv"
        frame = pd.read_csv(path).rename(
            columns={"breast density": "breast_density"})
        frame["abnormality"] = kind
        frame["official_split"] = official
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    records = pd.DataFrame(
        {
            "patient_id": raw["patient_id"],
            "lesion_id": raw["patient_id"] + "|" + raw["left or right breast"] + "|"
            + raw["abnormality"] + "|" + raw["abnormality id"].astype(str),
            "malignant": raw["pathology"].eq("MALIGNANT").astype(int),
            "abnormality": raw["abnormality"],
            "breast_density": raw["breast_density"].astype(int).astype(str),
            "assessment": raw["assessment"].astype(int).astype(str),
            "subtlety": raw["subtlety"].astype(int).astype(str),
            "view": raw["image view"],
            "laterality": raw["left or right breast"],
            "official_split": raw["official_split"],
            "mass_shape": raw.get("mass shape"),
            "mass_margins": raw.get("mass margins"),
            "calc_type": raw.get("calc type"),
            "calc_distribution": raw.get("calc distribution"),
        }
    )
    variables = ["malignant", "abnormality", "breast_density", "assessment", "subtlety",
                 "view", "laterality"]
    return Cohort("cbisddsm", "CBIS-DDSM", "Mammography", "lesion view record", records,
                  variables, "malignant", "official_split")


def nih() -> Cohort:
    raw = pd.read_csv(DATA / "external" / "nihchestxray14.csv.gz")
    labels = raw["Finding Labels"].str.get_dummies("|")
    records = pd.DataFrame(
        {
            "patient_id": raw["Patient ID"].astype(int),
            "image": raw["Image Index"],
            "any_finding": 1 - labels["No Finding"],
            "view": raw["View Position"],
            "sex": raw["Patient Gender"],
            "age": raw["Patient Age"].astype(float),
        }
    )
    for finding in NIH_FINDINGS:
        records[finding.lower()] = labels[finding].to_numpy()
    test = set((DATA / "external" / "nihtestlist.txt").read_text().split())
    records["official_split"] = np.where(records["image"].isin(test), "test", "train")
    variables = ["any_finding", *[f.lower() for f in NIH_FINDINGS], "view", "sex", "age"]
    return Cohort("nih", "NIH ChestX-ray14", "Chest radiography", "radiograph", records,
                  variables, "any_finding", "official_split")


def padchest(n_labels: int = 12) -> Cohort:
    raw = pd.read_csv(DATA / "external" / "padchest.csv.gz", low_memory=False)
    raw = raw[raw["Labels"].notna() & ~raw["Projection"].isin(["EXCLUDE", "UNK"])]
    raw = raw.reset_index(drop=True)
    parsed = [sorted({x.strip() for x in ast.literal_eval(text)}) for text in raw["Labels"]]
    counts = Counter(label for row in parsed for label in row)
    findings = [label for label, _ in counts.most_common()
                if label not in PADCHEST_NOT_FINDINGS and label != "normal"][: n_labels - 1]
    year = (raw["StudyDate_DICOM"] // 10000).astype(int)
    birth = raw.groupby("PatientID")["PatientBirth"].transform(
        lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
    sex = raw.groupby("PatientID")["PatientSex_DICOM"].transform(
        lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
    age = (year - birth).where(lambda a: (a >= 0) & (a <= 105))
    records = pd.DataFrame(
        {
            "patient_id": raw["PatientID"].astype(str),
            "image": raw["ImageID"],
            "normal": [int("normal" in row) for row in parsed],
            "projection": raw["Projection"],
            "manufacturer": raw["Manufacturer_DICOM"].replace(
                {"PhilipsMedicalSystems": "Philips", "ImagingDynamicsCompanyLtd": "Imaging Dynamics"}),
            "study_year": year.clip(lower=2009).astype(str),
            "label_source": raw["MethodLabel"].replace({"RNN_model": "automated"}),
            "sex": sex.where(sex.isin(["M", "F"])),
            "age": age,
        }
    )
    names = []
    for finding in findings:
        name = finding.replace(" ", "_")
        records[name] = [int(finding in row) for row in parsed]
        names.append(name)
    variables = ["normal", *names, "projection", "manufacturer", "study_year", "label_source",
                 "sex", "age"]
    return Cohort("padchest", "PadChest", "Chest radiography", "radiograph", records,
                  variables, "normal")


def covid() -> Cohort:
    raw = pd.read_csv(DATA / "external" / "covidmetadata.csv")
    raw = raw[raw["modality"] == "X-ray"].reset_index(drop=True)
    finding = raw["finding"].fillna("todo")
    group = np.select(
        [finding.str.contains("COVID-19"), finding.str.startswith("Pneumonia"),
         finding.eq("todo")],
        ["COVID-19", "other pneumonia", "unlabeled"], default="other")
    country = raw["location"].str.split(",").str[-1].str.strip()
    records = pd.DataFrame(
        {
            "patient_id": raw["patientid"].astype(str),
            "image": raw["filename"],
            "folder": raw["folder"],
            "covid19": (group == "COVID-19").astype(int),
            "finding": group,
            "view": raw["view"].replace({"AP Erect": "AP"}),
            "sex": raw["sex"],
            "age": raw["age"],
            "country": country,
        }
    )
    variables = ["finding", "view", "sex", "age", "country"]
    return Cohort("covid", "COVID-19 Image Data Collection", "Chest radiography", "radiograph",
                  records, variables, "covid19")


def _idc(collection: str) -> pd.DataFrame:
    """CT series of one Imaging Data Commons collection with acquisition parameters."""
    import idc_index_data

    index = pd.read_parquet(
        idc_index_data.IDC_INDEX_PARQUET_FILEPATH,
        columns=["collection_id", "PatientID", "SeriesInstanceUID", "Modality", "StudyDate",
                 "Manufacturer", "ManufacturerModelName", "instanceCount"],
        filters=[("collection_id", "==", collection), ("Modality", "==", "CT")])
    ct = pd.read_parquet(DATA / "external" / "idcctindex.parquet",
                         columns=["SeriesInstanceUID", "ImageType", "SliceThickness", "KVP",
                                  "ConvolutionKernel"])

    def text(value):
        if value is None or (np.isscalar(value) and pd.isna(value)):
            return None
        if isinstance(value, (list, np.ndarray)):
            return "|".join(map(str, value))
        return str(value)

    frame = index.merge(ct, on="SeriesInstanceUID", how="left")
    image_type = frame["ImageType"].map(text).fillna("")
    frame = frame[~image_type.str.contains("LOCALIZER")].reset_index(drop=True)
    thickness = pd.to_numeric(frame["SliceThickness"].map(text), errors="coerce").round(2)
    kvp = pd.to_numeric(frame["KVP"].map(text), errors="coerce").round(0)
    manufacturer = frame["Manufacturer"].astype(object).replace(
        {"GE MEDICAL S]STEMS": "GE MEDICAL SYSTEMS"}).map(
        lambda v: None if v is None or pd.isna(v) else str(v).split()[0].upper()
        .replace("PHILIPS", "Philips").replace("SIEMENS", "Siemens").replace("TOSHIBA", "Toshiba"))
    return pd.DataFrame(
        {
            "patient_id": frame["PatientID"].astype(str),
            "series": frame["SeriesInstanceUID"],
            "manufacturer": manufacturer,
            "scanner_model": frame["ManufacturerModelName"].astype(object),
            "kernel": frame["ConvolutionKernel"].map(text).replace({"": None}),
            "slice_thickness": thickness.map(lambda v: None if pd.isna(v) else f"{v:g} mm"),
            "kvp": kvp.map(lambda v: None if pd.isna(v) else f"{v:.0f}"),
            "study_year": frame["StudyDate"].astype(str).str[:4],
        }
    )


def nlst() -> Cohort:
    records = _idc("nlst")
    records["manufacturer_code"] = records["manufacturer"].astype("category").cat.codes
    variables = ["manufacturer", "scanner_model", "kernel", "slice_thickness", "kvp", "study_year"]
    return Cohort("nlst", "NLST", "Chest CT", "CT series", records, variables,
                  "manufacturer_code")


def lidc() -> Cohort:
    import sqlite3

    records = _idc("lidc_idri").drop(columns=["study_year"])
    with sqlite3.connect(DATA / "external" / "pylidc.sqlite") as connection:
        scans = pd.read_sql("select id, series_instance_uid, contrast_used from scans", connection)
        notes = pd.read_sql("select scan_id, malignancy from annotations", connection)
    per_scan = notes.groupby("scan_id")["malignancy"].agg(["size", "max"])
    scans["nodule"] = scans["id"].isin(per_scan.index).astype(int)
    scans["suspicious_nodule"] = scans["id"].map(per_scan["max"]).ge(4).astype(int)
    records = records.merge(scans.drop(columns="id"), left_on="series",
                            right_on="series_instance_uid", how="left")
    records["contrast"] = records.pop("contrast_used").astype(int)
    variables = ["suspicious_nodule", "nodule", "contrast", "manufacturer", "scanner_model",
                 "kernel", "slice_thickness", "kvp"]
    return Cohort("lidc", "LIDC-IDRI", "Chest CT", "CT series", records, variables,
                  "suspicious_nodule")


LOADERS = {"covid": covid, "lidc": lidc, "cbisddsm": cbisddsm, "nih": nih, "nlst": nlst,
           "padchest": padchest}


def load(key: str) -> Cohort:
    return LOADERS[key]()


def load_all() -> list[Cohort]:
    return [loader() for loader in LOADERS.values()]
