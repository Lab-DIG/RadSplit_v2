"""Column schema of the canonical manifest."""

MANIFEST_COLUMNS = [
    "patient_id", "study_id", "series_id", "instance_id", "path", "relative_path", "file_type",
    "modality", "role", "label", "mask_for", "site", "manufacturer", "scanner_model",
    "institution", "station_name", "software_version", "convolution_kernel",
    "slice_thickness", "kvp", "field_strength", "contrast_agent", "body_part",
    "series_description", "patient_sex", "patient_age",
    "study_date", "series_date", "rows", "columns", "depth", "spacing_x", "spacing_y",
    "spacing_z", "orientation", "affine", "frame_of_reference_uid", "study_instance_uid",
    "series_instance_uid", "sop_instance_uid", "checksum", "source", "parent_dataset_id",
]
SCHEMA_VERSION = 3
