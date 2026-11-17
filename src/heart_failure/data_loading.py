"""Raw data input and basic validation."""

from pathlib import Path

import pandas as pd


def load_raw_data(clinical_path: str | Path, medication_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the clinical and medication tables exported from the source dataset."""
    clinical = pd.read_csv(clinical_path, index_col=0)
    medication = pd.read_csv(medication_path, index_col=0)
    return clinical, medication


def validate_patient_ids(clinical: pd.DataFrame, medication: pd.DataFrame, patient_id: str) -> None:
    """Raise an error when the clinical table has duplicated patient identifiers."""
    if clinical[patient_id].duplicated().any():
        raise ValueError(f"Duplicate values found in {patient_id!r} in clinical data.")
    if not set(medication[patient_id]).issubset(set(clinical[patient_id])):
        raise ValueError("Medication data contains patient IDs absent from clinical data.")
