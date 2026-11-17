"""Medication feature engineering."""

import pandas as pd


def aggregate_medications(
    medication: pd.DataFrame, patient_id: str, group_column: str = "Drug_group"
) -> pd.DataFrame:
    """Return one binary feature per medication group and patient.

    The `Drug_group` mapping is intentionally left to the migrated clinical rules.
    """
    if group_column not in medication:
        raise KeyError(f"Expected {group_column!r} in medication data.")
    return pd.crosstab(medication[patient_id], medication[group_column]).reset_index()
