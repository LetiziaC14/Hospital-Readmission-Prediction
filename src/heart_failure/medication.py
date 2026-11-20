"""Medication feature engineering used in the original notebook."""

import pandas as pd

DRUG_GROUPS = {
    "sulfotanshinone sodium injection": "Antihypertensive",
    "Furosemide tablet": "Diuretic",
    "Enoxaparin Sodium injection": "Antithrombotic",
    "Meglumine Adenosine Cyclophosphate for injection": "Cardiac_stimulant",
    "Furosemide injection": "Diuretic",
    "Milrinone injection": "Cardiac_stimulant",
    "Metoprolol Succinate Sustained-release tablet": "Beta_blocker",
    "Deslanoside injection": "Cardiac_stimulant",
    "Torasemide tablet": "Cardiac_stimulant",
    "Benazepril hydrochloride tablet": "Antihypertensive",
    "Atorvastatin calcium tablet": "LLA",
    "Aspirin enteric-coated tablet": "Antithrombotic",
    "Digoxin tablet": "Cardiac_stimulant",
    "Hydrochlorothiazide tablet": "Diuretic",
    "metoprolol tartrate injection": "Beta_blocker",
    "Spironolactone tablet": "Diuretic",
    "Valsartan Dispersible tablet": "Antihypertensive",
    "Dobutamine hydrochloride injection": "Cardiac_stimulant",
    "Clopidogrel Hydrogen Sulphate tablet": "Antithrombotic",
    "Isoprenaline Hydrochloride injection": "Cardiac_stimulant",
    "Nitroglycerin injection": "Antihypertensive",
    "Shenfu injection": "Antihypertensive",
    "Isosorbide Mononitrate Sustained Release tablet": "Antihypertensive",
    "warfarin sodium tablet": "Antithrombotic",
    "Heparin Sodium injection": "Antithrombotic",
}


def assign_drug_groups(medication: pd.DataFrame) -> pd.DataFrame:
    """Map medication names to the six clinical groups from the notebook."""
    result = medication.copy()
    result["Drug_group"] = result["Drug_name"].map(DRUG_GROUPS)
    unknown = result.loc[result["Drug_group"].isna(), "Drug_name"].dropna().unique()
    if len(unknown):
        raise ValueError(f"Unmapped medication names: {unknown.tolist()}")
    return result


def aggregate_medications(
    medication: pd.DataFrame, patient_id: str, group_column: str = "Drug_group"
) -> pd.DataFrame:
    """Return one binary feature per medication group and patient.

    The `Drug_group` mapping is intentionally left to the migrated clinical rules.
    """
    if group_column not in medication:
        raise KeyError(f"Expected {group_column!r} in medication data.")
    encoded = (pd.crosstab(medication[patient_id], medication[group_column]) > 0).astype(int)
    encoded = encoded.add_suffix("_Drug")
    encoded.columns.name = None
    return encoded.reset_index()


def merge_medications(
    clinical: pd.DataFrame, medication: pd.DataFrame, patient_id: str
) -> pd.DataFrame:
    """Merge one binary medication-group feature per patient into clinical data."""
    grouped = aggregate_medications(assign_drug_groups(medication), patient_id)
    merged = clinical.merge(grouped, on=patient_id, how="left")
    drug_columns = [column for column in grouped if column != patient_id]
    merged[drug_columns] = merged[drug_columns].fillna(0).astype(int)
    return merged.drop(columns=patient_id)
