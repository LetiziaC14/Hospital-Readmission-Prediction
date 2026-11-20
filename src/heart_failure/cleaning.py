"""Clinical cleaning rules transferred from the original notebook."""

import numpy as np
import pandas as pd

TARGET = "re.admission.within.6.months"
READMISSION_TIME = "re.admission.time..days.from.admission."

TARGET_RELATED_COLUMNS = [
    "death.within.28.days", "death.within.3.months", "death.within.6.months",
    "re.admission.within.28.days", "re.admission.within.3.months",
    "time.of.death..days.from.admission.", READMISSION_TIME,
    "return.to.emergency.department.within.6.months",
    "time.to.emergency.department.within.6.months",
]

CLINICALLY_REDUNDANT_COLUMNS = [
    "total.bilirubin", "total.protein", "monocyte.count", "basophil.count",
    "eosinophil.count", "neutrophil.count", "creatine.kinase.isoenzyme",
    "lactate.dehydrogenase", "systolic.blood.pressure",
    "diastolic.blood.pressure", "height",
]

CORRELATED_COLUMNS = [
    "prothrombin.time.ratio", "mean.hemoglobin.volume", "movement", "cholesterol"
]

CATEGORICAL_COLUMNS = [
    "DestinationDischarge", "admission.ward", "admission.way", "occupation",
    "discharge.department", "gender", "type.of.heart.failure",
    "NYHA.cardiac.function.classification", "Killip.grade",
    "type.II.respiratory.failure", "consciousness", "respiratory.support.",
    "oxygen.inhalation", "outcome.during.hospitalization", "ageCat",
]

OUTLIER_LIMITS = {
    "map": {"eq": 0},
    "weight": {"le": 10.0},
    "BMI": {"lt": 5.0, "gt": 80.0},
    "cystatin": {"gt": 6.0},
    "international.normalized.ratio": {"gt": 5.0},
    "prothrombin.time.ratio": {"gt": 5.0},
    "potassium": {"gt": 7.0},
    "urea": {"gt": 35.0},
    "triglyceride": {"gt": 13.0},
    "globulin": {"lt": 10.0, "gt": 55.0},
    "D.dimer": {"gt": 30.0},
    "activated.partial.thromboplastin.time": {"gt": 90.0},
    "thrombin.time": {"gt": 100.0},
    "indirect.bilirubin": {"gt": 80.0},
    "direct.bilirubin": {"gt": 100.0},
    "creatinine.enzymatic.method": {"gt": 600.0},
    "alkaline.phosphatase": {"gt": 300.0},
    "hydroxybutyrate.dehydrogenase": {"gt": 900.0},
    "glutamic.oxaloacetic.transaminase": {"gt": 1000.0},
    "creatine.kinase": {"gt": 1500.0},
    "glutamic.pyruvic.transaminase": {"gt": 1000.0},
}


def drop_columns_above_missingness(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Drop columns whose missing proportion exceeds ``threshold``."""
    return df.drop(columns=df.columns[df.isna().mean() > threshold])


def drop_rows_above_missingness(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Drop rows whose missing proportion exceeds ``threshold``."""
    return df.loc[df.isna().mean(axis=1) <= threshold].copy()


def clean_target_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the death/readmission consistency rules used in the notebook."""
    result = df.copy()
    in_hospital_death = (result["DestinationDischarge"] == "Died") | (
        result["outcome.during.hospitalization"] == "Dead"
    )
    result = result.loc[~in_hospital_death].copy()

    death_columns = ["death.within.28.days", "death.within.3.months", "death.within.6.months"]
    died_without_readmission = result[death_columns].eq(1).any(axis=1) & result[READMISSION_TIME].isna()
    result = result.loc[~died_without_readmission].copy()

    not_readmitted = (
        result["re.admission.within.28.days"].eq(0)
        & result["re.admission.within.3.months"].eq(0)
        & result[TARGET].eq(0)
    )
    result = result.loc[~(not_readmitted & result[READMISSION_TIME].lt(180))].copy()
    not_readmitted = (
        result["re.admission.within.28.days"].eq(0)
        & result["re.admission.within.3.months"].eq(0)
        & result[TARGET].eq(0)
    )
    result.loc[not_readmitted & result[READMISSION_TIME].isna(), READMISSION_TIME] = -1
    result = result.dropna(subset=[READMISSION_TIME])
    return result.drop(columns=TARGET_RELATED_COLUMNS)


def replace_clinical_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the extreme values identified in the notebook with missing values."""
    result = df.copy()
    for column, limits in OUTLIER_LIMITS.items():
        if column not in result:
            continue
        invalid = pd.Series(False, index=result.index)
        if "eq" in limits:
            invalid |= result[column].eq(limits["eq"])
        if "le" in limits:
            invalid |= result[column].le(limits["le"])
        if "lt" in limits:
            invalid |= result[column].lt(limits["lt"])
        if "gt" in limits:
            invalid |= result[column].gt(limits["gt"])
        result.loc[invalid, column] = np.nan
    return result


def drop_low_variance_features(
    df: pd.DataFrame, numeric_threshold: float = 0.1, categorical_frequency: float = 0.9
) -> pd.DataFrame:
    """Apply the numeric and categorical near-zero-variance filters from the notebook."""
    result = df.copy()
    variance = result.var(numeric_only=True)
    numeric_drop = variance[(variance < numeric_threshold) & (variance.index != TARGET)].index
    result = result.drop(columns=numeric_drop)
    categorical_drop = []
    for column in result.select_dtypes(exclude="number"):
        frequencies = result[column].value_counts(normalize=True, dropna=False)
        if not frequencies.empty and frequencies.iloc[0] >= categorical_frequency:
            categorical_drop.append(column)
    return result.drop(columns=categorical_drop)


def clean_clinical_data(
    df: pd.DataFrame, column_missing_threshold: float = 0.25, row_missing_threshold: float = 0.20
) -> pd.DataFrame:
    """Run the deterministic cleaning section before train/test splitting."""
    result = clean_target_consistency(df)
    result = drop_columns_above_missingness(result, column_missing_threshold)
    result = result.convert_dtypes()
    result = result.astype({c: "category" for c in CATEGORICAL_COLUMNS if c in result})
    result = result.drop(columns=[c for c in CLINICALLY_REDUNDANT_COLUMNS if c in result])
    result = replace_clinical_outliers(result)
    result = drop_rows_above_missingness(result, row_missing_threshold)
    result = drop_low_variance_features(result)
    result = result.drop(columns=[c for c in CORRELATED_COLUMNS if c in result])
    return result
