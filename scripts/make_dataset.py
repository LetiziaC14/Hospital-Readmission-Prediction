"""Create the cleaned pre-split dataset using the notebook's deterministic rules."""

import json
from pathlib import Path

from heart_failure.cleaning import clean_clinical_data
from heart_failure.config import load_config
from heart_failure.data_loading import load_raw_data, validate_patient_ids
from heart_failure.medication import merge_medications


def main() -> None:
    config = load_config()
    paths, data = config["paths"], config["data"]
    clinical, medication = load_raw_data(paths["raw_clinical"], paths["raw_medication"])
    validate_patient_ids(clinical, medication, data["patient_id_column"])
    merged = merge_medications(clinical, medication, data["patient_id_column"])
    processed = clean_clinical_data(
        merged,
        column_missing_threshold=data["missing_column_threshold"],
        row_missing_threshold=data["missing_row_threshold"],
    )
    output = Path(paths["processed_data"])
    output.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output, index=True)
    audit = {
        "clinical_rows": len(clinical),
        "merged_columns": merged.shape[1],
        "processed_rows": processed.shape[0],
        "processed_columns": processed.shape[1],
    }
    with (output.parent / "dataset_summary.json").open("w", encoding="utf-8") as file:
        json.dump(audit, file, indent=2)
    print(f"Saved {processed.shape[0]} rows and {processed.shape[1]} columns to {output}")


if __name__ == "__main__":
    main()
