"""Create the processed dataset from the raw source tables."""

from pathlib import Path

from heart_failure.config import load_config
from heart_failure.data_loading import load_raw_data, validate_patient_ids


def main() -> None:
    config = load_config()
    paths, data = config["paths"], config["data"]
    clinical, medication = load_raw_data(paths["raw_clinical"], paths["raw_medication"])
    validate_patient_ids(clinical, medication, data["patient_id_column"])
    raise NotImplementedError(
        "Migrate and validate the medication merge and clinical cleaning rules from the notebook first."
    )


if __name__ == "__main__":
    main()
