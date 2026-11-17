import pandas as pd

from heart_failure.medication import aggregate_medications


def test_aggregate_medications_creates_one_row_per_patient():
    medication = pd.DataFrame({"inpatient.number": [1, 1, 2], "Drug_group": ["A", "B", "A"]})
    result = aggregate_medications(medication, "inpatient.number")
    assert len(result) == 2
    assert result.loc[result["inpatient.number"] == 1, "A"].item() == 1
