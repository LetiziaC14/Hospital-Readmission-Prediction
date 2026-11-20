import pandas as pd

from heart_failure.features import NotebookPreprocessor


def test_preprocessor_aligns_unseen_test_dummy_columns():
    train = pd.DataFrame(
        {
            "gender": ["Male", "Female", "Male"],
            "DestinationDischarge": ["Unknown", "Home", "Home"],
            "value": [1.0, 2.0, None],
        }
    )
    test = pd.DataFrame(
        {"gender": ["Female"], "DestinationDischarge": ["Other"], "value": [3.0]}
    )
    preprocessor = NotebookPreprocessor(seed=42)
    transformed_train = preprocessor.fit_transform(train)
    transformed_test = preprocessor.transform(test)
    assert transformed_test.columns.tolist() == transformed_train.columns.tolist()
