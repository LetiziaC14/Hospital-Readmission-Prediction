import pandas as pd

from heart_failure.cleaning import drop_columns_above_missingness, drop_rows_above_missingness


def test_drop_columns_above_missingness():
    frame = pd.DataFrame({"keep": [1, 2], "drop": [None, 2]})
    assert list(drop_columns_above_missingness(frame, 0.4).columns) == ["keep"]


def test_drop_rows_above_missingness():
    frame = pd.DataFrame({"a": [1, None], "b": [2, None]})
    assert len(drop_rows_above_missingness(frame, 0.5)) == 1
