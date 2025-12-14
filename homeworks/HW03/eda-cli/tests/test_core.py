import pytest
import pandas as pd
from eda_cli.core import compute_quality_flags, summarize_dataset, missing_table


def test_has_constant_columns():
    df = pd.DataFrame({
        'const_col': [42, 42, 42],
        'numeric_col': [1, 2, 3],
        'categorical_col': ['A', 'B', 'C']
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(df, summary, missing_df)

    assert flags["has_constant_columns"] is True
    assert flags["has_many_zero_values"] is False




def test_compute_quality_flags_basic_no_flags():
    df = pd.DataFrame({
        "col_a": list(range(100)),
        "col_b": list(range(100, 200)),
        "col_c": [f"val_{i}" for i in range(100)]
    })

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(df, summary, missing_df)

    assert flags["has_constant_columns"] is False
    assert flags["has_many_zero_values"] is False
    assert flags["too_few_rows"] is False
    expected_score = 1.0
    assert flags["quality_score"] == pytest.approx(expected_score, abs=1e-2)
