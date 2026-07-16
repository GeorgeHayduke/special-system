"""
Starter tests for the profiler.

Two of these are written to FAIL on a fresh checkout (pandas>=2.0) because
they exercise the real DataFrame.append() bugs in profiler/afd_profile.py.
That's intentional -- they're the seed for the TDD and multi-file-refactor
exercises in the Cursor tutorial. Run `pytest -v` to see the failures, then
work the fix with Cursor (see .cursor/rules/profiler.mdc and
specs/json_output_mode.md for context).
"""

import pandas as pd
import pytest

from profiler.afd_profile import (
    check_if_datetime_as_object_feature,
    check_if_nlp_feature,
    get_config,
    get_dataframe,
    get_overview,
    get_stats,
)

CONFIG_PATH = "config.sample.json"


@pytest.fixture
def config():
    cfg = get_config(CONFIG_PATH)
    cfg["input_file"] = "sample_data/transactions.csv"
    return cfg


@pytest.fixture
def df(config):
    return get_dataframe(config)


def test_get_overview_flags_small_dataset(config, df):
    """Passes today: get_overview() correctly warns when a dataset is
    under the 10,000-row minimum needed to train a model."""
    _, stats = get_overview(config, df)
    assert "Record count" in stats["overview_msg"]
    assert "10,000" in stats["overview_msg"]["Record count"]


def test_datetime_as_object_detection():
    """FAILS on pandas>=2.0: check_if_datetime_as_object_feature() calls
    DataFrame.append() internally, which was removed in pandas 2.0."""
    dates = pd.Series(["2026-05-01 04:50:00", "2026-05-02 17:05:00", "2026-05-03 10:36:00"] * 50)
    assert check_if_datetime_as_object_feature(dates) is True


def test_get_stats_handles_email_column(config, df):
    """FAILS on pandas>=2.0: get_stats() calls DataFrame.append() when it
    detects an EMAIL_ADDRESS column (to add the synthetic _EMAIL_DOMAIN
    row to the dtype table)."""
    _, df_stats, _ = get_stats(config, df)
    assert "customer_email" in df_stats["_column"].values


def test_merchant_category_misclassified_as_text(config, df):
    """Documents a real bug: three-word merchant category labels trip the
    NLP heuristic (unique_ratio > 0.01 and avg_words >= 3) and get
    misclassified as free text instead of a category. This test currently
    PASSES because it asserts the buggy behavior -- flip the assertion
    once check_if_nlp_feature() is fixed to use a smarter heuristic
    (e.g. also checking unique_ratio against a higher ceiling)."""
    assert check_if_nlp_feature(df["merchant_category"]) is True
