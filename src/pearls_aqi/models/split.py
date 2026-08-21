"""Time-based chronological dataset splitting without random shuffling."""

from typing import Tuple

import pandas as pd

from pearls_aqi.domain.exceptions import ModelTrainingError


def chronological_train_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    time_col: str = "event_time_utc",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically into train and test sets."""
    if df.empty:
        raise ModelTrainingError("Cannot split empty DataFrame.")

    df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
    n = len(df_sorted)
    split_idx = int(n * train_ratio)

    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()

    return train_df, test_df
