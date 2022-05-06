from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from default.constants import DATA_DIR, FEATURES_DTYPES
from default.settings import MAX_MISSING_FEATURE_PCT


__all__ = (
    "load_dataset",
    "parse_features_from_dict",
    "prepare_data",
    "setup_features_and_labels",
    "train_valid_test_split",
)


# Globals
VALID_DATASETS = {"raw", "interim", "processed"}


def load_dataset(kind: str = "processed") -> pd.DataFrame:
    """Load a dataset"""
    assert kind in VALID_DATASETS, "Invalid dataset"

    kwargs = {"filepath_or_buffer": f"{DATA_DIR}/{kind}.csv", "sep": ";", "engine": "c"}
    if kind == "raw":
        kwargs["dtype"] = FEATURES_DTYPES

    try:
        data = pd.read_csv(**kwargs)
    except FileNotFoundError:
        data = pd.DataFrame()

    if kind in {"preprocessed", "processed"}:
        data = data.apply(pd.to_numeric, downcast="float", errors="ignore")

    return data


def parse_features_from_dict(data: dict, fillna: float = -1.0) -> pd.DataFrame:
    """Turns data received from API request into a parsed and processed feature vector"""

    raw_df = pd.Series(data).to_frame().T
    to_drop = set()
    for feat, dtype in FEATURES_DTYPES.items():
        if feat in raw_df:
            raw_df[feat] = raw_df[feat].astype(dtype)
        else:
            to_drop.add(feat)

    raw_df.drop(to_drop, axis=1, errors="ignore", inplace=True)
    df = prepare_data(raw_df, is_prediction=True)
    df = df.drop("default", axis=1, errors="ignore").fillna(fillna)

    return df  # shape: (n_feats, 1)


def prepare_data(
    df: Optional[pd.DataFrame] = None, filename: str = "", is_prediction: bool = False
) -> pd.DataFrame:
    """Drop unwanted features and scale/encode the remaining ones"""
    if df is None:
        df = load_dataset("raw")

    if not is_prediction:
        to_drop = df.isna().mean(axis=0).ge(MAX_MISSING_FEATURE_PCT)
        to_drop = ["uuid", *to_drop[to_drop].index]
    else:
        to_drop = ["uuid"]
    df.drop(to_drop, axis=1, inplace=True, errors="ignore")

    numerical_cols = df.select_dtypes(np.number).columns

    # Convert categorial features
    for col, series in df.select_dtypes("category").iteritems():
        if col != "default":
            series = series.cat.codes.replace(-1, np.nan)
            fill = -1 if is_prediction else series.mode().get(0)
            df[col] = series.fillna(fill)

    # Fill numerical features
    for col in numerical_cols:
        fill = -1 if is_prediction else df[col].median()
        df[col].fillna(fill, inplace=True)

    if filename:
        if filename.endswith(".csv"):
            filename = filename[:-4]
        df.to_csv(f"{DATA_DIR}/{filename}.csv", sep=";", index=False)

    return df


def setup_features_and_labels(
    n_samples: int = 0,
    kind: str = "interim",
    label_col: str = "default",
    drop_missing: bool = True,
    fillna_X: float = -1,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load dataset, clean labels and fill missing feature values"""
    df = load_dataset(kind)

    if drop_missing:
        df = df.loc[df.default.notnull()]

    if n_samples:
        df = df.sample(n_samples, random_state=random_state)

    X = df.drop(label_col, axis=1).fillna(fillna_X).to_numpy()
    y = df[label_col].to_numpy()

    return df, X, y


def train_valid_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> tuple[np.ndarray]:
    """Utility to split data into train, validation and test sets"""
    X_train, X_rem, y_train, y_rem = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_rem, y_rem, test_size=0.5, random_state=random_state
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test


if __name__ == "__main__":
    _ = prepare_data(filename="interim")
    raw = load_dataset("raw")
    interim = load_dataset("interim")
