from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from default.constants import DATA_DIR, FEATURES_DTYPES
from default.settings import MAX_MISSING_FEATURE_PCT


__all__ = (
    "load_dataset",
    "preprocess_data",
    "setup_features_and_labels",
    "train_valid_test_split",
)


# Globals
VALID_DATASETS = {"raw", "interim", "preprocessed", "processed"}


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


def preprocess_data():
    """Adjust and scale the raw dataset"""
    df = load_dataset("raw")

    to_drop = df.isna().mean(axis=0).ge(MAX_MISSING_FEATURE_PCT)
    to_drop = ["uuid", *to_drop[to_drop].index]
    df.drop(to_drop, axis=1, inplace=True)

    numerical_cols = df.select_dtypes(np.number).columns

    # Convert categorial features
    for col, series in df.select_dtypes("category").iteritems():
        df[col] = series.cat.codes.replace(-1, np.nan)
        df[col].fillna(df[col].mode(), inplace=True)
    df.to_csv(f"{DATA_DIR}/interim.csv", sep=";", index=False)

    # Scale numerical features and downcast
    scaler = preprocessing.PowerTransformer()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    df.to_csv(f"{DATA_DIR}/preprocessed.csv", sep=";", index=False)


def setup_features_and_labels(
    n_samples: int = 0,
    kind: str = "preprocessed",
    label_col: str = "default",
    fillna_X: float = -1,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load dataset, clean labels and fill missing feature values"""
    df = load_dataset(kind)
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
    _ = preprocess_data()
    raw = load_dataset("raw")
    interim = load_dataset("interim")
    processed = load_dataset("preprocessed")
