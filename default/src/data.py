import numpy as np
import pandas as pd
from sklearn import preprocessing

from default.constants import DATA_DIR, FEATURES_DTYPES


def load_dataset(kind: str = "processed") -> pd.DataFrame:
    """Load a dataset"""
    assert kind in {"raw", "interim", "preprocessed", "processed"}

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


def prepare_data():
    """Adjust and scale the raw dataset"""
    df = load_dataset("raw")

    df.drop("uuid", axis=1, inplace=True)  # irrelevant for modeling

    numerical_cols = df.select_dtypes(np.number).columns

    # Convert categorial features
    for col, series in df.select_dtypes("category").iteritems():
        df[col] = series.cat.codes.replace(-1, np.nan)
    df.to_csv(f"{DATA_DIR}/interim.csv", sep=";", index=False)

    # Scale numerical features and downcast
    scaler = preprocessing.StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    df.to_csv(f"{DATA_DIR}/preprocessed.csv", sep=";", index=False)


if __name__ == "__main__":
    _ = prepare_data()
    raw = load_dataset("raw")
    interim = load_dataset("interim")
    processed = load_dataset("preprocessed")
