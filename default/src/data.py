import pandas as pd

from default.constants import DATA_DIR, FEATURES_DTYPES


def load_dataset(kind: str = "processed") -> pd.DataFrame:
    """Load a dataset"""
    assert kind in {"raw", "interim", "processed"}

    kwargs = {"filepath_or_buffer": f"{DATA_DIR}/{kind}.csv", "sep": ";", "engine": "c"}
    if kind == "raw":
        kwargs["dtype"] = FEATURES_DTYPES

    try:
        data = pd.read_csv(**kwargs)
    except FileNotFoundError:
        data = pd.DataFrame()

    return data


def prepare_data() -> pd.DataFrame:
    """Prepare a raw dataset, store it and return the prepared data"""
    df = load_dataset("raw")

    df.drop("uuid", axis=1, inplace=True)  # irrelevant for modeling
    for col, series in df.select_dtypes("category").iteritems():
        df[col] = series.cat.codes

    df.to_csv(f"{DATA_DIR}/interim.csv", sep=";", index=False)

    return df


if __name__ == "__main__":
    raw = load_dataset("raw")
    adj = prepare_data()
