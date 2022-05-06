import joblib

from default.constants import ROOT_DIR
from default.src.data import load_dataset, prepare_data


# Settings
# ----
MODEL_FILENAME = "random_forest_v1"
DATASET = "raw"


if __name__ == "__main__":
    # Setup
    # ----
    df = load_dataset(DATASET)
    missing = df.default.isna()
    missing = missing[missing].index

    clf = joblib.load(f"{ROOT_DIR}/models/{MODEL_FILENAME}.joblib")

    # Forecast
    # ----
    raw_df = df.loc[missing]
    interim_df = prepare_data(df, is_prediction=True)
    X = interim_df.drop("default", axis=1).fillna(-1).to_numpy()
    y_prob = clf.predict_proba(X)

    final = pd.Series(y_prob, index=raw_df.uuid)
    breakpoint()
    final.to_csv(f"{ROOT_DIR}/data/output.csv", sep=";")
